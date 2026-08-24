#!/usr/bin/env julia
#
#   julia --project=. examples/paper_session.jl [bars] [journal.jsonl]
#
# Drives the long-running session the way a live feed would: one tick at a time, no lookahead,
# refitting on a schedule, journalling every decision as it happens.
#
# This is the same code path a live session runs. The only difference is where the ticks come
# from, which is the whole point of the exercise.

using BayesTrade, Dates, Printf

const N_BARS = isempty(ARGS) ? 1_400 : parse(Int, ARGS[1])
const JOURNAL = length(ARGS) >= 2 ? ARGS[2] : nothing

series = generate_series(
    AR1Returns(phi = 0.55, annual_drift = 0.0);
    symbol = "SYNTH", n_bars = N_BARS, seed = 7, start = Date(2026, 1, 1),
)

session = PaperTradingSession(
    "SYNTH",
    (
        () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
        () -> BayesianVolatilityModel(; horizon_bars = 1),
        () -> MarketRegimeModel(; horizon_bars = 1),
    ),
    FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]);
    horizon_bars = 1, warmup = 700, refit_every = 50,
    interval = Day(1), interval_label = "1d", max_silence = Day(3),
    sector = "energy", journal = JOURNAL,
)

println("feeding ", length(series.bars), " bars one tick at a time")
for bar in series.bars
    on_tick!(session, Quote("SYNTH", bar.timestamp, bar.close; volume = bar.volume))
end
close_bar!(session)

report = session_report(session)
println("\n", "="^58)
@printf("%-22s %s\n", "symbol", report["symbol"])
@printf("%-22s %d\n", "bars seen", report["bars"])
@printf("%-22s %d\n", "refits", report["refits"])
println("-"^58)
@printf("%-22s %d\n", "predictions", report["predictions"])
@printf("%-22s %d\n", "  declined by gates", report["declined"])
@printf("%-22s %d\n", "  vetoed by risk", report["vetoed"])
@printf("%-22s %d\n", "  approved", report["approved"])
@printf("%-22s %d\n", "    of which reduced", report["reduced"])
@printf("%-22s %d\n", "fills", report["fills"])
@printf("%-22s %d\n", "skipped, feed quiet", report["stale_bars"])
@printf("%-22s %d of %d\n", "settled", report["settled"], report["predictions"])
println("-"^58)
@printf("%-22s %.2f\n", "equity", report["equity"])
@printf("%-22s %+.2f%%\n", "return, after costs", report["return_pct"])
@printf("%-22s %.2f%%\n", "peak drawdown", report["drawdown_pct"])
@printf("%-22s %d\n", "open positions", report["positions"])
println("-"^58)
for (index, name) in enumerate(report["models"])
    @printf("%-22s %6.2f%%\n", name, 100 * report["reliabilities"][index])
end
println("="^58)
JOURNAL === nothing || println("\njournal written to ", JOURNAL)
