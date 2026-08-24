#!/usr/bin/env julia
#
# Regenerate the dashboard payload by replaying the full pipeline.
#
#   julia --project=.. generate.jl [destination.json]
#
# Two scenarios, one with a genuine edge and one without, so the page can show
# both what the system trades and what it refuses.

using BayesTrade, Dates, Statistics, Distributions, JSON3

function scenario(name, blurb, process; n_bars, seed, warmup, symbol)
    series = generate_series(process; symbol=symbol, n_bars=n_bars, seed=seed, start=Date(2019,1,1))
    engine = FeatureEngine(InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]))
    ex = build_training_set(engine, symbol; horizon_bars=1)
    rep = replay((() -> BayesianReturnModel([:log_return_1]; horizon_bars=1),
                  () -> BayesianVolatilityModel(; horizon_bars=1),
                  () -> MarketRegimeModel(; horizon_bars=1)), ex;
                 config=ReplayConfig(warmup=warmup, refit_every=50))

    limits = RiskLimits()
    broker = PaperBroker(starting_cash=1_000_000.0)
    bars = Dict(b.timestamp => b for b in series.bars)
    start_eq = 1_000_000.0
    peak = start_eq
    rows = Vector{Dict{String,Any}}()
    trades = 0
    gate_counts = Dict{String,Int}()
    reason_counts = Dict{String,Int}()

    for rec in rep.records
        bar = get(bars, rec.as_of, nothing)
        bar === nothing && continue
        q = Quote(symbol, rec.as_of, bar.close; volume=bar.volume)
        mark_to_market!(broker, [q])
        eq = equity(broker)
        peak = max(peak, eq)
        book = portfolio(broker; peak_equity=peak, day_start_equity=eq, as_of=rec.as_of)

        intent = decide(rec.prediction, limits)
        ruling = review(intent, book, limits; sector="energy",
                        annualised_volatility=annualise(std(rec.prediction)),
                        daily_turnover=bar.volume*bar.close)
        filled = 0.0
        if approved(ruling)
            order = order_from_ruling(broker, ruling, q, eq)
            if order !== nothing
                r = place_order!(broker, order, q)
                if was_filled(r); trades += 1; filled = r.fill.quantity * (is_buy(order) ? 1 : -1); end
            end
        end
        intent.reason === nothing || (reason_counts[slug(intent.reason)] = get(reason_counts, slug(intent.reason), 0)+1)
        for c in failures(ruling); gate_counts[string(c.name)] = get(gate_counts, string(c.name), 0)+1; end

        push!(rows, Dict{String,Any}(
            "t"=>string(Date(rec.as_of)), "eq"=>round(eq, digits=2),
            "price"=>round(bar.close, digits=2),
            "mean"=>round(mean(rec.prediction), digits=6), "sd"=>round(std(rec.prediction), digits=6),
            "lo"=>round(quantile(rec.prediction.distribution,0.05), digits=6),
            "hi"=>round(quantile(rec.prediction.distribution,0.95), digits=6),
            "out"=>round(rec.outcome, digits=6),
            "pup"=>round(probability_positive(rec.prediction), digits=4),
            "epi"=>round(epistemic_share(rec.prediction), digits=4),
            "act"=>slug(intent.action),
            "why"=>intent.reason === nothing ? nothing : slug(intent.reason),
            "req"=>round(ruling.requested_weight, digits=5),
            "app"=>round(ruling.approved_weight, digits=5),
            "fill"=>round(filled, digits=2),
            "w"=>round.(collect(rec.prediction.weights.probabilities), digits=4)))
    end

    final_eq = equity(broker)
    grid = collect(range(-0.09, 0.09; length=41))
    step = max(1, length(rep.records) ÷ 240)
    idx = collect(1:step:length(rep.records))
    ridge = [[round(pdf(rep.records[i].prediction.distribution, g), digits=4) for g in grid] for i in idx]

    cal = rep.calibration
    return Dict{String,Any}(
        "key"=>name, "blurb"=>blurb, "symbol"=>symbol,
        "n_scored"=>length(rep.records), "trades"=>trades,
        "start_equity"=>start_eq, "final_equity"=>round(final_eq, digits=2),
        "return_pct"=>round((final_eq/start_eq - 1)*100, digits=3),
        "models"=>[slug(n) for n in rep.reliability.names],
        "reliabilities"=>round.(reliabilities(rep.reliability), digits=4),
        "mean_log_scores"=>round.(mean_log_scores(rep.reliability), digits=4),
        "calibration"=>Dict{String,Any}(
            "interval_error"=>round(interval_calibration_error(cal), digits=5),
            "brier"=>round(cal.brier_score, digits=4), "ks"=>round(cal.pit_ks_statistic, digits=4),
            "log_score"=>round(cal.mean_log_score, digits=4),
            "sharpness"=>round(cal.sharpness, digits=5), "bias"=>round(cal.bias, digits=6),
            "overconfident"=>is_overconfident(cal),
            "coverage"=>[Dict("level"=>p.level, "empirical"=>round(p.empirical, digits=4)) for p in cal.coverage]),
        "reasons"=>reason_counts, "gates"=>gate_counts,
        "grid"=>round.(grid, digits=4), "ridge"=>ridge,
        "ridge_idx"=>idx .- 1,
        "rows"=>rows)
end

out = Dict{String,Any}(
  "generated"=>string(Date(2026,8,25)),
  "limits"=>Dict("max_position_weight"=>0.05,"risk_budget_per_trade"=>0.005,
                 "min_probability_positive"=>0.6,"max_probability_large_loss"=>0.15,
                 "max_model_uncertainty"=>0.6,"max_portfolio_exposure"=>0.6),
  "scenarios"=>[
    scenario("edge","Returns carry genuine one-bar autocorrelation. The system finds it, trades it, and pays every cost.",
             AR1Returns(phi=0.55, annual_drift=0.0); n_bars=1800, seed=7, warmup=700, symbol="SYNTH-AR1"),
    scenario("noise","Regime-switching returns with no exploitable drift. The system declines every single bar.",
             RegimeSwitchingReturns(); n_bars=2600, seed=11, warmup=800, symbol="RELIANCE"),
  ])

const DEST = isempty(ARGS) ? joinpath(@__DIR__, "payload.json") : ARGS[1]
open(DEST, "w") do io
    JSON3.write(io, out)
end
for s in out["scenarios"]
    println(s["key"], ": bars=", s["n_scored"], " trades=", s["trades"],
            " return=", s["return_pct"], "%  ice=", s["calibration"]["interval_error"],
            "  reasons=", s["reasons"], "  gates=", s["gates"])
end
println("wrote ", DEST, "  ", filesize(DEST), " bytes")
