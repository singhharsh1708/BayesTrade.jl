#!/usr/bin/env julia
#
#   julia --project=. examples/backtest.jl [scenario]
#
# Replays the whole pipeline over generated history and prints what it believed and what it
# decided. `scenario` is `edge` (returns with genuine autocorrelation) or `noise` (regime
# switching with none). Defaults to both.

using BayesTrade, Dates, Printf, Statistics

function run_scenario(name::String, process, n_bars::Int, seed::Int, warmup::Int)
    series = generate_series(
        process; symbol = "SYNTH", n_bars = n_bars, seed = seed, start = Date(2019, 1, 1),
    )
    engine = FeatureEngine(
        InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, "SYNTH"; horizon_bars = 1)

    report = replay(
        (
            () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
            () -> MarketRegimeModel(; horizon_bars = 1),
        ),
        examples; config = ReplayConfig(warmup = warmup, refit_every = 50),
    )

    limits = RiskLimits()
    book = Portfolio(equity = 1.0e6, cash = 1.0e6, as_of = last(examples).features.as_of)
    actions = Dict{String, Int}()
    for record in report.records
        intent = decide(record.prediction, limits)
        key = slug(intent.action)
        actions[key] = get(actions, key, 0) + 1
    end

    println("\n", "="^68)
    println(uppercase(name))
    println("="^68)
    println(summarise(report))
    println("\n  decisions")
    for (action, count) in sort(collect(actions); by = last, rev = true)
        @printf("    %-10s %6d\n", action, count)
    end
    return report
end

scenario = isempty(ARGS) ? "both" : ARGS[1]

if scenario in ("edge", "both")
    run_scenario(
        "a market with a real edge", AR1Returns(phi = 0.55, annual_drift = 0.0),
        1_800, 7, 700,
    )
end
if scenario in ("noise", "both")
    run_scenario("a market with none", RegimeSwitchingReturns(), 2_600, 11, 800)
end
