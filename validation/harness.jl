"""
Shared scaffolding for the validation runs.

Everything here is deterministic. A validation number that moves between runs for reasons
nobody controls is not a measurement, it is noise with a decimal point, and the first time it
shifts somebody will spend a day looking for a regression that never happened.
"""

using BayesTrade
using Dates, Distributions, JSON3, Printf, Random, Statistics

const RESULTS = joinpath(@__DIR__, "results")

"""
    fixed_series(; n_bars, seed, process, symbol)

The dataset every run shares. Same seed, same bars, same everything.
"""
function fixed_series(;
        n_bars::Int = 2000, seed::Int = 20260825, symbol::AbstractString = "VALID",
        process = AR1Returns(phi = 0.35, annual_drift = 0.05),
    )
    return generate_series(
        process; symbol = symbol, n_bars = n_bars, seed = seed,
        start = Date(2016, 1, 4),
    )
end

standard_features() = FeatureSet(
    Feature[
        LogReturn(1), LogReturn(5), RealisedVolatility(20), Momentum(10),
    ],
)

"""
    training_set(series; horizon_bars)

Features and labels, built the way every entry point in the package builds them.
"""
function training_set(series; horizon_bars::Int = 1, features = standard_features())
    engine = FeatureEngine(InMemoryBarStore(series.bars), features)
    return build_training_set(engine, series.symbol; horizon_bars = horizon_bars)
end

standard_factories() = (
    () -> BayesianReturnModel([:log_return_1, :momentum_10_5]; horizon_bars = 1),
    () -> BayesianVolatilityModel(; horizon_bars = 1),
    () -> MarketRegimeModel(; horizon_bars = 1),
)

"""
    timed(work; samples, warmup)

Median seconds and allocated bytes for one call.

The median rather than the mean: one garbage collection during a run is a value ten times the
others, and it drags a mean somewhere no individual call ever went.
"""
function timed(work; samples::Int = 7, warmup::Int = 2)
    for _ in 1:warmup
        work()
    end
    times = Float64[]
    for _ in 1:samples
        push!(times, @elapsed work())
    end
    return (seconds = median(times), bytes = @allocated(work()))
end

"""
    record(name, payload)

Write one result file, and print it so a run is readable while it happens.
"""
function record(name::AbstractString, payload::AbstractDict)
    mkpath(RESULTS)
    path = joinpath(RESULTS, string(name, ".json"))
    open(path, "w") do handle
        JSON3.pretty(handle, payload)
        println(handle)
    end
    return path
end

rounded(value::Real, digits::Int = 6) =
    isfinite(value) ? round(Float64(value); digits = digits) : nothing

"""
    heading(text)

A visible boundary in a long run.
"""
function heading(text::AbstractString)
    println()
    println("="^74)
    println(uppercase(text))
    println("="^74)
    return nothing
end
