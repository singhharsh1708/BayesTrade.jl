#!/usr/bin/env julia
#
#   julia --project=validation validation/calibration.jl
#
# Sections 9, 10 and 11 of the validation brief.
#
# The baseline reported the system as underconfident and the dashboard run reported it as
# overconfident, on the same generator with a different feature and model set. Both cannot be a
# property of the system. This sweeps the space and finds what actually drives the direction,
# before anything is adjusted.

include(joinpath(@__DIR__, "harness.jl"))

const LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

"""
    GENERATORS

Markets whose behaviour is known because it was specified, chosen so that each poses a different
problem: one with genuine predictability, one with none, one whose volatility moves on its own,
one that switches regime.
"""
const GENERATORS = (
    (name = "ar1_strong", process = AR1Returns(phi = 0.35, annual_drift = 0.05)),
    (name = "ar1_weak", process = AR1Returns(phi = 0.08, annual_drift = 0.05)),
    (name = "gaussian", process = GaussianReturns(annual_drift = 0.05, annual_volatility = 0.25)),
    (
        name = "stochastic_vol", process = StochasticVolatilityReturns(
            annual_drift = 0.05, annual_volatility = 0.25, persistence = 0.97,
            volatility_of_volatility = 0.35,
        ),
    ),
    (name = "regime_switching", process = RegimeSwitchingReturns()),
)

"""
    FEATURE_SETS

Configurations that differ in how much they ask of the models. The point of varying this is that
the two contradictory calibration results came from two different sets.
"""
const FEATURE_SETS = (
    (
        name = "lean",
        features = FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
        columns = [:log_return_1],
    ),
    (
        name = "standard",
        features = FeatureSet(
            Feature[LogReturn(1), LogReturn(5), RealisedVolatility(20), Momentum(10)],
        ),
        columns = [:log_return_1, :momentum_10_5],
    ),
    (
        name = "wide",
        features = FeatureSet(
            Feature[
                LogReturn(1), LogReturn(5), RealisedVolatility(20), Momentum(10),
                PriceZScore(20), DrawdownFromHigh(30), TrendSlope(20),
            ],
        ),
        columns = [:log_return_1, :log_return_5, :momentum_10_5, :price_zscore_20],
    ),
)

"""
    MODEL_SETS

One model, or three. The dashboard run used three and the baseline used three, so this is not
what separated them, but a pool of one is the clean case for reading a single model's calibration
without the pool in the way.
"""
model_sets(columns) = (
    (
        name = "return_only",
        factories = (() -> BayesianReturnModel(columns; horizon_bars = 1),),
    ),
    (
        name = "return_volatility",
        factories = (
            () -> BayesianReturnModel(columns; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
        ),
    ),
    (
        name = "all_three",
        factories = (
            () -> BayesianReturnModel(columns; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
            () -> MarketRegimeModel(; horizon_bars = 1),
        ),
    ),
)

"""
    calibrate(process, features, columns, factories; n_bars, seed)

One walk-forward and the calibration it produces, or `nothing` when the configuration cannot
produce enough scored predictions to say anything.
"""
function calibrate(process, features, columns, factories; n_bars = 2000, seed = 20260825)
    series = fixed_series(n_bars = n_bars, seed = seed, process = process)
    engine = FeatureEngine(InMemoryBarStore(series.bars), features)
    examples = build_training_set(engine, series.symbol; horizon_bars = 1)
    length(examples) < 700 && return nothing
    report = replay(
        factories, examples; config = ReplayConfig(warmup = 500, refit_every = 25),
    )
    isempty(report.records) && return nothing
    return (report = report, series = series, examples = examples)
end

summarise_calibration(calibration) = Dict{String, Any}(
    "n" => calibration.n,
    "interval_error" => rounded(interval_calibration_error(calibration)),
    "signed_interval_error" => rounded(
        mean(point.empirical - point.level for point in calibration.coverage),
    ),
    "expected_calibration_error" => rounded(calibration.expected_calibration_error),
    "brier_score" => rounded(calibration.brier_score),
    "pit_ks" => rounded(calibration.pit_ks_statistic),
    "mean_log_score" => rounded(calibration.mean_log_score),
    "sharpness" => rounded(calibration.sharpness),
    "bias" => rounded(calibration.bias),
    "overconfident" => is_overconfident(calibration),
    "coverage" => Dict{String, Any}(
        string(point.level) => rounded(point.empirical, 4)
            for point in calibration.coverage
    ),
)

"""
    direction(calibration)

Which way the miscalibration goes, as one word.

Signed rather than absolute, because the absolute interval error the package reports cannot tell
intervals that are too narrow from intervals that are too wide, and those are opposite problems
with opposite fixes.
"""
function direction(calibration)
    signed = mean(point.empirical - point.level for point in calibration.coverage)
    signed < -0.005 && return "overconfident"
    signed > 0.005 && return "underconfident"
    return "calibrated"
end

function sweep()
    heading("calibration across generators, features and model sets")
    rows = Vector{Dict{String, Any}}()
    @printf(
        "%-18s %-10s %-18s %8s %10s %10s %8s  %s\n",
        "generator", "features", "models", "n", "signed", "abs", "sharp", "verdict",
    )
    for generator in GENERATORS, feature_set in FEATURE_SETS
        for models in model_sets(feature_set.columns)
            result = calibrate(
                generator.process, feature_set.features, feature_set.columns,
                models.factories,
            )
            result === nothing && continue
            calibration = result.report.calibration
            signed = mean(
                point.empirical - point.level for point in calibration.coverage
            )
            row = merge(
                Dict{String, Any}(
                    "generator" => generator.name,
                    "features" => feature_set.name,
                    "models" => models.name,
                    "direction" => direction(calibration),
                ),
                summarise_calibration(calibration),
            )
            push!(rows, row)
            @printf(
                "%-18s %-10s %-18s %8d %+10.4f %10.4f %8.4f  %s\n",
                generator.name, feature_set.name, models.name, calibration.n,
                signed, interval_calibration_error(calibration),
                calibration.sharpness, direction(calibration),
            )
        end
    end
    return rows
end

"""
    by_regime(rows)

Calibration broken down by the market the bar was in, which is section 11.

Aggregate calibration can be excellent while the model is dangerously overconfident in exactly
the conditions that matter, and an average over a quiet market and a violent one describes
neither.
"""
function by_regime()
    heading("calibration by regime")
    process = RegimeSwitchingReturns()
    features = FeatureSet(
        Feature[LogReturn(1), LogReturn(5), RealisedVolatility(20), Momentum(10)],
    )
    result = calibrate(
        process, features, [:log_return_1, :momentum_10_5],
        (
            () -> BayesianReturnModel([:log_return_1, :momentum_10_5]; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
            () -> MarketRegimeModel(; horizon_bars = 1),
        );
        n_bars = 4000,
    )
    result === nothing && return Dict{String, Any}()

    # Realised volatility over the trailing window, split at its own terciles, which is a
    # classification of the market rather than of the model's opinion about it.
    records = result.report.records
    spreads = Float64[std(record.prediction) for record in records]
    low, high = quantile(spreads, 0.33), quantile(spreads, 0.67)

    # Every split is on something known before the bar, never on the outcome. Bucketing by
    # the realised return would condition on the quantity being predicted: each bucket's
    # outcomes are then truncated on one side, and a symmetric interval mis-covers them by
    # construction. That reads as a calibration failure and is an artefact of the split.
    buckets = Dict(
        "low_volatility" => Int[], "mid_volatility" => Int[], "high_volatility" => Int[],
        "predicted_up" => Int[], "predicted_down" => Int[],
    )
    for (index, record) in enumerate(records)
        spread = spreads[index]
        bucket = spread <= low ? "low_volatility" :
            spread >= high ? "high_volatility" : "mid_volatility"
        push!(buckets[bucket], index)
        push!(
            buckets[mean(record.prediction) >= 0 ? "predicted_up" : "predicted_down"],
            index,
        )
    end

    output = Dict{String, Any}()
    @printf("%-18s %8s %10s %10s %10s  %s\n", "bucket", "n", "signed", "abs", "sharp", "verdict")
    for name in (
            "low_volatility", "mid_volatility", "high_volatility",
            "predicted_up", "predicted_down",
        )
        indices = buckets[name]
        length(indices) < 100 && continue
        subset = assess(
            [records[index].prediction.distribution for index in indices],
            [records[index].outcome for index in indices];
            levels = LEVELS,
        )
        signed = mean(point.empirical - point.level for point in subset.coverage)
        output[name] = merge(
            summarise_calibration(subset), Dict{String, Any}("direction" => direction(subset)),
        )
        @printf(
            "%-18s %8d %+10.4f %10.4f %10.4f  %s\n",
            name, subset.n, signed, interval_calibration_error(subset),
            subset.sharpness, direction(subset),
        )
    end
    return output
end

function main()
    rows = sweep()

    heading("what drives the direction")
    for key in ("generator", "features", "models")
        tally = Dict{String, Dict{String, Int}}()
        for row in rows
            group = get!(tally, row[key], Dict{String, Int}())
            group[row["direction"]] = get(group, row["direction"], 0) + 1
        end
        println("\nby ", key, ":")
        for name in sort(collect(keys(tally)))
            counts = tally[name]
            @printf(
                "  %-18s over=%-3d under=%-3d calibrated=%-3d\n", name,
                get(counts, "overconfident", 0), get(counts, "underconfident", 0),
                get(counts, "calibrated", 0),
            )
        end
    end

    regimes = by_regime()

    record(
        "calibration",
        Dict{String, Any}(
            "recorded_at" => string(now(UTC)),
            "commit" => strip(read(`git rev-parse --short HEAD`, String)),
            "sweep" => rows,
            "by_regime" => regimes,
        ),
    )
    heading("written")
    println(joinpath(RESULTS, "calibration.json"))
    return nothing
end

main()
