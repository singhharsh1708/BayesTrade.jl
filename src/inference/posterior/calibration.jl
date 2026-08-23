"""
Calibration: checking whether the probabilities mean anything.

A model that says seventy percent and is right half the time is worse than useless, because
everything downstream sizes positions off that number. Sharpe ratio and hit rate say nothing
about this. A model can have an excellent backtest and be badly calibrated, and it will then
fail the moment position sizing starts trusting its confidence.

Two things are checked, because the system uses the posterior in two ways.

**The whole distribution.** Under a correctly specified model the probability integral
transform `u = F(y)` is uniform. Deviations say precisely what is wrong: a U-shaped histogram
means the intervals are too narrow, a hump in the middle means too wide, and a shift means
biased.

**The directional probability.** The decision engine reads `P(return > 0)`, so that number is
checked on its own terms with a reliability curve and a Brier score.

Sharpness is reported alongside. A model that always predicts the unconditional distribution is
perfectly calibrated and worth nothing, so calibration is a constraint to satisfy rather than a
score to maximise.
"""

const DEFAULT_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
const DEFAULT_BINS = 10

"""
    CoveragePoint

Nominal against empirical coverage at one credible level.
"""
struct CoveragePoint
    level::Float64
    empirical::Float64
    n::Int
end

"""
    coverage_error(point)

Signed miss. Negative means the intervals were too narrow.
"""
coverage_error(point::CoveragePoint) = point.empirical - point.level

"""
    ReliabilityBin

One bucket of a reliability curve for the directional probability.
"""
struct ReliabilityBin
    lower::Float64
    upper::Float64
    n::Int
    mean_predicted::Float64
    observed_frequency::Float64
end

reliability_gap(bin::ReliabilityBin) = bin.observed_frequency - bin.mean_predicted

"""
    CalibrationReport

Everything worth knowing about whether a set of predictions can be trusted.
"""
struct CalibrationReport
    n::Int
    coverage::Vector{CoveragePoint}
    reliability::Vector{ReliabilityBin}
    pit::Vector{Float64}
    pit_ks_statistic::Float64
    brier_score::Float64
    expected_calibration_error::Float64
    mean_log_score::Float64
    sharpness::Float64
    bias::Float64
end

"""
    interval_calibration_error(report)

Mean absolute coverage miss across the levels checked.

This is the number the backtest report prints next to the Sharpe ratio. Under five percent is
usable; above ten, the posterior is not describing the outcomes.
"""
function interval_calibration_error(report::CalibrationReport)
    isempty(report.coverage) && return 0.0
    total = 0.0
    for point in report.coverage
        total += abs(coverage_error(point))
    end
    return total / length(report.coverage)
end

"""
    is_overconfident(report)

Whether intervals covered less often than they claimed, on average.

A signed average, so read it alongside [`interval_calibration_error`](@ref) rather than instead
of it. A model with a constant scale fitted to changing volatility is too wide in the middle
and too narrow in the tails, and those errors cancel here while every individual level is
wrong.
"""
function is_overconfident(report::CalibrationReport)
    isempty(report.coverage) && return false
    total = 0.0
    for point in report.coverage
        total += coverage_error(point)
    end
    return total < 0
end

function summarise(report::CalibrationReport)
    lines = String[
        string("calibration over ", report.n, " predictions"),
        @sprintf(
            "  interval error   %.3f%%  (%s)", 100 * interval_calibration_error(report),
            is_overconfident(report) ? "overconfident" : "underconfident"
        ),
        @sprintf("  direction ECE    %.3f%%", 100 * report.expected_calibration_error),
        @sprintf("  Brier score      %.4f", report.brier_score),
        @sprintf("  PIT KS           %.4f", report.pit_ks_statistic),
        @sprintf("  mean log score   %+.4f", report.mean_log_score),
        @sprintf("  sharpness        %.4f%%", 100 * report.sharpness),
        @sprintf("  bias             %+.4f%%", 100 * report.bias),
        "",
        "  level   nominal   empirical",
    ]
    for point in report.coverage
        push!(
            lines,
            @sprintf(
                "  %5.0f%%   %6.1f%%   %8.1f%%",
                100 * point.level, 100 * point.level, 100 * point.empirical
            ),
        )
    end
    return join(lines, '\n')
end

"""
    probability_integral_transform(predictives, outcomes)

`F_i(y_i)` for each prediction. Uniform if and only if the model is calibrated.
"""
function probability_integral_transform(predictives, outcomes)
    length(predictives) == length(outcomes) || throw(
        ArgumentError(
            string(
                length(predictives), " predictions against ", length(outcomes), " outcomes",
            ),
        ),
    )
    transformed = Vector{Float64}(undef, length(outcomes))
    for index in eachindex(transformed)
        transformed[index] = cdf(predictives[index], Float64(outcomes[index]))
    end
    return transformed
end

"""
    kolmogorov_smirnov_uniform(values)

Largest gap between the empirical distribution of `values` and the uniform.

Computed directly rather than through a test statistic with a p-value. A p-value on a few
thousand correlated daily predictions would be badly overstated, and the size of the deviation
is the useful number anyway.
"""
function kolmogorov_smirnov_uniform(values::AbstractVector{Float64})
    isempty(values) && return 0.0
    ordered = sort(values)
    n = length(ordered)
    largest = 0.0
    for index in 1:n
        step = index / n
        largest = max(largest, step - ordered[index], ordered[index] - (step - 1 / n))
    end
    return largest
end

"""
    coverage_curve(pit; levels)

How often the central credible interval at each level actually contained the outcome.
"""
function coverage_curve(pit::AbstractVector{Float64}, levels = DEFAULT_LEVELS)
    points = CoveragePoint[]
    for level in levels
        0 < level < 1 ||
            throw(ArgumentError(string("level must lie in (0, 1), got ", level)))
        tail = (1 - level) / 2
        inside = 0
        for value in pit
            (tail <= value <= 1 - tail) && (inside += 1)
        end
        empirical = isempty(pit) ? 0.0 : inside / length(pit)
        push!(points, CoveragePoint(Float64(level), empirical, length(pit)))
    end
    return points
end

"""
    reliability_curve(probabilities, outcomes; bins)

Predicted against realised frequency, bucketed.

Empty buckets are dropped rather than reported as zero. A bucket nothing landed in says nothing
about the model, and counting it as a perfect miss would be a lie in whichever direction the
arithmetic happened to fall.
"""
function reliability_curve(
        probabilities::AbstractVector{Float64}, outcomes::AbstractVector{Bool};
        bins::Integer = DEFAULT_BINS,
    )
    bins >= 1 || throw(ArgumentError(string("bins must be positive, got ", bins)))
    length(probabilities) == length(outcomes) || throw(
        ArgumentError(
            string(
                length(probabilities), " probabilities against ", length(outcomes),
                " outcomes",
            ),
        ),
    )

    curve = ReliabilityBin[]
    for index in 1:bins
        lower = (index - 1) / bins
        upper = index / bins
        count = 0
        predicted_total = 0.0
        observed_total = 0
        for position in eachindex(probabilities)
            value = probabilities[position]
            inside = index < bins ? (lower <= value < upper) : (lower <= value <= upper)
            inside || continue
            count += 1
            predicted_total += value
            outcomes[position] && (observed_total += 1)
        end
        count == 0 && continue
        push!(
            curve,
            ReliabilityBin(
                lower, upper, count, predicted_total / count, observed_total / count,
            ),
        )
    end
    return curve
end

"""
    expected_calibration_error(curve)

Weighted mean gap between predicted and realised frequency.
"""
function expected_calibration_error(curve::AbstractVector{ReliabilityBin})
    total = 0
    weighted = 0.0
    for bin in curve
        total += bin.n
        weighted += bin.n * abs(reliability_gap(bin))
    end
    return total == 0 ? 0.0 : weighted / total
end

"""
    brier_score(probabilities, outcomes)

Mean squared error of the directional probability. Lower is better; 0.25 is a coin.
"""
function brier_score(probabilities::AbstractVector{Float64}, outcomes::AbstractVector{Bool})
    length(probabilities) == length(outcomes) || throw(
        ArgumentError(
            string(
                length(probabilities), " probabilities against ", length(outcomes),
                " outcomes",
            ),
        ),
    )
    isempty(probabilities) && return 0.0
    total = 0.0
    for index in eachindex(probabilities)
        total += (probabilities[index] - (outcomes[index] ? 1.0 : 0.0))^2
    end
    return total / length(probabilities)
end

"""
    assess(predictives, outcomes; levels, bins)

Score a set of out-of-sample predictions against what actually happened.

In sample this measures nothing: a model scored on the data it was fitted to will look
calibrated whether or not it is. Use [`walk_forward`](@ref) to produce the inputs.
"""
function assess(
        predictives, outcomes;
        levels = DEFAULT_LEVELS, bins::Integer = DEFAULT_BINS,
    )
    length(predictives) == length(outcomes) || throw(
        ArgumentError(
            string(
                length(predictives), " predictions against ", length(outcomes), " outcomes",
            ),
        ),
    )
    isempty(predictives) && throw(ArgumentError("nothing to assess"))

    n = length(predictives)
    realised = Vector{Float64}(undef, n)
    directional = Vector{Float64}(undef, n)
    went_up = Vector{Bool}(undef, n)
    log_total = 0.0
    log_count = 0
    spread_total = 0.0
    spread_count = 0
    error_total = 0.0

    for index in 1:n
        distribution = predictives[index]
        outcome = Float64(outcomes[index])
        realised[index] = outcome
        directional[index] = probability_positive(distribution)
        went_up[index] = outcome > 0

        score = logpdf(distribution, outcome)
        isfinite(score) && (log_total += score; log_count += 1)

        spread = std(distribution)
        isfinite(spread) && (spread_total += spread; spread_count += 1)

        error_total += outcome - mean(distribution)
    end

    pit = probability_integral_transform(predictives, realised)
    curve = reliability_curve(directional, went_up; bins = bins)

    return CalibrationReport(
        n,
        coverage_curve(pit, levels),
        curve,
        pit,
        kolmogorov_smirnov_uniform(pit),
        brier_score(directional, went_up),
        expected_calibration_error(curve),
        log_count == 0 ? -Inf : log_total / log_count,
        spread_count == 0 ? Inf : spread_total / spread_count,
        error_total / n,
    )
end
