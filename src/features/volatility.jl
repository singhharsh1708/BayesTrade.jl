"""
Volatility features.

Every volatility here is annualised on the 252-day convention, so a close-to-close estimate
and a range estimate over the same window are directly comparable. Comparing them is the
point: they disagree in informative ways, and a single number would hide it.

Range estimators use the high and low, which carry information the close throws away. A day
that travelled two percent in each direction and finished flat has the same close-to-close
return as a day that did nothing, and a very different volatility. The cost is that a range
estimator only sees the part of the move the price walked through, so it reads low on a
market that gaps at the open. That is a real property rather than a defect, and the ratio
between the two estimates is itself a signal about how much of the risk arrives overnight.
"""

const GARMAN_KLASS_COEFFICIENT = 2 * log(2) - 1
const PARKINSON_COEFFICIENT = 4 * log(2)
const EWMA_VOLATILITY_WARMUP = 3

"""
    RealisedVolatility(window; annualised = true)

Annualised standard deviation of close-to-close log returns.

Uses the sample standard deviation about the window mean rather than the root mean square
about zero. Over short windows a strong drift would otherwise be counted as volatility, which
is exactly backwards: a steady climb is the low-risk case.
"""
struct RealisedVolatility <: Feature
    window::Int
    annualised::Bool

    function RealisedVolatility(window::Integer = 20; annualised::Bool = true)
        window >= 3 || throw(ArgumentError("window must be at least 3, got $window"))
        return new(Int(window), annualised)
    end
end

feature_name(feature::RealisedVolatility) = Symbol("volatility_", feature.window)
lookback(feature::RealisedVolatility) = feature.window

function compute(feature::RealisedVolatility, window::BarWindow)
    returns = view(window.log_returns, (length(window.log_returns) - feature.window + 1):length(window.log_returns))
    value = std(returns)
    return feature.annualised ? annualise(value) : value
end

"""
    EwmaVolatility(span; annualised = true)

Exponentially weighted volatility, weighting recent bars more heavily.

Seeded from a plain average of squared returns far enough back that the seed has decayed, for
the same reason the moving averages are: a value that depends on when the download started is
a property of the download, not of the market.
"""
struct EwmaVolatility <: Feature
    span::Int
    annualised::Bool

    function EwmaVolatility(span::Integer = 20; annualised::Bool = true)
        span >= 2 || throw(ArgumentError("span must be at least 2, got $span"))
        return new(Int(span), annualised)
    end
end

feature_name(feature::EwmaVolatility) = Symbol("ewma_volatility_", feature.span)
lookback(feature::EwmaVolatility) = feature.span * EWMA_VOLATILITY_WARMUP

function compute(feature::EwmaVolatility, window::BarWindow)
    span = lookback(feature)
    squared = Float64[
        value^2 for value in
            view(window.log_returns, (length(window.log_returns) - span + 1):length(window.log_returns))
    ]
    alpha = 2 / (feature.span + 1)
    level = mean(view(squared, 1:feature.span))
    for index in (feature.span + 1):length(squared)
        level += alpha * (squared[index] - level)
    end
    level <= 0 && return nothing
    value = sqrt(level)
    return feature.annualised ? annualise(value) : value
end

"""
    ParkinsonVolatility(window; annualised = true)

Annualised volatility from the high-low range.

Roughly five times more efficient than close-to-close at the same window, because the range of
a path carries more about its diffusion than its endpoints do. Assumes the high and low are
extremes of a continuous path, which is why the synthetic generator simulates an intraday
bridge rather than sprinkling wicks around the body.
"""
struct ParkinsonVolatility <: Feature
    window::Int
    annualised::Bool

    function ParkinsonVolatility(window::Integer = 20; annualised::Bool = true)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window), annualised)
    end
end

feature_name(feature::ParkinsonVolatility) = Symbol("parkinson_volatility_", feature.window)
lookback(feature::ParkinsonVolatility) = feature.window - 1

function compute(feature::ParkinsonVolatility, window::BarWindow)
    span = (length(window) - feature.window + 1):length(window)
    variance = mean(
        log(window.highs[i] / window.lows[i])^2 for i in span
    ) / PARKINSON_COEFFICIENT
    variance <= 0 && return nothing
    value = sqrt(variance)
    return feature.annualised ? annualise(value) : value
end

"""
    GarmanKlassVolatility(window; annualised = true)

Annualised volatility from the full open, high, low and close.

More efficient again than Parkinson, since it uses the open-to-close move as well as the
range.

The estimator is a difference of two terms, which invites the worry that it could come out
negative. It cannot, on a valid bar: the high is at or above both the open and the close and
the low at or below both, so the range term dominates the body term with a coefficient of at
least `0.5 - (2 log 2 - 1)`. The guard below is therefore only for a bar with no range at all,
where the honest answer is that nothing was observed.
"""
struct GarmanKlassVolatility <: Feature
    window::Int
    annualised::Bool

    function GarmanKlassVolatility(window::Integer = 20; annualised::Bool = true)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window), annualised)
    end
end

feature_name(feature::GarmanKlassVolatility) =
    Symbol("garman_klass_volatility_", feature.window)
lookback(feature::GarmanKlassVolatility) = feature.window - 1

function compute(feature::GarmanKlassVolatility, window::BarWindow)
    span = (length(window) - feature.window + 1):length(window)
    variance = mean(
        0.5 * log(window.highs[i] / window.lows[i])^2 -
            GARMAN_KLASS_COEFFICIENT * log(window.closes[i] / window.opens[i])^2
            for i in span
    )
    variance <= 0 && return nothing
    value = sqrt(variance)
    return feature.annualised ? annualise(value) : value
end

"""
    DownsideVolatility(window; annualised = true)

Annualised volatility of the negative returns only.

A position sizer cares about the left tail, and a symmetric volatility charges the same risk
premium to a stock that jumps up as to one that gaps down. Returns are measured about zero
rather than about the downside mean, so this is a semi-deviation and not the standard
deviation of a truncated sample.
"""
struct DownsideVolatility <: Feature
    window::Int
    annualised::Bool

    function DownsideVolatility(window::Integer = 20; annualised::Bool = true)
        window >= 3 || throw(ArgumentError("window must be at least 3, got $window"))
        return new(Int(window), annualised)
    end
end

feature_name(feature::DownsideVolatility) = Symbol("downside_volatility_", feature.window)
lookback(feature::DownsideVolatility) = feature.window

function compute(feature::DownsideVolatility, window::BarWindow)
    n = length(window.log_returns)
    returns = view(window.log_returns, (n - feature.window + 1):n)
    value = sqrt(mean(min(r, 0.0)^2 for r in returns))
    return feature.annualised ? annualise(value) : value
end

"""
    VolatilityRatio(fast, slow)

Log ratio of a short volatility estimate to a long one.

Positive means volatility is currently above its own recent norm, which is the state a regime
model has to detect and a position sizer has to shrink into. The log form makes a doubling and
a halving equal and opposite.
"""
struct VolatilityRatio <: Feature
    fast::RealisedVolatility
    slow::RealisedVolatility

    function VolatilityRatio(fast::Integer = 5, slow::Integer = 60)
        fast < slow ||
            throw(ArgumentError("fast window $fast must be shorter than slow window $slow"))
        return new(
            RealisedVolatility(fast; annualised = false),
            RealisedVolatility(slow; annualised = false),
        )
    end
end

feature_name(feature::VolatilityRatio) =
    Symbol("volatility_ratio_", feature.fast.window, "_", feature.slow.window)
lookback(feature::VolatilityRatio) = lookback(feature.slow)

function compute(feature::VolatilityRatio, window::BarWindow)
    fast = compute(feature.fast, window)
    slow = compute(feature.slow, window)
    (fast <= 0 || slow <= 0) && return nothing
    return log(fast / slow)
end

"""
    AverageTrueRange(window)

Average true range as a fraction of the current close.

True range includes the gap from the previous close, so unlike the range estimators above this
does see overnight moves. Reported as a fraction rather than in rupees, because it is used to
place stops as a multiple of typical movement and a rupee value would mean something different
for every symbol.
"""
struct AverageTrueRange <: Feature
    window::Int

    function AverageTrueRange(window::Integer = 14)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::AverageTrueRange) = Symbol("atr_pct_", feature.window)
lookback(feature::AverageTrueRange) = feature.window

function compute(feature::AverageTrueRange, window::BarWindow)
    n = length(window)
    total = 0.0
    for index in (n - feature.window + 1):n
        previous = window.closes[index - 1]
        total += max(
            window.highs[index] - window.lows[index],
            abs(window.highs[index] - previous),
            abs(window.lows[index] - previous),
        )
    end
    close = window.closes[end]
    close <= 0 && return nothing
    return total / feature.window / close
end
