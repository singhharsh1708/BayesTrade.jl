"""
Trend and momentum features.

Two conventions run through this module.

**Everything is scale-free.** A feature reports a log ratio or a standardised distance, never
a rupee amount. A model fitted on the rupee gap between a price and its moving average learns
the price level as much as the signal, and stops working after a split.

**Nothing is path-dependent from inception.** The textbook exponential moving average and
Wilder's RSI both depend on every bar since the series began, which makes them a function of
when the download started rather than of the market. Since a feature here sees a bounded
window by construction, both are replaced with bounded equivalents whose value is fully
determined by that window: an EMA seeded from a simple average far enough back that the seed
has decayed, and Cutler's RSI, which is a plain ratio of average gains to average losses.
"""

"""
    EMA_WARMUP_MULTIPLE

Window multiples of history before an exponential average is reported.

At three multiples the seed's weight is about `exp(-4)`, under two percent, so the reported
value is a property of the window rather than of where the data began.
"""
const EMA_WARMUP_MULTIPLE = 3

"""
    is_flat(values)

Whether `values` vary only by floating-point noise.

An exact comparison against zero is not enough: bit-identical inputs can still produce a
non-zero residual once a mean has been subtracted from them.

Measured as a spread rather than as a variance about the mean, which is the whole point. A
variance has to subtract a mean, and the rounding error of that mean grows with the length of
the vector, so a tolerance calibrated on a short window declares a long, genuinely flat one
to be varying. The spread of a bit-identical vector is exactly zero at any length, so the
tolerance only ever has to cover inputs that genuinely differ by a few units in the last
place.
"""
function is_flat(values::AbstractVector{Float64})
    scale = maximum(abs, values)
    spread = maximum(values) - minimum(values)
    return spread <= eps(Float64) * max(scale, 1.0) * length(values)
end

seeded_ema(values::AbstractVector{Float64}, window::Int) = begin
    alpha = 2 / (window + 1)
    level = mean(view(values, 1:window))
    for index in (window + 1):length(values)
        level += alpha * (values[index] - level)
    end
    level
end

"""
    MovingAverage(window; exponential = false)

Simple or exponential moving average of the close.

Rarely useful to a model directly, since it carries the price level. It is exposed because
the scale-free features below are defined against it and a reader should be able to check
them.
"""
struct MovingAverage <: Feature
    window::Int
    exponential::Bool

    function MovingAverage(window::Integer = 20; exponential::Bool = false)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window), exponential)
    end
end

feature_name(feature::MovingAverage) =
    Symbol(feature.exponential ? "ema_" : "sma_", feature.window)
lookback(feature::MovingAverage) =
    (feature.exponential ? feature.window * EMA_WARMUP_MULTIPLE : feature.window) - 1

function compute(feature::MovingAverage, window::BarWindow)
    feature.exponential || return mean(view(window.closes, (length(window) - feature.window + 1):length(window)))
    span = required_bars(feature)
    return seeded_ema(view(window.closes, (length(window) - span + 1):length(window)), feature.window)
end

"""
    PriceToMovingAverage(window; exponential = false)

Log distance from the close to its moving average.

Positive means trading above the average. Log rather than a percentage so the feature is
symmetric: twenty percent above and twenty percent below are equal and opposite.
"""
struct PriceToMovingAverage <: Feature
    average::MovingAverage
end

PriceToMovingAverage(window::Integer = 20; exponential::Bool = false) =
    PriceToMovingAverage(MovingAverage(window; exponential = exponential))

feature_name(feature::PriceToMovingAverage) =
    Symbol("close_to_", feature_name(feature.average))
lookback(feature::PriceToMovingAverage) = lookback(feature.average)

function compute(feature::PriceToMovingAverage, window::BarWindow)
    average = compute(feature.average, window)
    average <= 0 && return nothing
    return log(window.closes[end] / average)
end

"""
    MovingAverageSpread(fast, slow; exponential = true)

Log spread between a fast and a slow moving average.

The scale-free form of a moving-average crossover: sign gives the crossover, magnitude gives
how far apart they are, which the sign alone throws away.
"""
struct MovingAverageSpread <: Feature
    fast::MovingAverage
    slow::MovingAverage
    exponential::Bool

    function MovingAverageSpread(
            fast::Integer = 12, slow::Integer = 26; exponential::Bool = true,
        )
        fast < slow ||
            throw(ArgumentError("fast window $fast must be shorter than slow window $slow"))
        return new(
            MovingAverage(fast; exponential = exponential),
            MovingAverage(slow; exponential = exponential),
            exponential,
        )
    end
end

feature_name(feature::MovingAverageSpread) = Symbol(
    feature.exponential ? "ema" : "sma", "_spread_",
    feature.fast.window, "_", feature.slow.window,
)
lookback(feature::MovingAverageSpread) = lookback(feature.slow)

function compute(feature::MovingAverageSpread, window::BarWindow)
    fast = compute(feature.fast, window)
    slow = compute(feature.slow, window)
    (fast <= 0 || slow <= 0) && return nothing
    return log(fast / slow)
end

"""
    Momentum(window, skip)

Log return over `window` bars, skipping the most recent `skip`.

The skip is not decoration. Cross-sectional momentum reverses over the most recent month, so
the classic construction measures twelve months ending one month ago. Including the last
month mixes a reversal signal into a trend signal and weakens both.
"""
struct Momentum <: Feature
    window::Int
    skip::Int

    function Momentum(window::Integer = 60, skip::Integer = 5)
        window >= 1 || throw(ArgumentError("window must be at least 1, got $window"))
        skip >= 0 || throw(ArgumentError("skip must be non-negative, got $skip"))
        return new(Int(window), Int(skip))
    end
end

feature_name(feature::Momentum) = feature.skip == 0 ?
    Symbol("momentum_", feature.window) :
    Symbol("momentum_", feature.window, "_", feature.skip)
lookback(feature::Momentum) = feature.window + feature.skip

function compute(feature::Momentum, window::BarWindow)
    stop = length(window) - feature.skip
    return window.log_closes[stop] - window.log_closes[stop - feature.window]
end

"""
    RelativeStrengthIndex(window)

Cutler's RSI over the window, on the usual zero to one hundred scale.

Wilder's original smoothing depends on every bar since the series began, so two downloads of
the same symbol starting a year apart disagree about today's value. Cutler's variant is a
plain ratio of average gain to average loss over a fixed window, which is the same idea and
is fully determined by what the feature can see.
"""
struct RelativeStrengthIndex <: Feature
    window::Int

    function RelativeStrengthIndex(window::Integer = 14)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::RelativeStrengthIndex) = Symbol("rsi_", feature.window)
lookback(feature::RelativeStrengthIndex) = feature.window

function compute(feature::RelativeStrengthIndex, window::BarWindow)
    changes = view(diff(window.closes), (length(window) - feature.window):(length(window) - 1))
    gains = mean(max(change, 0.0) for change in changes)
    losses = mean(max(-change, 0.0) for change in changes)
    gains == 0 && losses == 0 && return 50.0
    losses == 0 && return 100.0
    return 100 - 100 / (1 + gains / losses)
end

"""
    PriceZScore(window)

How far the log price sits from its recent mean, in recent standard deviations.

A mean-reversion signal, and the natural counterpart to momentum. Undefined on a perfectly
flat window, where it reports nothing rather than dividing by zero.
"""
struct PriceZScore <: Feature
    window::Int

    function PriceZScore(window::Integer = 20)
        window >= 3 || throw(ArgumentError("window must be at least 3, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::PriceZScore) = Symbol("price_zscore_", feature.window)
lookback(feature::PriceZScore) = feature.window - 1

function compute(feature::PriceZScore, window::BarWindow)
    recent = Float64.(view(window.log_closes, (length(window) - feature.window + 1):length(window)))
    is_flat(recent) && return nothing
    return (recent[end] - mean(recent)) / std(recent)
end

"""
    DrawdownFromHigh(window)

Log distance from the close to the highest high in the window.

Zero at a new high and negative below it, never positive. Uses the high rather than the
close, because a stop is hit intraday and a drawdown measured close to close does not know
that.
"""
struct DrawdownFromHigh <: Feature
    window::Int

    function DrawdownFromHigh(window::Integer = 60)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::DrawdownFromHigh) = Symbol("drawdown_", feature.window)
lookback(feature::DrawdownFromHigh) = feature.window - 1

function compute(feature::DrawdownFromHigh, window::BarWindow)
    peak = maximum(view(window.highs, (length(window) - feature.window + 1):length(window)))
    return log(window.closes[end] / peak)
end

"""
    TrendSlope(window)

Least-squares slope of log price against time, in log return per bar.

Directly comparable with a return: a slope of 0.001 is a tenth of a percent per bar, whatever
the price level.
"""
struct TrendSlope <: Feature
    window::Int

    function TrendSlope(window::Integer = 20)
        window >= 3 || throw(ArgumentError("window must be at least 3, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::TrendSlope) = Symbol("trend_slope_", feature.window)
lookback(feature::TrendSlope) = feature.window - 1
compute(feature::TrendSlope, window::BarWindow) =
    first(regress(trend_values(feature.window, window)))

"""
    TrendQuality(window)

R-squared of that same regression: how much of the move is trend, not noise.

Zero to one. Reported alongside the slope because a steep noisy line and a shallow clean one
are different situations, and the slope alone cannot tell them apart.
"""
struct TrendQuality <: Feature
    window::Int

    function TrendQuality(window::Integer = 20)
        window >= 3 || throw(ArgumentError("window must be at least 3, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::TrendQuality) = Symbol("trend_quality_", feature.window)
lookback(feature::TrendQuality) = feature.window - 1
compute(feature::TrendQuality, window::BarWindow) =
    last(regress(trend_values(feature.window, window)))

trend_values(span::Int, window::BarWindow) =
    Float64.(view(window.log_closes, (length(window) - span + 1):length(window)))

"""
    regress(values)

Slope and R-squared of `values` against an evenly spaced index.

R-squared is undefined on a flat series, where the regression explains none of a variance
that is itself zero. It reports nothing there rather than one.
"""
function regress(values::AbstractVector{Float64})
    n = length(values)
    x = collect(1.0:n)
    x_centred = x .- mean(x)
    y_centred = values .- mean(values)

    slope = dot(x_centred, y_centred) / dot(x_centred, x_centred)
    is_flat(values) && return 0.0, nothing
    explained = slope^2 * dot(x_centred, x_centred)
    return slope, min(1.0, explained / dot(y_centred, y_centred))
end
