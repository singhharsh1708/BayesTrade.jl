"""
Volume and liquidity features.

Volume features are relative, because a million shares means something different for every
symbol. Liquidity features are absolute, because a risk limit that says a position may not
exceed two percent of daily turnover has to be measured in rupees.

Comparisons use the median rather than the mean. Volume distributions have a long right tail,
and one results-day spike pulls a mean far enough that ordinary days afterwards look quiet by
comparison.
"""

"""
    AMIHUD_SCALE

Amihud's ratio is tiny in rupee units; scaling keeps it in a readable range.
"""
const AMIHUD_SCALE = 1.0e6

"""
    RelativeVolume(window)

Log ratio of today's volume to the window median.

Zero on a typical day, positive on a busy one. The log makes twice as busy and half as busy
equal and opposite, which a raw ratio does not.
"""
struct RelativeVolume <: Feature
    window::Int

    function RelativeVolume(window::Integer = 20)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::RelativeVolume) = Symbol("relative_volume_", feature.window)
lookback(feature::RelativeVolume) = feature.window - 1

function compute(feature::RelativeVolume, window::BarWindow)
    volumes = view(window.volumes, (length(window) - feature.window + 1):length(window))
    reference = median(volumes)
    (reference <= 0 || volumes[end] <= 0) && return nothing
    return log(volumes[end] / reference)
end

"""
    VolumeZScore(window)

How unusual today's volume is, in standard deviations of log volume.

Log volume rather than volume, because volume is roughly log-normal and a z-score of the raw
series is dominated by the few largest days.
"""
struct VolumeZScore <: Feature
    window::Int

    function VolumeZScore(window::Integer = 20)
        window >= 3 || throw(ArgumentError("window must be at least 3, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::VolumeZScore) = Symbol("volume_zscore_", feature.window)
lookback(feature::VolumeZScore) = feature.window - 1

function compute(feature::VolumeZScore, window::BarWindow)
    volumes = view(window.volumes, (length(window) - feature.window + 1):length(window))
    minimum(volumes) <= 0 && return nothing
    logs = log.(volumes)
    spread = std(logs)
    spread <= 0 && return nothing
    return (logs[end] - mean(logs)) / spread
end

"""
    MedianTurnover(window)

Median traded value over the window, in rupees.

Deliberately not scale-free. This is the number a liquidity limit is written against: an order
may not exceed some fraction of what actually trades, and that fraction has to be taken of a
real amount.
"""
struct MedianTurnover <: Feature
    window::Int

    function MedianTurnover(window::Integer = 20)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::MedianTurnover) = Symbol("turnover_median_", feature.window)
lookback(feature::MedianTurnover) = feature.window - 1

compute(feature::MedianTurnover, window::BarWindow) = median(
    turnover(window.bars[index])
        for index in (length(window) - feature.window + 1):length(window)
)

"""
    AmihudIlliquidity(window)

Average absolute return per rupee traded, scaled.

How far the price moves for a given amount of trading. High means a small order moves the
price a lot, which is exactly the condition under which a backtest's assumed fill is a
fiction. Bars with no turnover are skipped rather than treated as infinitely illiquid, since a
halted day says nothing about how the stock trades.
"""
struct AmihudIlliquidity <: Feature
    window::Int

    function AmihudIlliquidity(window::Integer = 20)
        window >= 2 || throw(ArgumentError("window must be at least 2, got $window"))
        return new(Int(window))
    end
end

feature_name(feature::AmihudIlliquidity) = Symbol("amihud_", feature.window)
lookback(feature::AmihudIlliquidity) = feature.window

function compute(feature::AmihudIlliquidity, window::BarWindow)
    n = length(window)
    total = 0.0
    traded = 0
    for index in (n - feature.window + 1):n
        value = turnover(window.bars[index])
        value <= 0 && continue
        total += abs(window.log_returns[index - 1]) / value
        traded += 1
    end
    traded == 0 && return nothing
    return total / traded * AMIHUD_SCALE
end
