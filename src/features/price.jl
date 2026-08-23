"""
Price-derived features.

Returns are computed and reported in log space. A model that adds and averages returns is
doing something meaningful only if they compose additively, and simple returns do not:
chaining +50% and -50% is not zero.
"""

"""
    LogReturn(horizon)

Log return over `horizon` bars, ending at the current bar.
"""
struct LogReturn <: Feature
    horizon::Int

    function LogReturn(horizon::Integer = 1)
        horizon >= 1 || throw(ArgumentError("horizon must be at least 1, got $horizon"))
        return new(Int(horizon))
    end
end

feature_name(feature::LogReturn) = Symbol("log_return_", feature.horizon)
lookback(feature::LogReturn) = feature.horizon
compute(feature::LogReturn, window::BarWindow) =
    window.log_closes[end] - window.log_closes[end - feature.horizon]

"""
    SimpleReturn(horizon)

Arithmetic return over `horizon` bars.

Kept alongside the log return because position sizing works in money, and money compounds
arithmetically even when the model reasons in log space.
"""
struct SimpleReturn <: Feature
    horizon::Int

    function SimpleReturn(horizon::Integer = 1)
        horizon >= 1 || throw(ArgumentError("horizon must be at least 1, got $horizon"))
        return new(Int(horizon))
    end
end

feature_name(feature::SimpleReturn) = Symbol("return_", feature.horizon)
lookback(feature::SimpleReturn) = feature.horizon
compute(feature::SimpleReturn, window::BarWindow) =
    window.closes[end] / window.closes[end - feature.horizon] - 1
