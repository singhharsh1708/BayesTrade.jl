"""
Forward-looking labels, for offline fitting only.

Everything in this file reads bars *after* the moment it is asked about. That is what a label
is, and it is also exactly what must never happen anywhere else in the system. The separation
is the safeguard: labels live in their own file, take a store rather than a window, and are
never reachable from a [`Feature`](@ref).

A label that is not fully realised is `nothing`, never a partial estimate. The last few
examples in any dataset have no future left to measure, and quietly shortening their horizon
would make the most recent data look different from the rest.
"""

"""
    Label

What happened after a decision point.

Excursions are measured against the intraday extremes rather than the closes, because a stop
is hit by the path. A label built from closes alone would report a comfortable holding period
for a trade that was stopped out on day two.
"""
struct Label
    symbol::String
    as_of::DateTime
    realised_at::DateTime
    horizon_bars::Int
    forward_log_return::Float64
    max_adverse_excursion::Float64
    max_favourable_excursion::Float64

    function Label(;
            symbol::AbstractString,
            as_of::DateTime,
            realised_at::DateTime,
            horizon_bars::Integer,
            forward_log_return::Real,
            max_adverse_excursion::Real,
            max_favourable_excursion::Real,
        )
        realised_at > as_of ||
            throw(ArgumentError("a label must be realised strictly after the decision point"))
        horizon_bars >= 1 ||
            throw(ArgumentError("horizon_bars must be at least 1, got $horizon_bars"))
        max_adverse_excursion <= 0 ||
            throw(ArgumentError("adverse excursion is measured downward and cannot be positive"))
        max_favourable_excursion >= 0 ||
            throw(ArgumentError("favourable excursion cannot be negative"))
        return new(
            String(symbol), as_of, realised_at, Int(horizon_bars),
            Float64(forward_log_return), Float64(max_adverse_excursion),
            Float64(max_favourable_excursion),
        )
    end
end

is_positive(label::Label) = label.forward_log_return > 0

"""
    TrainingExample

One row of a fitting set: what was known, and what followed.
"""
struct TrainingExample
    features::FeatureVector
    label::Label

    function TrainingExample(features::FeatureVector, label::Label)
        features.as_of == label.as_of || throw(
            ArgumentError(
                string(
                    "features at ", features.as_of, " paired with a label at ", label.as_of,
                ),
            ),
        )
        features.symbol == label.symbol ||
            throw(ArgumentError("features and label describe different symbols"))
        return new(features, label)
    end
end

"""
    forward_label(store, symbol; as_of, horizon_bars)

What the next `horizon_bars` bars did, or `nothing` if they have not happened.
"""
function forward_label(
        store::BarStore, symbol::AbstractString; as_of::DateTime, horizon_bars::Integer,
    )
    horizon_bars >= 1 ||
        throw(ArgumentError("horizon_bars must be at least 1, got $horizon_bars"))

    entry = latest(store, symbol; as_of = as_of)
    entry === nothing && return nothing

    ahead = Bar[
        bar for bar in load_range(store, symbol, entry.timestamp)
            if bar.timestamp > entry.timestamp
    ]
    length(ahead) < horizon_bars && return nothing

    holding = view(ahead, 1:horizon_bars)
    exit_bar = holding[end]
    entry_price = entry.close

    return Label(
        symbol = symbol,
        as_of = as_of,
        realised_at = exit_bar.timestamp,
        horizon_bars = horizon_bars,
        forward_log_return = log(exit_bar.close / entry_price),
        max_adverse_excursion = min(
            0.0, minimum(log(bar.low / entry_price) for bar in holding),
        ),
        max_favourable_excursion = max(
            0.0, maximum(log(bar.high / entry_price) for bar in holding),
        ),
    )
end

"""
    build_training_set(engine, symbol; horizon_bars, start, stop, complete_only)

Pair every usable feature vector with the outcome that followed it.

Rows whose label is not fully realised are dropped, which always removes the most recent
`horizon_bars` decision points. That is not a bug to work around: those rows do not have an
answer yet, and inventing one is how a backtest learns to predict its own truncation.
"""
function build_training_set(
        engine::FeatureEngine, symbol::AbstractString;
        horizon_bars::Integer,
        start::Union{DateTime, Nothing} = nothing,
        stop::Union{DateTime, Nothing} = nothing,
        complete_only::Bool = true,
    )
    examples = TrainingExample[]
    for features in walk(
            engine, symbol; start = start, stop = stop, complete_only = complete_only,
        )
        label = forward_label(
            engine.store, symbol; as_of = features.as_of, horizon_bars = horizon_bars,
        )
        label === nothing && continue
        push!(examples, TrainingExample(features, label))
    end
    return examples
end
