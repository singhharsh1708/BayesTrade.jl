"""
The feature contract.

A feature is a pure function from a window of bars to one number. It has no state, no store,
no clock and no knowledge of any other symbol. Everything a feature could use to cheat has
been kept out of its reach rather than forbidden by convention.

`lookback` is how many bars a feature needs *before* the current one. During warm-up a
feature returns `nothing`, and the vector records it as missing rather than substituting a
zero. A zero is a number a model will happily fit to, and it means nothing.
"""

"""
    Feature

One named quantity computed from a window of bars.

Concrete features define [`feature_name`](@ref), [`lookback`](@ref) and [`compute`](@ref).
"""
abstract type Feature end

"""
    feature_name(feature)

The column name. Stable across versions, because models are fitted on it.
"""
feature_name(feature::Feature) =
    throw(ArgumentError("$(typeof(feature)) must define feature_name"))

"""
    lookback(feature)

Bars required before the current one.
"""
lookback(feature::Feature) =
    throw(ArgumentError("$(typeof(feature)) must define lookback"))

"""
    compute(feature, window)

The value at the window's current bar, assuming enough history. Called only through
[`evaluate`](@ref), which enforces the warm-up.
"""
compute(feature::Feature, window::BarWindow) =
    throw(ArgumentError("$(typeof(feature)) must define compute"))

"""
    required_bars(feature)

Total bars needed, including the current one.
"""
required_bars(feature::Feature) = lookback(feature) + 1

"""
    evaluate(feature, window)

The feature's value, or `nothing` while warming up.

The warm-up is enforced here rather than in each implementation, so a feature that would
happily return a value from too little history cannot.
"""
function evaluate(feature::Feature, window::BarWindow)
    length(window) < required_bars(feature) && return nothing
    return compute(feature, window)
end

Base.show(io::IO, feature::Feature) = print(
    io, '<', nameof(typeof(feature)), ' ', feature_name(feature),
    " lookback=", lookback(feature), '>',
)

"""
    FeatureSet

A fixed collection of features, computed together over one window.
"""
struct FeatureSet
    features::Vector{Feature}

    function FeatureSet(features::AbstractVector{<:Feature})
        isempty(features) && throw(ArgumentError("a feature set needs at least one feature"))
        # Tallied through a dictionary rather than counting each name against the whole
        # list. Quadratic is irrelevant at this size, but the counting predicate does not
        # infer cleanly and this reads as what it is.
        tally = Dict{Symbol, Int}()
        for feature in features
            name = feature_name(feature)
            tally[name] = get(tally, name, 0) + 1
        end
        duplicates = sort!(Symbol[name for (name, seen) in tally if seen > 1])
        isempty(duplicates) || throw(
            ArgumentError(
                string("duplicate feature names: ", join(string.(duplicates), ", ")),
            ),
        )
        return new(Feature[feature for feature in features])
    end
end

Base.length(set::FeatureSet) = length(set.features)
Base.iterate(set::FeatureSet, state...) = iterate(set.features, state...)

"""
    columns(set)

Feature names in declaration order. This is the canonical column order.

Named `columns` rather than `names` because `Base.names` means something else entirely, and
a package that exports a name shadowing Base makes every caller choose between them.
"""
columns(set::FeatureSet) = Symbol[feature_name(feature) for feature in set.features]

"""
    lookback(set)

The longest warm-up in the set.
"""
lookback(set::FeatureSet) = maximum(lookback(feature) for feature in set.features)
required_bars(set::FeatureSet) = lookback(set) + 1

"""
    compute(set, window; as_of = nothing)

Evaluate every feature over `window`.

`as_of` is the moment the question was asked, which may be later than the last bar in the
window. Defaulting it to the window's own end is right for a backtest stepping bar by bar
and wrong for a live loop asking between bars, so the live loop passes its own clock.
"""
function compute(
        set::FeatureSet, window::BarWindow; as_of::Union{DateTime, Nothing} = nothing,
    )
    values = Dict{Symbol, Float64}()
    missing_features = Symbol[]
    for feature in set.features
        value = evaluate(feature, window)
        if value === nothing
            push!(missing_features, feature_name(feature))
        else
            values[feature_name(feature)] = Float64(value)
        end
    end
    return FeatureVector(
        symbol = window_symbol(window),
        as_of = as_of === nothing ? window_as_of(window) : as_of,
        data_as_of = window_as_of(window),
        values = values,
        missing_features = missing_features,
        n_bars = length(window),
    )
end

"""
    empty_vector(set, symbol, as_of)

A vector with nothing computed, for a symbol with no bars yet.
"""
empty_vector(set::FeatureSet, symbol::AbstractString, as_of::DateTime) = FeatureVector(
    symbol = symbol, as_of = as_of, missing_features = columns(set), n_bars = 0,
)

"""
    subset(set, wanted)

A feature set holding only the named features.
"""
function subset(set::FeatureSet, wanted)
    requested = Set(wanted)
    unknown = setdiff(requested, Set(columns(set)))
    isempty(unknown) || throw(
        KeyError(
            string("unknown features: ", join(sort(String[string(n) for n in unknown]), ", ")),
        ),
    )
    return FeatureSet(
        Feature[f for f in set.features if feature_name(f) in requested],
    )
end

Base.vcat(left::FeatureSet, right::FeatureSet) =
    FeatureSet(vcat(left.features, right.features))
