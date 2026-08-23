"""
The output of the feature engine: a named vector at a point in time.

A feature vector carries two timestamps, not one. `as_of` is the moment the question was
asked; `data_as_of` is the close of the most recent bar that answered it. They are usually
the same, and the gap between them is exactly what matters when they are not: a decision
taken now from a bar three days old is stale, and a vector stamped only with the bar's own
time cannot say so.

A feature can be absent for two different reasons, and they need different responses. It may
not have enough history yet, which time fixes. Or it may have had enough history and still
declined, because the window was degenerate: a stock that did not move all week has no
z-score, and one that did not trade has no illiquidity ratio. Waiting does not fix that, so
`undefined_features` records it separately from the warm-up.

Non-finite values are rejected at construction. A NaN that reaches a model does not crash
it; it silently poisons a posterior and every decision downstream, which is far worse than
failing here.
"""

"""
    FeatureVector

Feature values for one symbol at one moment.
"""
struct FeatureVector
    symbol::String
    as_of::DateTime
    data_as_of::Union{DateTime, Nothing}
    values::Dict{Symbol, Float64}
    missing_features::Vector{Symbol}
    undefined_features::Vector{Symbol}
    n_bars::Int

    function FeatureVector(;
            symbol::AbstractString,
            as_of::DateTime,
            data_as_of::Union{DateTime, Nothing} = nothing,
            values::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
            missing_features::Vector{Symbol} = Symbol[],
            undefined_features::Vector{Symbol} = Symbol[],
            n_bars::Integer = 0,
        )
        isempty(symbol) && throw(ArgumentError("feature vector needs a symbol"))
        n_bars >= 0 || throw(ArgumentError("n_bars cannot be negative, got $n_bars"))

        if data_as_of !== nothing && data_as_of > as_of
            throw(
                ArgumentError(
                    string(
                        symbol, ": features asked for at ", as_of,
                        " were computed from a bar closing at ", data_as_of,
                    ),
                ),
            )
        end
        n_bars > 0 && data_as_of === nothing &&
            throw(ArgumentError("$symbol: $n_bars bars used but none dated"))

        bad = Symbol[name for (name, value) in values if !isfinite(value)]
        isempty(bad) || throw(
            ArgumentError(
                string(
                    symbol, ": non-finite feature values for ",
                    join(sort(string.(bad)), ", "),
                    "; a NaN reaching a model poisons the posterior instead of failing",
                ),
            ),
        )

        overlap = intersect(Set(keys(values)), Set(missing_features))
        isempty(overlap) || throw(
            ArgumentError(
                string(
                    symbol, ": ", join(sort(string.(collect(overlap))), ", "),
                    " are both present and missing",
                ),
            ),
        )
        stray = setdiff(Set(undefined_features), Set(missing_features))
        isempty(stray) || throw(
            ArgumentError(
                string(
                    symbol, ": ", join(sort(String[string(n) for n in stray]), ", "),
                    " are undefined but not recorded as missing",
                ),
            ),
        )
        return new(
            String(symbol), as_of, data_as_of, values,
            missing_features, undefined_features, Int(n_bars),
        )
    end
end

"""
    staleness(vector)

How old the newest bar was when the question was asked, or `nothing` with no data.
"""
staleness(vector::FeatureVector) =
    vector.data_as_of === nothing ? nothing : vector.as_of - vector.data_as_of

"""
    is_stale(vector, max_age)

Whether the data behind this vector is too old to act on.

A vector with no data at all counts as stale. Treating "nothing known" as fresh is how a
system ends up trading on a prior.
"""
function is_stale(vector::FeatureVector, max_age::Period)
    age = staleness(vector)
    return age === nothing || age > max_age
end

"""
    is_complete(vector)

Whether every requested feature produced a value.
"""
is_complete(vector::FeatureVector) = isempty(vector.missing_features)

feature_names(vector::FeatureVector) = sort!(collect(keys(vector.values)))

Base.haskey(vector::FeatureVector, name::Symbol) = haskey(vector.values, name)
Base.get(vector::FeatureVector, name::Symbol, default) = get(vector.values, name, default)

"""
    require(vector, name)

Read a feature, failing loudly if it is absent.

Absence is a real state a model must handle deliberately rather than by reading a zero that
means nothing, and the three ways it arises are reported separately because each calls for a
different response: wait, look at the data, or fix the caller.
"""
function require(vector::FeatureVector, name::Symbol)
    haskey(vector.values, name) && return vector.values[name]
    reason = if name in vector.undefined_features
        "is undefined on this window"
    elseif name in vector.missing_features
        "still warming up"
    else
        "was never requested"
    end
    throw(KeyError(string(vector.symbol, " at ", vector.as_of, ": ", name, " ", reason)))
end

"""
    design_row(vector, names)

Feature values in a fixed order, for a model that expects a design row.

The order is the caller's, never the dictionary's: a model fitted on one column order and
scored on another produces plausible numbers that are entirely wrong.
"""
design_row(vector::FeatureVector, names) =
    Float64[require(vector, name) for name in names]

"""
    subset(vector, names)

A vector holding only `names`, keeping their missing status.
"""
function subset(vector::FeatureVector, names)
    wanted = Set(names)
    return FeatureVector(
        symbol = vector.symbol,
        as_of = vector.as_of,
        data_as_of = vector.data_as_of,
        values = Dict{Symbol, Float64}(
            name => value for (name, value) in vector.values if name in wanted
        ),
        missing_features = Symbol[
            name for name in vector.missing_features if name in wanted
        ],
        undefined_features = Symbol[
            name for name in vector.undefined_features if name in wanted
        ],
        n_bars = vector.n_bars,
    )
end
