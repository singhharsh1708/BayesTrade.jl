"""
Binding a feature set to a store, one decision point at a time.

Every vector is computed from a fresh point-in-time slice. Vectorising the whole history at
once would be faster and is deliberately not done: a rolling computation over a full array is
the single easiest place for a future value to leak backwards through an off-by-one, and the
mistake is invisible in the output.

The cost is real but bounded. Each slice is a bisect into a sorted index and the window is
capped at the set's warm-up length, so the work per bar does not grow with history.
"""

"""
    FeatureEngine

Computes feature vectors for symbols held in a store.
"""
struct FeatureEngine{S <: BarStore}
    store::S
    features::FeatureSet
end

"""
    warmup_bars(engine)

Bars needed before every feature in the set can produce a value.
"""
warmup_bars(engine::FeatureEngine) = required_bars(engine.features)

"""
    features_at(engine, symbol, as_of)

The feature vector for `symbol` as it stood at `as_of`.
"""
function features_at(engine::FeatureEngine, symbol::AbstractString, as_of::DateTime)
    bars = history(engine.store, symbol; as_of = as_of, count = warmup_bars(engine))
    isempty(bars) && return empty_vector(engine.features, symbol, as_of)
    return compute(engine.features, BarWindow(bars); as_of = as_of)
end

"""
    walk(engine, symbol; start, stop, complete_only)

Feature vectors at every bar close in the window, oldest first.

The decision points come from the stored timestamps, which is a setup-time read of the
calendar rather than of the data. Each vector is then computed through [`features_at`](@ref),
so no vector can see past its own timestamp.
"""
function walk(
        engine::FeatureEngine, symbol::AbstractString;
        start::Union{DateTime, Nothing} = nothing,
        stop::Union{DateTime, Nothing} = nothing,
        complete_only::Bool = false,
    )
    vectors = FeatureVector[]
    for bar in load_range(engine.store, symbol, start, stop)
        vector = features_at(engine, symbol, bar.timestamp)
        (complete_only && !is_complete(vector)) && continue
        push!(vectors, vector)
    end
    return vectors
end

"""
    first_complete_at(engine, symbol)

The earliest moment every feature actually has a value, or `nothing` if there is no such
moment.

Counting bars against the set's warm-up is not enough. A feature can have all the history it
asked for and still decline, because the window was degenerate: a stock that did not move all
week has no z-score. Returning the warm-up bar's timestamp regardless would hand a backtest a
starting point at which its first decision cannot be made.

So this walks forward and checks. It is called once at setup, and paying a scan there is
better than discovering the gap on the first bar of a run.
"""
function first_complete_at(engine::FeatureEngine, symbol::AbstractString)
    bars = load_range(engine.store, symbol)
    length(bars) < warmup_bars(engine) && return nothing
    for index in warmup_bars(engine):length(bars)
        moment = bars[index].timestamp
        is_complete(features_at(engine, symbol, moment)) && return moment
    end
    return nothing
end
