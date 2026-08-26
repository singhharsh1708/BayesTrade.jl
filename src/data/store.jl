"""
Storage for historical bars, with the point-in-time read separated from the bulk read.

Two ways to get bars out of a store, and the distinction is the whole point:

`history(store, symbol; as_of)`
    Everything that had closed by `as_of`, and nothing else. This is what features, models
    and the backtester use. It cannot return a future bar.

`load_range(store, symbol, start, stop)`
    An arbitrary window, including bars in the future relative to any simulated clock. This
    exists for loading, inspecting and reporting on a dataset. Using it inside a feature is a
    look-ahead bug, and it is named to make that obvious in review.

Bars for one symbol must all share an interval. Silently mixing daily and five-minute bars
would produce features that look fine and mean nothing.
"""

"""
    IntervalConflictError

Raised when bars of a different interval are added to an existing symbol.
"""
struct IntervalConflictError <: Exception
    message::String
end

Base.showerror(io::IO, error::IntervalConflictError) =
    print(io, "IntervalConflictError: ", error.message)

"""
    Coverage

What a store holds for one symbol.
"""
struct Coverage
    symbol::String
    interval::String
    first::DateTime
    last::DateTime
    count::Int
end

Base.in(moment::DateTime, coverage::Coverage) =
    coverage.first <= moment <= coverage.last

"""
    BarStore

Read and write access to historical bars.
"""
abstract type BarStore end

"""
    InMemoryBarStore

Bars held in sorted per-symbol vectors.

Reads bisect rather than scan, which matters more than it looks: the backtester issues one
point-in-time query per symbol per bar, so a linear scan would make the whole backtest
quadratic in the length of history.
"""
struct InMemoryBarStore <: BarStore
    bars::Dict{String, Vector{Bar}}
    stamps::Dict{String, Vector{DateTime}}
    intervals::Dict{String, String}
end

InMemoryBarStore() = InMemoryBarStore(
    Dict{String, Vector{Bar}}(), Dict{String, Vector{DateTime}}(), Dict{String, String}(),
)

function InMemoryBarStore(bars)
    store = InMemoryBarStore()
    upsert!(store, bars)
    return store
end

"""
    upsert!(store, bars)

Insert or replace bars, returning how many were written.

Replacing on a repeated timestamp is deliberate: vendors revise bars, and the revision is
the better datum.
"""
function upsert!(store::InMemoryBarStore, bars)
    grouped = Dict{String, Vector{Bar}}()
    for bar in bars
        push!(get!(grouped, bar.symbol, Bar[]), bar)
    end

    written = 0
    for (symbol, incoming) in grouped
        intervals = unique(bar.interval for bar in incoming)
        length(intervals) == 1 || throw(
            IntervalConflictError(
                "$symbol: a single batch mixes intervals $(sort(intervals))",
            ),
        )
        interval = first(intervals)
        existing = get(store.intervals, symbol, nothing)
        if existing !== nothing && existing != interval
            throw(
                IntervalConflictError(
                    string(
                        "$symbol: store holds \"$existing\" bars, ",
                        "refusing to add \"$interval\" bars",
                    ),
                ),
            )
        end

        merged = Dict{DateTime, Bar}(bar.timestamp => bar for bar in get(store.bars, symbol, Bar[]))
        for bar in incoming
            merged[bar.timestamp] = bar
        end
        ordered = Bar[merged[stamp] for stamp in sort!(collect(keys(merged)))]

        store.bars[symbol] = ordered
        store.stamps[symbol] = DateTime[bar.timestamp for bar in ordered]
        store.intervals[symbol] = interval
        written += length(incoming)
    end
    return written
end

symbols(store::InMemoryBarStore) = sort!(collect(keys(store.bars)))

function coverage(store::InMemoryBarStore, symbol::AbstractString)
    bars = get(store.bars, symbol, nothing)
    (bars === nothing || isempty(bars)) && return nothing
    return Coverage(
        String(symbol), store.intervals[symbol],
        first(bars).timestamp, last(bars).timestamp, length(bars),
    )
end

"""
    history(store, symbol; as_of, count = nothing, since = nothing)

Bars that had closed by `as_of`, oldest first.

`count` keeps the most recent `count` of them, which is what a rolling feature window needs.
`since` bounds the window from below and is inclusive, like `as_of` bounds it from above.

Both bounds are inclusive, so the lower one bisects with `searchsortedfirst` and the upper
with `searchsortedlast`. Using the same bisect for both would quietly include one bar from
before the requested window whenever the bound does not land exactly on a stored timestamp,
which is most of the time.
"""
function history(
        store::InMemoryBarStore, symbol::AbstractString;
        as_of::DateTime,
        count::Union{Integer, Nothing} = nothing,
        since::Union{DateTime, Nothing} = nothing,
    )
    count !== nothing && count < 0 &&
        throw(ArgumentError("count must be non-negative, got $count"))
    bars = get(store.bars, symbol, nothing)
    (bars === nothing || isempty(bars)) && return Bar[]

    stamps = store.stamps[symbol]
    stop = searchsortedlast(stamps, as_of)
    start = since === nothing ? 1 : searchsortedfirst(stamps, since)
    stop < start && return Bar[]

    window = view(bars, start:stop)
    count === nothing && return collect(window)
    count == 0 && return Bar[]
    return collect(window[max(1, end - count + 1):end])
end

"""
    upcoming(store, symbol; after, count)

The first `count` bars strictly after `after`, oldest first.

The mirror of [`history`](@ref), which takes the last `count` at or before a moment. It exists
because a label needs a bounded look forward and had no way to ask for one: the only forward
read was [`load_range`](@ref), which returns everything from a moment to the end of history, and
a caller wanting three bars got all of them.

Not safe inside a feature, and safe inside a label. That is the whole distinction: a feature at
time t may not see past t, and a label at time t is *defined* by what happened after it.
"""
function upcoming(
        store::InMemoryBarStore, symbol::AbstractString;
        after::DateTime, count::Integer,
    )
    count >= 0 || throw(ArgumentError("count must be non-negative, got $count"))
    count == 0 && return Bar[]
    bars = get(store.bars, symbol, nothing)
    (bars === nothing || isempty(bars)) && return Bar[]

    stamps = store.stamps[symbol]
    start = searchsortedfirst(stamps, after)
    # Strictly after, so a bar landing exactly on the bound is the one being labelled rather
    # than part of what happens next.
    start <= length(stamps) && stamps[start] == after && (start += 1)
    start > length(bars) && return Bar[]
    return collect(view(bars, start:min(length(bars), start + count - 1)))
end

"""
    load_range(store, symbol, start = nothing, stop = nothing)

An arbitrary window, inclusive at both ends. Not safe inside a feature; see the module
docstring.
"""
function load_range(
        store::InMemoryBarStore, symbol::AbstractString,
        start::Union{DateTime, Nothing} = nothing,
        stop::Union{DateTime, Nothing} = nothing,
    )
    bars = get(store.bars, symbol, nothing)
    (bars === nothing || isempty(bars)) && return Bar[]
    stamps = store.stamps[symbol]
    lower = start === nothing ? 1 : searchsortedfirst(stamps, start)
    upper = stop === nothing ? length(bars) : searchsortedlast(stamps, stop)
    upper < lower && return Bar[]
    return collect(view(bars, lower:upper))
end

"""
    latest(store, symbol; as_of)

The most recent bar that had closed by `as_of`, or `nothing`.
"""
function latest(store::BarStore, symbol::AbstractString; as_of::DateTime)
    window = history(store, symbol; as_of = as_of, count = 1)
    return isempty(window) ? nothing : last(window)
end

"""
    bar_count(store, symbol)

How many bars are held for a symbol.
"""
function bar_count(store::BarStore, symbol::AbstractString)
    found = coverage(store, symbol)
    return found === nothing ? 0 : found.count
end

Base.in(symbol::AbstractString, store::BarStore) = coverage(store, symbol) !== nothing

"""
    clear!(store, symbol = nothing)

Drop one symbol, or everything.
"""
function clear!(store::InMemoryBarStore, symbol::Union{AbstractString, Nothing} = nothing)
    if symbol === nothing
        empty!(store.bars)
        empty!(store.stamps)
        empty!(store.intervals)
        return store
    end
    delete!(store.bars, symbol)
    delete!(store.stamps, symbol)
    delete!(store.intervals, symbol)
    return store
end

"""
    all_bars(store)

Every bar held, ordered by symbol then time. For persistence and reporting.
"""
all_bars(store::InMemoryBarStore) = Bar[
    bar for symbol in symbols(store) for bar in store.bars[symbol]
]

"""
    align(store, symbols; as_of)

The latest bar for each symbol as of one moment, skipping symbols with none.

Symbols are dropped rather than filled forward. A stale bar presented as current is how a
portfolio ends up marked at a price that no longer exists.
"""
function align(store::BarStore, wanted; as_of::DateTime)
    aligned = Dict{String, Bar}()
    for symbol in wanted
        bar = latest(store, symbol; as_of = as_of)
        bar === nothing || (aligned[String(symbol)] = bar)
    end
    return aligned
end
