"""
The live market feed.

A backtest is handed bars that are already complete and already correct. A live feed is handed
ticks, out of order, sometimes twice, and sometimes not at all. Everything in this file exists
because those three failures are silent by default: a feed that stops sending looks exactly
like a market that stopped moving, and a system that cannot tell them apart will keep trading
on a price from an hour ago.

Nothing here calls `now()`. The clock is always passed in, so a session can be replayed bar for
bar and a staleness test is a test rather than a race.
"""

"""
    TickSource

Where quotes come from. A replay source and a socket are interchangeable at the call site.
"""
abstract type TickSource end

next_tick!(source::TickSource) = throw(
    ArgumentError("$(typeof(source)) must define next_tick!"),
)
source_symbols(source::TickSource) = throw(
    ArgumentError("$(typeof(source)) must define source_symbols"),
)

"""
    ReplayTickSource(quotes)

A deterministic source that hands back recorded quotes in order.

The same code the socket will drive, run against a recording. Every test in this file uses it,
which is the point: a feed bug should be reproducible without a market.
"""
mutable struct ReplayTickSource <: TickSource
    quotes::Vector{Quote}
    position::Int

    function ReplayTickSource(quotes::AbstractVector{Quote})
        return new(collect(quotes), 0)
    end
end

function next_tick!(source::ReplayTickSource)
    source.position >= length(source.quotes) && return nothing
    source.position += 1
    return source.quotes[source.position]
end

source_symbols(source::ReplayTickSource) =
    unique(String[price.symbol for price in source.quotes])

exhausted(source::ReplayTickSource) = source.position >= length(source.quotes)

"""
    FeedHealth

What the feed has been doing, and whether it can still be believed.

`max_silence` is the heart of it. A feed that has said nothing for longer than that is not
reporting a quiet market, it is not reporting, and the difference decides whether the models
absorb a bar or skip it.
"""
mutable struct FeedHealth
    max_silence::Period
    last_tick_at::Union{DateTime, Nothing}
    n_accepted::Int
    n_stale::Int
    n_out_of_order::Int
    n_duplicates::Int

    function FeedHealth(; max_silence::Period = Minute(2))
        Dates.toms(max_silence) > 0 ||
            throw(ArgumentError("max_silence must be positive"))
        return new(max_silence, nothing, 0, 0, 0, 0)
    end
end

"""
    is_stale(health, now)

Whether the feed has gone quiet for longer than it is allowed to.

A feed that has never ticked is stale, not fresh. Treating an absent feed as healthy until it
proves otherwise is the assumption that gets a system trading on nothing.
"""
function is_stale(health::FeedHealth, now::DateTime)
    last = health.last_tick_at
    last === nothing && return true
    return now - last > health.max_silence
end

"""
    silence(health, now)

How long the feed has been quiet, or `nothing` if it has never spoken.
"""
function silence(health::FeedHealth, now::DateTime)
    # Bound to a local before the check. Narrowing a union field inside an expression does
    # not refine the field's type for the branch that follows.
    last = health.last_tick_at
    last === nothing && return nothing
    return now - last
end

"""
    accept!(health, quote)

Record a tick and say whether it should be believed.

Returns `:accepted`, `:duplicate` or `:out_of_order`. A tick older than the last one is not
merged in: on a real socket that is a reconnection replaying history, and folding it into the
current bar would rewrite a price the system has already acted on.
"""
function accept!(health::FeedHealth, price::Quote)
    last = health.last_tick_at
    if last !== nothing
        if price.timestamp < last
            health.n_out_of_order += 1
            return :out_of_order
        elseif price.timestamp == last
            health.n_duplicates += 1
            return :duplicate
        end
    end
    health.last_tick_at = price.timestamp
    health.n_accepted += 1
    return :accepted
end

"""
    mark_stale!(health)

Record that a bar passed with the feed quiet.
"""
function mark_stale!(health::FeedHealth)
    health.n_stale += 1
    return health
end

"""
    BarAggregator

Ticks in, bars out.

A bar is published only when a tick from the next bucket arrives, never when the clock passes
its end. That is deliberate: the last trade of a bucket is the close, and a bar published on a
timer would be published before the market had finished deciding what its close was.

The consequence is that the final bar of a session needs [`flush!`](@ref), and that is better
than the alternative, where every bar is a guess that the next tick will not arrive.
"""
mutable struct BarAggregator
    symbol::String
    interval::Period
    label::String
    bucket_start::Union{DateTime, Nothing}
    open::Float64
    high::Float64
    low::Float64
    close::Float64
    volume::Float64
    n_ticks::Int

    function BarAggregator(
            symbol::AbstractString; interval::Period = Minute(1), label::AbstractString = "1m",
        )
        isempty(symbol) && throw(ArgumentError("an aggregator needs a symbol"))
        Dates.toms(interval) > 0 || throw(ArgumentError("interval must be positive"))
        return new(
            String(symbol), interval, String(label), nothing, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
        )
    end
end

"""
    bucket_of(aggregator, moment)

The start of the bucket a moment falls in.

Floored to the interval rather than measured from the first tick, so two aggregators started at
different times agree on where the boundaries are.
"""
function bucket_of(aggregator::BarAggregator, moment::DateTime)
    step = Dates.toms(aggregator.interval)
    ms = Dates.value(moment)
    return DateTime(Dates.UTM(ms - mod(ms, step)))
end

has_open_bar(aggregator::BarAggregator) = aggregator.bucket_start !== nothing

"""
    push_tick!(aggregator, quote)

Fold a tick in, returning a completed bar when this tick belongs to a later bucket.
"""
function push_tick!(aggregator::BarAggregator, price::Quote)
    price.symbol == aggregator.symbol || throw(
        ArgumentError(
            string("aggregator for ", aggregator.symbol, " given ", price.symbol),
        ),
    )
    bucket = bucket_of(aggregator, price.timestamp)
    current = aggregator.bucket_start

    if current !== nothing && bucket < current
        throw(
            ArgumentError(
                string("a tick at ", price.timestamp, " is older than the open bar at ", current),
            ),
        )
    end

    completed = (current !== nothing && bucket > current) ? build_bar(aggregator) : nothing
    if current === nothing || bucket > current
        aggregator.bucket_start = bucket
        aggregator.open = price.last_price
        aggregator.high = price.last_price
        aggregator.low = price.last_price
        aggregator.volume = 0.0
        aggregator.n_ticks = 0
    end

    aggregator.high = max(aggregator.high, price.last_price)
    aggregator.low = min(aggregator.low, price.last_price)
    aggregator.close = price.last_price
    aggregator.volume += price.volume
    aggregator.n_ticks += 1
    return completed
end

"""
    build_bar(aggregator)

The open bar as a [`Bar`](@ref).
"""
function build_bar(aggregator::BarAggregator)
    start = aggregator.bucket_start
    start === nothing && throw(ArgumentError("no open bar to build"))
    return Bar(
        aggregator.symbol, start, aggregator.open, aggregator.high, aggregator.low,
        aggregator.close, aggregator.volume; interval = aggregator.label,
    )
end

"""
    flush!(aggregator)

Close the open bar and return it, or `nothing` if there is none.

For the end of a session, where no further tick is coming to close the last bucket.
"""
function flush!(aggregator::BarAggregator)
    has_open_bar(aggregator) || return nothing
    bar = build_bar(aggregator)
    aggregator.bucket_start = nothing
    aggregator.n_ticks = 0
    return bar
end

"""
    FeedSession

One symbol's live feed: health, aggregation, and the bars it has produced.
"""
mutable struct FeedSession
    symbol::String
    health::FeedHealth
    aggregator::BarAggregator
    bars::Vector{Bar}

    function FeedSession(
            symbol::AbstractString; interval::Period = Minute(1),
            label::AbstractString = "1m", max_silence::Period = Minute(2),
        )
        return new(
            String(symbol), FeedHealth(; max_silence = max_silence),
            BarAggregator(symbol; interval = interval, label = label), Bar[],
        )
    end
end

"""
    handle_tick!(session, quote)

Take one tick through health and aggregation.

Returns `(verdict, bar)`: what the health check made of the tick, and a completed bar if this
tick closed one. A rejected tick reaches the aggregator not at all, so a replayed history
cannot rewrite a bar the system has already acted on.
"""
function handle_tick!(session::FeedSession, price::Quote)
    verdict = accept!(session.health, price)
    verdict === :accepted || return verdict, nothing
    bar = push_tick!(session.aggregator, price)
    bar === nothing || push!(session.bars, bar)
    return verdict, bar
end

"""
    close_session!(session)

Flush the last bar at the end of trading.
"""
function close_session!(session::FeedSession)
    bar = flush!(session.aggregator)
    bar === nothing || push!(session.bars, bar)
    return bar
end

"""
    run_feed!(session, source; until)

Drain a source into a session.

Deterministic and finite: it stops when the source is exhausted, which is what makes a feed
testable without a market.
"""
function run_feed!(session::FeedSession, source::TickSource)
    while true
        price = next_tick!(source)
        price === nothing && break
        price.symbol == session.symbol || continue
        handle_tick!(session, price)
    end
    return session
end

Base.show(io::IO, health::FeedHealth) = @printf(
    io, "<FeedHealth accepted=%d duplicate=%d out_of_order=%d stale=%d>",
    health.n_accepted, health.n_duplicates, health.n_out_of_order, health.n_stale
)

Base.show(io::IO, session::FeedSession) = @printf(
    io, "<FeedSession %s bars=%d ticks=%d>",
    session.symbol, length(session.bars), session.health.n_accepted
)
