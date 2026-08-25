"""
Groww as a bar source.

The vendor side of the [`BarSource`](@ref) contract: fetch candles for one symbol over one
window, hand back `Bar`s stamped with the local canonical symbol. Everything above this line
already works against `BarSource`, so pointing a backtest at real NSE history is a constructor
change and nothing else.

Four things this adapter has to get right, and each of them is a way a backtest quietly
becomes wrong rather than a way it fails:

**Windows.** Groww caps how much history one request may span, and the cap depends on the
interval. A window wider than the cap is split, fetched in pieces and stitched, because the
alternative is a truncated series that still looks like a series.

**Duplicates.** Stitched pieces meet at their edges and the endpoints are inclusive, so the
same candle can arrive twice. Two bars at one timestamp is a repeated observation, which moves
every estimate that counts observations.

**Unfinished candles.** The candle covering the current period is still forming. Feeding a
half-formed bar to a backtest is a look-ahead bug wearing the clothes of a data problem: its
close is not the close, and the model trains on a number that never existed at that time. Any
candle whose period has not demonstrably ended is dropped.

**Bad candles.** A vendor sends a high below a low occasionally. `Bar` refuses to construct,
which is correct, and this layer decides whether that ends the fetch or is reported and
skipped.
"""

"""
    MalformedBarError

A vendor sent a candle that cannot be a bar.

Not transient, so a retry wrapper must not spend attempts on it: the same candle will be just
as impossible on the third request.
"""
struct MalformedBarError <: DataSourceError
    message::String
end

Base.showerror(io::IO, error::MalformedBarError) =
    print(io, "MalformedBarError: ", error.message)

"""
    GROWW_INTERVALS

This package's interval labels, mapped to the strings Groww's API expects.

The mapping is a table rather than string surgery because the two vocabularies genuinely
disagree: `"1d"` here is `"1day"` there, and deriving one from the other by rule works until
the month, which is `"1month"` while the local label is `"1mo"`.
"""
const GROWW_INTERVALS = Dict{String, String}(
    "1m" => "1minute", "2m" => "2minute", "3m" => "3minute", "5m" => "5minute",
    "10m" => "10minute", "15m" => "15minute", "30m" => "30minute",
    "1h" => "1hour", "4h" => "4hour",
    "1d" => "1day", "1w" => "1week", "1mo" => "1month",
)

"""
    GROWW_MAX_WINDOW_DAYS

How many days of history Groww will serve in one request, per interval.

Published limits: thirty days up to the five minute candle, ninety through the half hour, a
hundred and eighty for anything hourly or longer. Exceeded, the request fails; respected, a
long fetch is simply several requests.
"""
const GROWW_MAX_WINDOW_DAYS = Dict{String, Int}(
    "1m" => 30, "2m" => 30, "3m" => 30, "5m" => 30,
    "10m" => 90, "15m" => 90, "30m" => 90,
    "1h" => 180, "4h" => 180, "1d" => 180, "1w" => 180, "1mo" => 180,
)

"""
    GrowwSource

Historical candles from Groww, for one exchange and segment.

`symbols` maps this package's canonical ticker to Groww's spelling for the cases the default
rule does not cover. The default rule is `"NSE-WIPRO"` for `WIPRO` on the NSE, which is what
Groww's own documentation uses.

`clock` returns exchange-local time and exists so the unfinished-candle rule can be tested
without waiting for a market to close. Its default is UTC shifted by the exchange offset rather
than `now()`, because the machine running a backtest is not necessarily in the exchange's
timezone and `now()` would silently be someone else's afternoon.
"""
struct GrowwSource{S <: GrowwSession, C, F} <: BarSource
    session::S
    exchange::String
    segment::String
    symbols::Dict{String, String}
    hours::MarketHours
    clock::C
    strict::Bool
    pause_seconds::Float64
    sleeper::F

    function GrowwSource(
            session::S;
            exchange::AbstractString = "NSE",
            segment::AbstractString = "CASH",
            symbols::Dict{String, String} = Dict{String, String}(),
            hours::MarketHours = MarketHours(),
            clock::C = () -> now(UTC) + IST_OFFSET,
            strict::Bool = true,
            pause_seconds::Real = 0.2,
            sleeper::F = sleep,
        ) where {S <: GrowwSession, C, F}
        isempty(exchange) && throw(ArgumentError("exchange is required"))
        isempty(segment) && throw(ArgumentError("segment is required"))
        pause_seconds >= 0 ||
            throw(ArgumentError("pause_seconds must be non-negative"))
        return new{S, C, F}(
            session, String(exchange), String(segment), symbols, hours, clock,
            strict, Float64(pause_seconds), sleeper,
        )
    end
end

source_name(::GrowwSource) = "groww"
supported_intervals(::GrowwSource) = Set(keys(GROWW_INTERVALS))

"""
    vendor_symbol(source, symbol)

Groww's spelling of a local ticker.
"""
vendor_symbol(source::GrowwSource, symbol::AbstractString) =
    get(source.symbols, String(symbol), string(source.exchange, "-", symbol))

format_moment(moment::DateTime) = Dates.format(moment, "yyyy-mm-dd HH:MM:SS")

"""
    window_chunks(start, stop, span_days)

Split a date range into requestable pieces, oldest first.

The pieces do not overlap, which keeps the duplicate problem smaller, but they are not trusted
not to: the endpoints Groww treats as inclusive are its own business, and deduplication happens
on the returned candles regardless.
"""
function window_chunks(start::Date, stop::Date, span_days::Int)
    chunks = Tuple{Date, Date}[]
    cursor = start
    while cursor <= stop
        finish = min(cursor + Day(span_days - 1), stop)
        push!(chunks, (cursor, finish))
        cursor = finish + Day(1)
    end
    return chunks
end

"""
    candle_moment(source, raw)

The timestamp of a candle, in exchange-local time.

Groww sends either an ISO string already in exchange time or epoch seconds, depending on the
endpoint and, apparently, the era. Both are accepted, and epoch seconds go through the one
conversion this package has for the purpose rather than a second one written here.
"""
function candle_moment(source::GrowwSource, raw)
    raw isa Real && return exchange_time(source.hours, raw)
    if raw isa AbstractString
        moment = tryparse(DateTime, replace(strip(String(raw)), " " => "T"))
        moment === nothing &&
            throw(MalformedBarError(string("unreadable candle timestamp \"", raw, "\"")))
        return moment
    end
    return throw(
        MalformedBarError(string("candle timestamp is a ", typeof(raw), ", not a time")),
    )
end

"""
    period_end(source, interval, opened)

When the period a candle covers finishes.

Minute and hour candles end a fixed span after they open. A daily candle ends at the session
close, not at midnight, so a completed day is usable the same afternoon rather than after
another eight hours of waiting. Weekly and monthly candles are given their nominal span, which
is conservative: it withholds a finished candle for a while rather than ever serving an
unfinished one.
"""
function period_end(source::GrowwSource, interval::AbstractString, opened::DateTime)
    interval == "1d" && return DateTime(Date(opened)) +
        Hour(hour(source.hours.close)) + Minute(minute(source.hours.close)) +
        Second(second(source.hours.close))
    interval == "1w" && return DateTime(Date(opened)) + Day(7)
    interval == "1mo" && return DateTime(Date(opened)) + Month(1)
    interval == "1h" && return opened + Hour(1)
    interval == "4h" && return opened + Hour(4)
    return opened + Minute(parse(Int, interval[1:(end - 1)]))
end

"""
    bar_from_candle(source, symbol, interval, raw)

One candle as a `Bar`, or `nothing` when it is unusable and the source is not strict.

Volume is absent on index candles, which is a real state rather than an error, and is read as
zero. The seventh field is open interest, which this package has no use for.
"""
function bar_from_candle(
        source::GrowwSource, symbol::AbstractString, interval::AbstractString, raw,
    )
    (raw isa AbstractVector && length(raw) >= 6) || throw(
        MalformedBarError(
            string(symbol, ": a candle should be six or more fields, got ", raw),
        ),
    )
    moment = candle_moment(source, raw[1])
    prices = ntuple(index -> raw[index + 1], 4)
    all(value -> value isa Real && isfinite(value), prices) || throw(
        MalformedBarError(string(symbol, " at ", moment, ": prices are ", prices)),
    )
    volume = raw[6]
    quantity = volume isa Real && isfinite(volume) ? Float64(volume) : 0.0

    return try
        Bar(
            symbol, moment, prices[1], prices[2], prices[3], prices[4], quantity;
            interval = interval,
        )
    catch error
        error isa ArgumentError || rethrow()
        message = string(symbol, " at ", moment, ": ", error.msg)
        source.strict && throw(MalformedBarError(message))
        @warn "dropping a candle that cannot be a bar" detail = message
        nothing
    end
end

"""
    translate_error(error)

Groww's refusals, sorted into the vendor-neutral kinds the ingestion run knows about.

The distinction that matters is retryable or not. A rate limit and a gateway timeout will pass;
an unknown symbol and a rejected token will not, and spending three attempts proving it delays
every symbol behind this one in the run.
"""
function translate_error(error::GrowwError)
    # Minting a token and being allowed to read market data are separate things on Groww, and
    # the reply for the second does not say so. Without the hint this reads as a bug in the
    # client, which is where the time then goes.
    error.status == 403 && return GrowwError(
        403, error.code,
        string(
            error.message,
            " Market data on Groww (quotes, OHLC and historical candles) needs an active",
            " Trading API subscription. Authentication succeeding does not imply access to it.",
        ),
    )
    error.status == 404 && return SymbolNotFoundError(error.message)
    error.status in (429, 500, 502, 503, 504) &&
        return TransientSourceError(string(error.status, ": ", error.message))
    error.status == 0 && return TransientSourceError(error.message)
    return error
end

"""
    fetch_bars(source::GrowwSource, symbol; start, stop, interval)

Candles for `symbol` between `start` and `stop` inclusive, oldest first, complete only.
"""
function fetch_bars(
        source::GrowwSource, symbol::AbstractString;
        start::Date, stop::Date, interval::AbstractString = "1d",
    )
    check_window(source, start, stop, interval)
    label = GROWW_INTERVALS[String(interval)]
    ticker = vendor_symbol(source, symbol)
    chunks = window_chunks(start, stop, GROWW_MAX_WINDOW_DAYS[String(interval)])

    # Keyed by timestamp rather than appended: the pieces meet at their edges, and two bars at
    # one moment is a repeated observation that moves every estimate downstream.
    collected = Dict{DateTime, Bar}()
    for (index, (from, to)) in enumerate(chunks)
        index == 1 || source.pause_seconds == 0 || source.sleeper(source.pause_seconds)
        payload = try
            groww_call(
                source.session,
                build_request(
                    source.session, :GET, "/historical/candles";
                    query = Dict{String, String}(
                        "exchange" => source.exchange,
                        "segment" => source.segment,
                        "groww_symbol" => ticker,
                        "start_time" => format_moment(DateTime(from)),
                        "end_time" => format_moment(
                            DateTime(to) + Hour(23) + Minute(59) + Second(59),
                        ),
                        "candle_interval" => label,
                    ),
                ),
            )
        catch error
            error isa GrowwError ? throw(translate_error(error)) : rethrow()
        end

        candles = get(payload, "candles", nothing)
        candles === nothing && continue
        candles isa AbstractVector || throw(
            MalformedBarError(
                string(symbol, ": candles came back as a ", typeof(candles), ", not a list"),
            ),
        )
        for raw in candles
            bar = bar_from_candle(source, symbol, String(interval), raw)
            bar === nothing && continue
            haskey(collected, bar.timestamp) || (collected[bar.timestamp] = bar)
        end
    end

    # The period covering right now is still forming. Its close is not a close, and a model
    # trained on it learns from a number that never existed at that time.
    moment = source.clock()
    usable = [
        bar for bar in values(collected)
            if period_end(source, String(interval), bar.timestamp) <= moment
    ]
    sort!(usable, by = bar -> bar.timestamp)
    return usable
end
