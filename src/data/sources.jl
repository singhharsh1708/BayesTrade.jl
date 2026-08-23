"""
The contract every market data vendor is adapted to, and the runs that use it.

A source fetches bars for one symbol over one window. It does not decide what to fetch,
where to put the result, or what to do when a vendor misbehaves; those belong to the
ingestion run, the store and the retry wrapper respectively.

Keeping the interface this narrow is what makes the vendor replaceable. Everything above
this line works against `BarSource`, so swapping a synthetic feed for Zerodha is a
constructor change and nothing else.

Only the synthetic source lives here. A real vendor needs an HTTP client, and a trading
system's dependency list is a liability of its own, so network adapters arrive as a package
extension rather than as a hard dependency of the core.
"""

"""
    DataSourceError

A vendor request failed.
"""
abstract type DataSourceError <: Exception end

"""
    SymbolNotFoundError

The vendor does not know this symbol.

Distinct from a transient failure: retrying will not help, and a retry wrapper must not
waste attempts on it.
"""
struct SymbolNotFoundError <: DataSourceError
    message::String
end

"""
    TransientSourceError

A failure that may succeed on a retry: a timeout, a rate limit, or a 5xx.
"""
struct TransientSourceError <: DataSourceError
    message::String
end

Base.showerror(io::IO, error::SymbolNotFoundError) =
    print(io, "SymbolNotFoundError: ", error.message)
Base.showerror(io::IO, error::TransientSourceError) =
    print(io, "TransientSourceError: ", error.message)

"""
    BarSource

Fetches historical bars for one symbol.

Implementations return bars stamped with the local canonical symbol, never the vendor's
spelling, so nothing downstream has to know which vendor was used.
"""
abstract type BarSource end

"""
    source_name(source)

The vendor this source speaks to, as it should appear in a report.
"""
source_name(source::BarSource) =
    throw(ArgumentError("$(typeof(source)) must define source_name"))

"""
    supported_intervals(source)

Intervals this source can provide.
"""
supported_intervals(::BarSource) = Set(["1d"])

supports(source::BarSource, interval::AbstractString) =
    interval in supported_intervals(source)

"""
    fetch_bars(source, symbol; start, stop, interval)

Bars for `symbol` between `start` and `stop` inclusive, oldest first.
"""
fetch_bars(source::BarSource, symbol::AbstractString; kwargs...) =
    throw(ArgumentError("$(typeof(source)) must define fetch_bars"))

function check_window(source::BarSource, start::Date, stop::Date, interval::AbstractString)
    start <= stop || throw(ArgumentError("start $start is after stop $stop"))
    supports(source, interval) || throw(
        ArgumentError(
            string(
                source_name(source), " does not provide \"", interval, "\" bars; ",
                "it supports ", join(sort(collect(supported_intervals(source))), ", "),
            ),
        ),
    )
    return nothing
end

const MAX_GENERATED_BARS = 20_000

"""
    SyntheticSource

Generates bars on demand, with no network and no credentials.

Every symbol gets its own path, derived deterministically from the symbol name, so a universe
looks like a universe rather than the same series under several tickers. The derivation is a
content hash rather than `hash`, which is salted per session and would make a run
irreproducible.
"""
Base.@kwdef struct SyntheticSource{P <: ReturnProcess} <: BarSource
    process::P = GaussianReturns()
    seed::Int = 7
    initial_price::Float64 = 1000.0
    shape::BarShape = BarShape()
    vary_by_symbol::Bool = true
end

source_name(::SyntheticSource) = "synthetic"

"""
    seed_for(source, symbol)

A stable per-symbol seed, so one symbol's path never depends on another's.
"""
function seed_for(source::SyntheticSource, symbol::AbstractString)
    source.vary_by_symbol || return source.seed
    digest = stable_hash(Dict("seed" => source.seed, "symbol" => String(symbol)))
    return parse(Int, digest[1:8], base = 16)
end

function fetch_bars(
        source::SyntheticSource, symbol::AbstractString;
        start::Date, stop::Date, interval::AbstractString = "1d",
    )
    check_window(source, start, stop, interval)
    n_bars = count(is_trading_day, start:Day(1):stop)
    n_bars == 0 && return Bar[]
    n_bars > MAX_GENERATED_BARS && throw(
        ArgumentError(
            "window of $n_bars bars exceeds the $MAX_GENERATED_BARS bar generation limit",
        ),
    )
    series = generate_series(
        source.process; symbol = symbol, n_bars = n_bars, start = start,
        initial_price = source.initial_price, seed = seed_for(source, symbol),
        shape = source.shape, interval = interval,
    )
    return series.bars
end

"""
    RetryingSource

Wraps a source, retrying transient failures with exponential backoff.

Only [`TransientSourceError`](@ref) is retried. An unknown symbol will still be unknown on
the third attempt, and burning the retry budget on it delays every symbol behind it in the
run.

`sleeper` is injected so the backoff schedule is testable without a suite that takes seven
seconds to prove a delay doubles.
"""
struct RetryingSource{S <: BarSource, F} <: BarSource
    inner::S
    attempts::Int
    backoff_seconds::Float64
    max_backoff_seconds::Float64
    sleeper::F

    function RetryingSource(
            inner::S;
            attempts::Integer = 3,
            backoff_seconds::Real = 1.0,
            max_backoff_seconds::Real = 30.0,
            sleeper::F = sleep,
        ) where {S <: BarSource, F}
        attempts >= 1 ||
            throw(ArgumentError("attempts must be at least 1, got $attempts"))
        backoff_seconds >= 0 ||
            throw(ArgumentError("backoff_seconds must be non-negative"))
        return new{S, F}(
            inner, Int(attempts), Float64(backoff_seconds),
            Float64(max_backoff_seconds), sleeper,
        )
    end
end

"""
    source_name(source::RetryingSource)

Report the vendor, not the wrapper. A retry is plumbing, not a data source.
"""
source_name(source::RetryingSource) = source_name(source.inner)
supported_intervals(source::RetryingSource) = supported_intervals(source.inner)

"""
    delay_for(source, attempt)

Backoff before the attempt after `attempt`, capped.
"""
delay_for(source::RetryingSource, attempt::Integer) =
    min(source.backoff_seconds * 2.0^attempt, source.max_backoff_seconds)

function fetch_bars(source::RetryingSource, symbol::AbstractString; kwargs...)
    # The final failure is thrown from inside the loop rather than from a variable carried
    # out of it. Carrying it out leaves a value the compiler cannot prove was ever assigned,
    # and reading a field off it is exactly the kind of latent nothing-dereference that only
    # fires on the unlucky path.
    for attempt in 0:(source.attempts - 1)
        try
            return fetch_bars(source.inner, symbol; kwargs...)
        catch error
            error isa TransientSourceError || rethrow()
            if attempt + 1 >= source.attempts
                throw(
                    TransientSourceError(
                        string(
                            symbol, ": ", source_name(source.inner), " failed ",
                            source.attempts, " times: ", error.message,
                        ),
                    ),
                )
            end
            source.sleeper(delay_for(source, attempt))
        end
    end
    throw(
        TransientSourceError(
            string(symbol, ": ", source_name(source.inner), " made no attempt"),
        ),
    )
end

"""
    IngestionReport

What an ingestion run did, per symbol.
"""
Base.@kwdef struct IngestionReport
    source::String
    start::Date
    stop::Date
    interval::String
    written::Dict{String, Int} = Dict{String, Int}()
    failures::Dict{String, String} = Dict{String, String}()
end

total_written(report::IngestionReport) = sum(values(report.written); init = 0)
succeeded(report::IngestionReport) = sort!(collect(keys(report.written)))
failed(report::IngestionReport) = sort!(collect(keys(report.failures)))
is_complete(report::IngestionReport) = isempty(report.failures)

function summarise(report::IngestionReport)
    count = length(report.written)
    head = string(
        report.source, ": ", total_written(report), " bars for ", count,
        count == 1 ? " symbol" : " symbols",
        " (", report.start, " to ", report.stop, ", ", report.interval, ")",
    )
    is_complete(report) && return head
    detail = join(
        ["$symbol: $(report.failures[symbol])" for symbol in failed(report)], "; ",
    )
    return string(head, "\n  failed: ", detail)
end

"""
    ingest!(source, store, symbols; start, stop, interval)

Fetch every symbol into `store`, collecting failures rather than raising.

A universe of fifty where one has been delisted should ingest forty-nine and say which one it
could not, rather than leaving the store in a half-filled state that nobody can reason about.
"""
function ingest!(
        source::BarSource, store::BarStore, wanted;
        start::Date, stop::Date, interval::AbstractString = "1d",
    )
    written = Dict{String, Int}()
    failures = Dict{String, String}()

    for symbol in wanted
        bars = try
            fetch_bars(source, symbol; start = start, stop = stop, interval = interval)
        catch error
            (error isa DataSourceError || error isa ArgumentError) || rethrow()
            failures[String(symbol)] = sprint(showerror, error)
            continue
        end
        if isempty(bars)
            failures[String(symbol)] = "vendor returned no bars"
            continue
        end
        written[String(symbol)] = upsert!(store, bars)
    end

    return IngestionReport(
        source = source_name(source), start = start, stop = stop,
        interval = interval, written = written, failures = failures,
    )
end
