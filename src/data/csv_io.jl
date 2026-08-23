"""
CSV persistence for bars.

CSV rather than a binary format on purpose: a stored dataset should be inspectable with the
tools already on the machine. Every row is validated back through [`Bar`](@ref) on read, so a
hand-edited or truncated file fails at load with a line number rather than three layers
later.

Written by hand rather than through a CSV package. The format is eight fixed columns with no
quoting, embedded newlines or type inference to get wrong, and a trading system's dependency
list is a liability of its own.
"""

const BAR_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume", "interval")

"""
    BarFileError

Raised when a bar file cannot be read as bars.
"""
struct BarFileError <: Exception
    message::String
end

Base.showerror(io::IO, error::BarFileError) = print(io, "BarFileError: ", error.message)

"""
    write_bars(path, bars)

Write bars to `path`, oldest first. Returns the number of rows written.
"""
function write_bars(path::AbstractString, bars)
    ordered = sort(collect(bars), by = bar -> (bar.symbol, bar.timestamp))
    mkpath(dirname(abspath(path)))
    open(path, "w") do handle
        println(handle, join(BAR_COLUMNS, ','))
        for bar in ordered
            println(
                handle, join(
                    (
                        bar.symbol, bar.timestamp, bar.open, bar.high,
                        bar.low, bar.close, bar.volume, bar.interval,
                    ), ',',
                ),
            )
        end
    end
    return length(ordered)
end

"""
    read_bars(path)

Read bars from `path`, validating every row.
"""
function read_bars(path::AbstractString)
    isfile(path) || throw(BarFileError("no such bar file: $path"))
    lines = readlines(path)
    isempty(lines) && throw(BarFileError("$path is empty"))

    header = split(first(lines), ',')
    missing_columns = setdiff(collect(BAR_COLUMNS), header)
    isempty(missing_columns) ||
        throw(BarFileError("$path is missing columns: $(join(sort(missing_columns), ", "))"))
    index = Dict(String(name) => position for (position, name) in enumerate(header))

    bars = Bar[]
    for (offset, line) in enumerate(lines[2:end])
        isempty(strip(line)) && continue
        number = offset + 1
        fields = split(line, ',')
        length(fields) == length(header) ||
            throw(BarFileError("$path:$number: expected $(length(header)) fields"))
        try
            push!(
                bars, Bar(
                    String(fields[index["symbol"]]),
                    DateTime(fields[index["timestamp"]]),
                    parse(Float64, fields[index["open"]]),
                    parse(Float64, fields[index["high"]]),
                    parse(Float64, fields[index["low"]]),
                    parse(Float64, fields[index["close"]]),
                    parse(Float64, fields[index["volume"]]);
                    interval = String(fields[index["interval"]]),
                ),
            )
        catch error
            error isa InterruptException && rethrow()
            throw(BarFileError("$path:$number: $(sprint(showerror, error))"))
        end
    end
    return bars
end

"""
    save_store(store, directory)

Write one file per symbol under `directory`. Returns rows written per symbol.
"""
function save_store(store::BarStore, directory::AbstractString)
    written = Dict{String, Int}()
    for symbol in symbols(store)
        path = joinpath(directory, "$(safe_filename(symbol)).csv")
        written[symbol] = write_bars(path, load_range(store, symbol))
    end
    return written
end

"""
    load_store(directory)

Load every bar file under `directory` into a fresh store.
"""
function load_store(directory::AbstractString)
    isdir(directory) || throw(BarFileError("no such directory: $directory"))
    store = InMemoryBarStore()
    for name in sort(readdir(directory))
        endswith(name, ".csv") || continue
        upsert!(store, read_bars(joinpath(directory, name)))
    end
    return store
end

"""
    safe_filename(symbol)

Make a symbol safe as a filename without losing which symbol it was.
"""
safe_filename(symbol::AbstractString) = replace(String(symbol), '/' => '_', ':' => '_')
