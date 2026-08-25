"""
Rebuilding the position book after a restart.

The watermark in `ops/recovery.jl` answers "which bars has this system already acted on". It does
not answer "what does this system hold", and until now nothing did: a resumed session started
with an empty book and a correct watermark, which is safe in paper mode where the positions are
notional and wrong everywhere else.

What survives a restart is the journal. Every fill is in it, with its signed quantity, its price
and its fees, and the session's opening cash is in the first line. That is enough to reconstruct
the account by replaying it, which is the same reconstruction a ledger has always been for.

**The reconstruction is checked before it is trusted.** A book rebuilt from a journal with a
missing opening balance, or one whose reconstructed equity disagrees with the equity the journal
last recorded, is a book that describes something other than the account. Where that happens the
positions are not installed and the session refuses to trade, because a confident wrong book is
worse than an empty one.
"""

"""
    RebuiltAccount

An account reconstructed from a journal, and whether it can be believed.

`problems` is empty exactly when `consistent` is true. Both are carried because the caller needs
to act on the second and an operator needs to read the first.
"""
struct RebuiltAccount
    positions::Dict{String, Position}
    cash::Float64
    pending::Vector{Order}
    watermark::Union{DateTime, Nothing}
    n_fills::Int
    starting_cash::Union{Float64, Nothing}
    journalled_equity::Union{Float64, Nothing}
    rebuilt_equity::Union{Float64, Nothing}
    consistent::Bool
    problems::Vector{String}
end

Base.show(io::IO, account::RebuiltAccount) = @printf(
    io, "<RebuiltAccount %s positions=%d cash=%.2f fills=%d%s>",
    account.consistent ? "consistent" : "INCONSISTENT",
    length(account.positions), account.cash, account.n_fills,
    account.consistent ? "" : string(" [", join(account.problems, "; "), "]"),
)

"""
    REBUILD_EQUITY_TOLERANCE

How far the reconstructed equity may sit from the equity the journal last recorded.

Relative, and loose enough to absorb the difference between marking at the last close in the
journal and whatever the original process marked at, which is not always the same bar. Tight
enough that a missed fill shows.
"""
const REBUILD_EQUITY_TOLERANCE = 0.01

"""
    rebuild_account(path)

Replay a journal into an account.

Positions accumulate signed quantity and a weighted average price, the way a position is built in
the first place. A quantity that crosses through zero resets the average price rather than
averaging a long into a short, which is the arithmetic a broker uses and the arithmetic that
makes the number mean anything.
"""
function rebuild_account(path::AbstractString)
    problems = String[]
    isfile(path) || return RebuiltAccount(
        Dict{String, Position}(), 0.0, Order[], nothing, 0, nothing, nothing, nothing,
        false, ["no journal at $path"],
    )

    starting_cash = nothing
    cash = 0.0
    quantities = Dict{String, Float64}()
    averages = Dict{String, Float64}()
    opened = Dict{String, DateTime}()
    last_price = Dict{String, Float64}()
    pending = Dict{String, Order}()
    watermark = nothing
    journalled_equity = nothing
    n_fills = 0
    torn = 0

    for line in eachline(path)
        isempty(strip(line)) && continue
        entry = try
            JSON3.read(line)
        catch error
            error isa InterruptException && rethrow()
            torn += 1
            continue
        end
        entry isa AbstractDict || continue
        event = get(entry, "event", "")

        if event == "session_started"
            value = get(entry, "starting_cash", nothing)
            if value isa Real && isfinite(value)
                starting_cash = Float64(value)
                cash = starting_cash
            end
            continue
        end
        event == "bar" || continue

        stamp = get(entry, "as_of", nothing)
        moment = stamp isa AbstractString ? tryparse(DateTime, String(stamp)) : nothing
        if moment !== nothing && (watermark === nothing || moment > watermark)
            watermark = moment
        end

        equity = get(entry, "equity", nothing)
        equity isa Real && isfinite(equity) && (journalled_equity = Float64(equity))

        symbol = get(entry, "symbol", nothing)
        symbol isa AbstractString || continue
        name = String(symbol)
        close = get(entry, "close", nothing)
        close isa Real && isfinite(close) && (last_price[name] = Float64(close))

        filled = get(entry, "filled", nothing)
        (filled isa Real && isfinite(filled) && !iszero(filled)) || continue
        price = get(entry, "fill_price", nothing)
        if !(price isa Real) || !isfinite(price) || price <= 0
            push!(
                problems,
                string("a fill at ", something(moment, "an unknown time"), " has no price"),
            )
            continue
        end
        fees = get(entry, "fees", nothing)
        charge = fees isa Real && isfinite(fees) ? Float64(fees) : 0.0

        quantity = Float64(filled)
        n_fills += 1
        cash -= quantity * Float64(price) + charge

        held = get(quantities, name, 0.0)
        updated = held + quantity
        if iszero(held) || sign(held) != sign(updated) && !iszero(updated)
            # Opening, or crossing through zero into the other direction. Averaging a long
            # into a short produces a number that is neither.
            averages[name] = Float64(price)
            opened[name] = something(moment, DateTime(1970, 1, 1))
        elseif abs(updated) > abs(held)
            averages[name] =
                (
                abs(held) * get(averages, name, Float64(price)) +
                    abs(quantity) * Float64(price)
            ) / abs(updated)
        end
        quantities[name] = updated
    end

    torn > 0 && torn > 1 &&
        push!(problems, string(torn, " unreadable lines, not just a torn final write"))
    starting_cash === nothing &&
        push!(problems, "the journal has no session_started line, so the opening cash is unknown")

    positions = Dict{String, Position}()
    for (name, quantity) in quantities
        iszero(quantity) && continue
        mark = get(last_price, name, get(averages, name, 0.0))
        mark > 0 || (mark = get(averages, name, 1.0))
        positions[name] = Position(
            symbol = name, quantity = quantity,
            average_price = get(averages, name, mark),
            last_price = mark,
            opened_at = get(opened, name, something(watermark, DateTime(1970, 1, 1))),
        )
    end

    rebuilt_equity = starting_cash === nothing ? nothing :
        cash + sum(
            Float64[
                position.quantity * position.last_price for position in values(positions)
            ];
            init = 0.0,
        )

    if rebuilt_equity !== nothing && journalled_equity !== nothing
        reference = max(abs(journalled_equity), 1.0)
        gap = abs(rebuilt_equity - journalled_equity) / reference
        gap > REBUILD_EQUITY_TOLERANCE && push!(
            problems,
            string(
                "rebuilt equity ", round(rebuilt_equity; digits = 2),
                " disagrees with the journalled ", round(journalled_equity; digits = 2),
            ),
        )
    end

    any(!isfinite, values(quantities)) && push!(problems, "a rebuilt quantity is not finite")
    isfinite(cash) || push!(problems, "the rebuilt cash balance is not finite")

    return RebuiltAccount(
        positions, cash, collect(values(pending)), watermark, n_fills, starting_cash,
        journalled_equity, rebuilt_equity, isempty(problems), problems,
    )
end

"""
    rebuild_report(account)

The reconstruction as an operator reads it.
"""
function rebuild_report(account::RebuiltAccount)
    buffer = IOBuffer()
    println(buffer, "Rebuilt account")
    println(buffer, "-"^62)
    @printf(buffer, "%-20s %d\n", "fills replayed", account.n_fills)
    @printf(
        buffer, "%-20s %s\n", "opening cash",
        account.starting_cash === nothing ? "unknown" :
            string(round(account.starting_cash; digits = 2)),
    )
    @printf(buffer, "%-20s %.2f\n", "cash", account.cash)
    @printf(buffer, "%-20s %d\n", "positions", length(account.positions))
    for symbol in sort(collect(keys(account.positions)))
        position = account.positions[symbol]
        @printf(
            buffer, "  %-16s %12.4f @ %.4f\n", symbol, position.quantity,
            position.average_price,
        )
    end
    @printf(
        buffer, "%-20s %s\n", "rebuilt equity",
        account.rebuilt_equity === nothing ? "unknown" :
            string(round(account.rebuilt_equity; digits = 2)),
    )
    @printf(
        buffer, "%-20s %s\n", "journalled equity",
        account.journalled_equity === nothing ? "unknown" :
            string(round(account.journalled_equity; digits = 2)),
    )
    @printf(buffer, "%-20s %s\n", "watermark", something(account.watermark, "none"))
    println(buffer, "-"^62)
    if account.consistent
        println(buffer, "Consistent. Positions may be installed.")
    else
        println(buffer, "INCONSISTENT. Positions must not be installed, and the session")
        println(buffer, "must not trade. See docs/RUNBOOK.md, restart.")
        for problem in account.problems
            println(buffer, "  - ", problem)
        end
    end
    return String(take!(buffer))
end
