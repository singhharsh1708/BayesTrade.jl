"""
Reconciling what we think we hold against what the venue says we hold.

**The venue is the authority.** Not because the local book is badly written, but because it is a
model of something else, and every way a model of a venue can drift from the venue has already
happened to somebody: a fill that arrived twice, a fill that never arrived, an order rejected
after the local book had already assumed it, a position opened by a human in the app, a process
that restarted holding a book from an hour ago.

None of those announce themselves. The local book stays perfectly self-consistent while
describing an account that no longer exists, and every decision made from it is sized against a
position that is not there.

So the rule here is narrow and absolute: **a mismatch stops new trading and nothing resolves it
automatically.** The temptation is to have the local book adopt the venue's numbers and carry on,
and that is precisely the wrong move. A quantity that differs by one lot could be a missed fill,
a duplicate fill, or a manual trade, and those have different consequences and different
remedies. Silently overwriting the local book erases the evidence of which one it was.

Nothing in this file can place an order. It reads two views of the same account and says whether
they agree.
"""

"""
    ExternalPosition

One position as the venue reports it.

Deliberately not a [`Position`](@ref). The local type carries an opening timestamp and a sector,
which are facts about our own bookkeeping rather than about the account, and a venue does not
have opinions about either. Reusing it would invite comparing fields the venue never sent.
"""
struct ExternalPosition
    symbol::String
    quantity::Float64
    average_price::Float64

    function ExternalPosition(;
            symbol::AbstractString, quantity::Real, average_price::Real,
        )
        isempty(symbol) && throw(ArgumentError("a position needs a symbol"))
        isfinite(quantity) ||
            throw(ArgumentError(string(symbol, ": quantity must be finite")))
        isfinite(average_price) && average_price > 0 || throw(
            ArgumentError(string(symbol, ": average_price must be finite and positive")),
        )
        return new(String(symbol), Float64(quantity), Float64(average_price))
    end
end

"""
    ExternalOrder

One order the venue still considers live.
"""
struct ExternalOrder
    id::String
    symbol::String
    side::OrderSide
    quantity::Float64
    status::OrderStatus

    function ExternalOrder(;
            id::AbstractString, symbol::AbstractString, side::OrderSide,
            quantity::Real, status::OrderStatus = OPEN,
        )
        isempty(id) && throw(ArgumentError("an order needs an identifier"))
        isempty(symbol) && throw(ArgumentError("an order needs a symbol"))
        isfinite(quantity) && quantity > 0 || throw(
            ArgumentError(string(id, ": quantity must be finite and positive")),
        )
        return new(String(id), String(symbol), side, Float64(quantity), status)
    end
end

"""
    VenueSnapshot

Everything the venue said, and when it said it.

`as_of` is carried because a snapshot has an age, and an old snapshot agreeing with the local
book proves nothing about now. A reconciliation that ignores the age of its evidence is a
reconciliation that passes during an outage.
"""
struct VenueSnapshot
    as_of::DateTime
    positions::Dict{String, ExternalPosition}
    pending::Vector{ExternalOrder}
    cash::Union{Float64, Nothing}

    function VenueSnapshot(;
            as_of::DateTime,
            positions::AbstractVector{ExternalPosition} = ExternalPosition[],
            pending::AbstractVector{ExternalOrder} = ExternalOrder[],
            cash::Union{Real, Nothing} = nothing,
        )
        keyed = Dict{String, ExternalPosition}()
        for position in positions
            haskey(keyed, position.symbol) && throw(
                ArgumentError(
                    string("the venue reported ", position.symbol, " twice"),
                ),
            )
            keyed[position.symbol] = position
        end
        return VenueSnapshot(
            as_of, keyed, collect(pending),
            cash === nothing ? nothing : Float64(cash),
        )
    end

    VenueSnapshot(
        as_of::DateTime, positions::Dict{String, ExternalPosition},
        pending::Vector{ExternalOrder}, cash::Union{Float64, Nothing},
    ) = new(as_of, positions, pending, cash)
end

"""
    AccountSource

Where a [`VenueSnapshot`](@ref) comes from.

An interface rather than a broker. Reading an account and placing an order are different
capabilities, and keeping them in different types is what lets reconciliation be built and tested
now without a live order path existing at all.
"""
abstract type AccountSource end

"""
    fetch_account(source, as_of)

The venue's view of the account, or an error.

Implementations return a [`VenueSnapshot`](@ref). Raising is a valid answer and is handled: an
account that cannot be read is [`RECONCILE_UNAVAILABLE`](@ref), never a match.
"""
fetch_account(source::AccountSource, ::DateTime) =
    throw(ArgumentError("$(typeof(source)) must define fetch_account"))

"""
    LocalAccount

The session's own view, in the shape reconciliation compares.
"""
struct LocalAccount
    as_of::DateTime
    positions::Dict{String, Position}
    pending::Vector{Order}
    cash::Union{Float64, Nothing}
end

"""
    local_account(broker, as_of; pending)

The local view, taken from a broker's own book.
"""
local_account(broker::Broker, as_of::DateTime; pending::AbstractVector{Order} = Order[]) =
    LocalAccount(as_of, copy(broker.positions), collect(pending), broker.cash)

"""
    Discrepancy

One way the two views disagree.

Every field is filled in even when one side has nothing to say, because a discrepancy is read
after the fact by somebody who was not there, and "local 100, venue absent" is a different
morning from "local absent, venue 100".
"""
struct Discrepancy
    kind::Symbol
    symbol::String
    local_value::Union{Float64, Nothing}
    venue_value::Union{Float64, Nothing}
    detail::String
end

"""
    ReconciliationStatus

Whether the books agree, disagree, or could not be compared.

Three states rather than two. An account that could not be read is not a match and it is not a
mismatch either: nothing was compared, and reporting it as either would be a claim the evidence
does not support.
"""
@enum ReconciliationStatus RECONCILE_MATCHED RECONCILE_MISMATCHED RECONCILE_UNAVAILABLE

"""
    Reconciliation

The result, with everything needed to act on it and to explain it later.
"""
struct Reconciliation
    status::ReconciliationStatus
    as_of::DateTime
    venue_as_of::Union{DateTime, Nothing}
    discrepancies::Vector{Discrepancy}
    n_local::Int
    n_venue::Int
    detail::String
end

"""
    reconciled(result)

Whether the books agreed. The only state that permits new trading.
"""
reconciled(result::Reconciliation) = result.status === RECONCILE_MATCHED

"""
    may_open_new_positions(result)

Fail closed. Anything other than a clean match stops new trading.

Reducing an existing position is a different question and is not answered here: unwinding
exposure while the books disagree may well be the right move, and it is a decision for an
operator with the runbook open, not for this function.
"""
may_open_new_positions(result::Reconciliation) = reconciled(result)

Base.show(io::IO, result::Reconciliation) = @printf(
    io, "<Reconciliation %s local=%d venue=%d%s>",
    replace(string(result.status), "RECONCILE_" => ""), result.n_local, result.n_venue,
    isempty(result.discrepancies) ? "" :
        string(" [", join(String[string(d.kind, ":", d.symbol) for d in result.discrepancies], ", "), "]"),
)

"""
    ReconciliationTolerances

How close counts as equal.

Quantities are compared exactly by default. A share count is an integer in disguise and a venue
that disagrees by a thousandth of one has not rounded, it has told us about a different account.

Average prices are compared on a relative tolerance, because they genuinely are a rounded
weighted mean on both sides and the last digit is not evidence of anything.

`max_staleness` is how old a snapshot may be before it stops being evidence about now.
"""
Base.@kwdef struct ReconciliationTolerances
    quantity::Float64 = 0.0
    average_price::Float64 = 1.0e-4
    max_staleness::Period = Minute(5)
end

"""
    reconcile(local_view, snapshot; tolerances, as_of)

Compare two views of one account.

Every check runs even after the first failure. A reconciliation that stops at the first
discrepancy tells an operator to fix one thing and try again, three times, when the whole picture
was available at the start.
"""
function reconcile(
        local_view::LocalAccount, snapshot::VenueSnapshot;
        tolerances::ReconciliationTolerances = ReconciliationTolerances(),
        as_of::DateTime = local_view.as_of,
    )
    discrepancies = Discrepancy[]

    age = as_of - snapshot.as_of
    if age > tolerances.max_staleness || age < Millisecond(0)
        push!(
            discrepancies,
            Discrepancy(
                :stale_snapshot, "", nothing, nothing,
                string(
                    "the venue snapshot is stamped ", snapshot.as_of,
                    " and the comparison is at ", as_of,
                ),
            ),
        )
    end

    symbols = sort(collect(union(keys(local_view.positions), keys(snapshot.positions))))
    for symbol in symbols
        held = get(local_view.positions, symbol, nothing)
        reported = get(snapshot.positions, symbol, nothing)

        if held === nothing
            # Both absent cannot happen: the symbol came from the union of the two key sets.
            # Written as a test rather than assumed, because the compiler cannot see that and
            # otherwise reads a field access on `Nothing` below.
            reported === nothing && continue
            # A venue row of exactly zero is a closed position the venue is still listing, not
            # a position. `Position` refuses a zero quantity for the same reason, so the two
            # sides agree that nothing is held and there is nothing to report.
            iszero(reported.quantity) && continue
            # The venue holds something we do not know about: a manual trade, a fill we never
            # saw, or a position from a process whose state we lost.
            push!(
                discrepancies,
                Discrepancy(
                    :unexpected_position, symbol, nothing, reported.quantity,
                    string("the venue reports ", reported.quantity, " and the session holds none"),
                ),
            )
            continue
        end
        if reported === nothing
            push!(
                discrepancies,
                Discrepancy(
                    :missing_position, symbol, held.quantity, nothing,
                    string("the session holds ", held.quantity, " and the venue reports none"),
                ),
            )
            continue
        end

        if abs(held.quantity - reported.quantity) > tolerances.quantity
            push!(
                discrepancies,
                Discrepancy(
                    :quantity_mismatch, symbol, held.quantity, reported.quantity,
                    string(
                        "the session holds ", held.quantity, " and the venue reports ",
                        reported.quantity,
                    ),
                ),
            )
        end

        # Only meaningful where both sides hold something. An average price on a flat position
        # is a leftover, not a fact about the account.
        if !iszero(held.quantity) && !iszero(reported.quantity)
            reference = max(abs(held.average_price), abs(reported.average_price))
            gap = abs(held.average_price - reported.average_price)
            if reference > 0 && gap / reference > tolerances.average_price
                push!(
                    discrepancies,
                    Discrepancy(
                        :average_price_mismatch, symbol, held.average_price,
                        reported.average_price,
                        string(
                            "average price ", round(held.average_price; digits = 4),
                            " against ", round(reported.average_price; digits = 4),
                        ),
                    ),
                )
            end
        end
    end

    local_orders = Dict(order.id => order for order in local_view.pending)
    venue_orders = Dict(order.id => order for order in snapshot.pending)
    for id in sort(collect(union(keys(local_orders), keys(venue_orders))))
        ours = get(local_orders, id, nothing)
        theirs = get(venue_orders, id, nothing)
        # The identifier came from the union of both key sets, so at least one side has it.
        # Written as nested tests rather than a short-circuit guard: the guard reads better and
        # does not narrow the union for the compiler, which then sees a field access on
        # `Nothing` in a branch that cannot be reached.
        if ours === nothing
            theirs === nothing && continue
            push!(
                discrepancies,
                Discrepancy(
                    :unexpected_order, theirs.symbol, nothing, theirs.quantity,
                    string("the venue has order ", id, " live and the session does not"),
                ),
            )
        elseif theirs === nothing
            push!(
                discrepancies,
                Discrepancy(
                    :missing_order, ours.symbol, ours.quantity, nothing,
                    string(
                        "the session believes order ", id,
                        " is live and the venue does not",
                    ),
                ),
            )
        elseif ours.quantity != theirs.quantity || ours.side !== theirs.side
            push!(
                discrepancies,
                Discrepancy(
                    :order_mismatch, ours.symbol, ours.quantity, theirs.quantity,
                    string(
                        "order ", id, ": session ", slug(ours.side), " ", ours.quantity,
                        ", venue ", slug(theirs.side), " ", theirs.quantity,
                    ),
                ),
            )
        end
    end

    status = isempty(discrepancies) ? RECONCILE_MATCHED : RECONCILE_MISMATCHED
    detail = isempty(discrepancies) ? "the books agree" :
        string(length(discrepancies), " discrepancies; new trading must stop")
    return Reconciliation(
        status, as_of, snapshot.as_of, discrepancies,
        length(local_view.positions), length(snapshot.positions), detail,
    )
end

"""
    reconcile(local_view, source; tolerances, as_of)

Fetch the venue's view and compare it.

A source that raises produces [`RECONCILE_UNAVAILABLE`](@ref) rather than an exception escaping.
An account that cannot be read is exactly the situation reconciliation exists for, and a caller
who has to wrap this in a try block to stay safe will one day forget to.
"""
function reconcile(
        local_view::LocalAccount, source::AccountSource;
        tolerances::ReconciliationTolerances = ReconciliationTolerances(),
        as_of::DateTime = local_view.as_of,
    )
    snapshot = try
        fetch_account(source, as_of)
    catch error
        error isa InterruptException && rethrow()
        return Reconciliation(
            RECONCILE_UNAVAILABLE, as_of, nothing,
            Discrepancy[
                Discrepancy(
                    :unavailable, "", nothing, nothing, sprint(showerror, error),
                ),
            ],
            length(local_view.positions), 0,
            "the venue could not be read; new trading must stop",
        )
    end
    snapshot isa VenueSnapshot || return Reconciliation(
        RECONCILE_UNAVAILABLE, as_of, nothing,
        Discrepancy[
            Discrepancy(
                :unavailable, "", nothing, nothing,
                string("the source returned a ", typeof(snapshot), ", not a VenueSnapshot"),
            ),
        ],
        length(local_view.positions), 0,
        "the venue could not be read; new trading must stop",
    )
    return reconcile(local_view, snapshot; tolerances = tolerances, as_of = as_of)
end

"""
    reconciliation_record(result)

The result as a journal entry.

A mismatch that is not written down is a mismatch nobody can investigate tomorrow, and by then
the venue's view will have moved on.
"""
reconciliation_record(result::Reconciliation) = Dict{String, Any}(
    "event" => "reconciliation",
    "status" => lowercase(replace(string(result.status), "RECONCILE_" => "")),
    "as_of" => string(result.as_of),
    "venue_as_of" => result.venue_as_of === nothing ? nothing : string(result.venue_as_of),
    "n_local" => result.n_local,
    "n_venue" => result.n_venue,
    "detail" => result.detail,
    "discrepancies" => [
        Dict{String, Any}(
                "kind" => string(item.kind),
                "symbol" => item.symbol,
                "local" => item.local_value,
                "venue" => item.venue_value,
                "detail" => item.detail,
            ) for item in result.discrepancies
    ],
)

"""
    reconciliation_report(result)

The result as an operator reads it at three in the morning.
"""
function reconciliation_report(result::Reconciliation)
    buffer = IOBuffer()
    println(buffer, "Reconciliation at ", result.as_of)
    println(buffer, "-"^62)
    @printf(
        buffer, "%-16s %s\n", "status",
        replace(string(result.status), "RECONCILE_" => ""),
    )
    @printf(buffer, "%-16s %s\n", "venue snapshot", something(result.venue_as_of, "none"))
    @printf(buffer, "%-16s %d local, %d venue\n", "positions", result.n_local, result.n_venue)
    if isempty(result.discrepancies)
        println(buffer, "-"^62)
        println(buffer, "The books agree. New trading permitted.")
        return String(take!(buffer))
    end
    println(buffer, "-"^62)
    for item in result.discrepancies
        @printf(
            buffer, "%-24s %-10s %s\n", string(item.kind),
            isempty(item.symbol) ? "-" : item.symbol, item.detail,
        )
    end
    println(buffer, "-"^62)
    println(buffer, "NEW TRADING MUST STOP. See docs/RUNBOOK.md, broker and API failure.")
    println(buffer, "Do not adopt the venue's numbers to clear this. Establish which side is")
    println(buffer, "wrong first; overwriting the local book erases the evidence.")
    return String(take!(buffer))
end
