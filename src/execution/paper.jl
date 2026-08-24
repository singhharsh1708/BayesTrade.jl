"""
A broker that fills orders against recorded prices.

The point is not realism for its own sake. It is that every cost which makes a strategy look
better on paper than in life is charged here, so a paper record that looks good is evidence
rather than an artefact. Three of them:

* **slippage**, because a market order does not execute at the last printed price
* **commission and taxes**, which on Indian equities are a real fraction of a small edge
* **liquidity**, because an order larger than the bar's volume does not simply fill

A paper broker that skipped these would produce exactly the pleasant, wrong answer this whole
system exists to avoid.
"""

"""
    PaperCosts

What a trade costs beyond the price.

Defaults are Indian equity delivery figures at the time of writing, deliberately on the
pessimistic side: a paper record built on optimistic costs is not evidence of anything.
"""
Base.@kwdef struct PaperCosts
    commission_rate::Float64 = 0.0003
    minimum_commission::Float64 = 0.0
    slippage_rate::Float64 = 0.0005
    max_participation::Float64 = 0.1

    function PaperCosts(commission_rate, minimum_commission, slippage_rate, max_participation)
        for (name, value) in (
                :commission_rate => commission_rate, :slippage_rate => slippage_rate,
            )
            0 <= value < 1 ||
                throw(ArgumentError(string(name, " must lie in [0, 1), got ", value)))
        end
        minimum_commission >= 0 ||
            throw(ArgumentError("minimum_commission cannot be negative"))
        0 < max_participation <= 1 ||
            throw(ArgumentError("max_participation must lie in (0, 1]"))
        return new(
            Float64(commission_rate), Float64(minimum_commission), Float64(slippage_rate),
            Float64(max_participation),
        )
    end
end

"""
    PaperBroker

A simulated venue that keeps its own cash and positions.

The default everywhere. Reaching a real venue takes a different type, not a flag on this one.
"""
mutable struct PaperBroker <: Broker
    cash::Float64
    positions::Dict{String, Position}
    costs::PaperCosts
    fills::Vector{Fill}
    receipts::Vector{OrderReceipt}
    next_id::Int

    function PaperBroker(; starting_cash::Real = 1.0e6, costs::PaperCosts = PaperCosts())
        starting_cash > 0 ||
            throw(ArgumentError(string("starting_cash must be positive, got ", starting_cash)))
        return new(
            Float64(starting_cash), Dict{String, Position}(), costs, Fill[],
            OrderReceipt[], 1,
        )
    end
end

broker_mode(::PaperBroker) = PAPER

"""
    next_order_id!(broker)

A fresh identifier. Sequential rather than random so a replay is reproducible.
"""
function next_order_id!(broker::PaperBroker)
    id = string("paper-", lpad(broker.next_id, 8, '0'))
    broker.next_id += 1
    return id
end

"""
    fill_price(costs, side, reference)

Where a market order actually executes.

Slippage always moves against the order: a buy pays more, a sell receives less. Modelling it
as a spread around the reference price, which would sometimes help, would be modelling a
different market from the one anybody trades in.
"""
fill_price(costs::PaperCosts, side::OrderSide, reference::Float64) =
    side === BUY_SIDE ? reference * (1 + costs.slippage_rate) :
    reference * (1 - costs.slippage_rate)

"""
    commission(costs, notional)

What the venue and the state take.
"""
commission(costs::PaperCosts, notional::Float64) =
    max(costs.commission_rate * notional, costs.minimum_commission)

"""
    tradeable_quantity(costs, requested, volume)

How much of an order the bar could actually absorb.

An order larger than a sensible share of the bar's volume does not fill at the printed price;
it moves the price. Rather than pretend to model impact, the broker refuses to fill more than
`max_participation` of the volume and says so.
"""
function tradeable_quantity(costs::PaperCosts, requested::Float64, volume::Float64)
    volume <= 0 && return requested
    return min(requested, costs.max_participation * volume)
end

"""
    place_order!(broker, order, quote)

Fill an order against a quote, or refuse and say why.

Refusals are ordinary. A trading system that cannot handle its broker declining is a trading
system that has never met one.
"""
function place_order!(broker::PaperBroker, order::Order, price::Quote)
    order.symbol == price.symbol || throw(
        ArgumentError(
            string("order for ", order.symbol, " priced against ", price.symbol),
        ),
    )

    record(status, fill, detail) = begin
        receipt = OrderReceipt(order, status, fill, detail)
        push!(broker.receipts, receipt)
        receipt
    end

    reference = price.last_price
    executed = fill_price(broker.costs, order.side, reference)

    if order.order_type === LIMIT
        limit = order.limit_price
        if limit !== nothing
            # A limit order the market has not reached is not a rejection, it is an order
            # that is still working.
            if (is_buy(order) && executed > limit) || (!is_buy(order) && executed < limit)
                return record(
                    OPEN, nothing, string("limit ", limit, " not reached at ", reference),
                )
            end
            # Marketable, so it fills at the market rather than at the limit. A buy whose
            # limit sits above the offer pays the offer, not the limit, and filling at the
            # limit would charge a price nobody was asking. The limit is a bound on how bad
            # the fill may be, never a price to seek out.
            executed = is_buy(order) ? min(executed, limit) : max(executed, limit)
        end
    elseif order.order_type === STOP_LOSS || order.order_type === STOP_LOSS_MARKET
        # Not simulated. Filling one at the market ignores its trigger entirely, which
        # would make a stop look like protection it never provided.
        return record(
            REJECTED, nothing,
            string(slug(order.order_type), " is not simulated by the paper broker"),
        )
    end

    quantity = tradeable_quantity(broker.costs, order.quantity, price.volume)
    quantity > 0 || return record(REJECTED, nothing, "no volume to trade against")

    charge = commission(broker.costs, quantity * executed)
    if is_buy(order) && quantity * executed + charge > broker.cash
        return record(
            REJECTED, nothing,
            string(
                "needs ", round(quantity * executed + charge; digits = 2),
                " against ", round(broker.cash; digits = 2), " in cash",
            ),
        )
    end

    fill = Fill(
        order_id = order.id, symbol = order.symbol, side = order.side,
        quantity = quantity, price = executed, commission = charge,
        filled_at = price.timestamp,
    )
    apply!(broker, fill)
    push!(broker.fills, fill)

    partial = quantity < order.quantity
    return record(
        partial ? PARTIALLY_FILLED : FILLED, fill,
        partial ?
            string(
                "filled ", round(quantity; digits = 2), " of ", round(order.quantity; digits = 2),
                ", limited by volume",
            ) : "filled in full",
    )
end

"""
    apply!(broker, fill)

Move cash and positions to match a fill.

Averaging only on the way in. Adding to a position moves the average price; reducing one
leaves it alone, because the cost of what remains has not changed. Crossing through zero opens
a new position at the fill price, since the old one is gone.
"""
function apply!(broker::PaperBroker, fill::Fill)
    broker.cash += cash_flow(fill)
    delta = fill.side === BUY_SIDE ? fill.quantity : -fill.quantity
    existing = get(broker.positions, fill.symbol, nothing)

    if existing === nothing
        broker.positions[fill.symbol] = Position(
            symbol = fill.symbol, quantity = delta, average_price = fill.price,
            last_price = fill.price, opened_at = fill.filled_at,
        )
        return broker
    end

    combined = existing.quantity + delta
    if abs(combined) < 1.0e-9
        delete!(broker.positions, fill.symbol)
        return broker
    end

    average = if sign(combined) != sign(existing.quantity)
        fill.price
    elseif abs(combined) > abs(existing.quantity)
        (existing.quantity * existing.average_price + delta * fill.price) / combined
    else
        existing.average_price
    end

    broker.positions[fill.symbol] = Position(
        symbol = fill.symbol, quantity = combined, average_price = average,
        last_price = fill.price, opened_at = existing.opened_at, sector = existing.sector,
    )
    return broker
end

"""
    cancel_order!(broker, id)

Cancel a working order.

The paper broker fills or refuses immediately, so there is never anything working to cancel.
Saying so plainly is better than pretending to succeed.
"""
function cancel_order!(broker::PaperBroker, id::AbstractString)
    for receipt in broker.receipts
        if receipt.order.id == id
            receipt.status === OPEN &&
                return OrderReceipt(receipt.order, CANCELLED, nothing, "cancelled")
            return OrderReceipt(
                receipt.order, receipt.status, receipt.fill,
                string("cannot cancel an order that is ", slug(receipt.status)),
            )
        end
    end
    throw(ArgumentError(string("no order with id ", id)))
end

"""
    mark_to_market!(broker, quotes)

Reprice every position from the latest quotes.
"""
function mark_to_market!(broker::PaperBroker, quotes::AbstractVector{Quote})
    for price in quotes
        haskey(broker.positions, price.symbol) || continue
        held = broker.positions[price.symbol]
        broker.positions[price.symbol] = Position(
            symbol = held.symbol, quantity = held.quantity,
            average_price = held.average_price, last_price = price.last_price,
            opened_at = held.opened_at, sector = held.sector,
        )
    end
    return broker
end

"""
    equity(broker)

Cash plus the market value of everything held.
"""
function equity(broker::PaperBroker)
    total = broker.cash
    for position in values(broker.positions)
        total += exposure(position)
    end
    return total
end

"""
    portfolio(broker; peak_equity, day_start_equity, as_of)

The broker's state as a [`Portfolio`](@ref), which is what the risk engine rules on.
"""
portfolio(
    broker::PaperBroker; peak_equity::Union{Real, Nothing} = nothing,
    day_start_equity::Union{Real, Nothing} = nothing, as_of::DateTime,
) = Portfolio(
    equity = equity(broker), cash = broker.cash, positions = copy(broker.positions),
    peak_equity = peak_equity, day_start_equity = day_start_equity, as_of = as_of,
)

"""
    order_from_ruling(broker, ruling, price, equity)

Turn an approved weight into an order, or `nothing` when there is nothing to send.

The only place a weight becomes a quantity. It reads the **approved** weight and never the
requested one, so a trade the risk engine reduced cannot be restored by an arithmetic slip
here.
"""
function order_from_ruling(
        broker::PaperBroker, ruling::RiskRuling, price::Quote, account_equity::Real,
    )
    approved(ruling) || return nothing
    is_actionable(ruling.action) || return nothing
    account_equity > 0 || throw(ArgumentError("equity must be positive"))

    quantity = ruling.approved_weight * account_equity / price.last_price
    quantity > 0 || return nothing
    return Order(
        id = next_order_id!(broker), symbol = ruling.symbol,
        side = ruling.action === BUY ? BUY_SIDE : SELL_SIDE,
        quantity = quantity, order_type = MARKET, placed_at = price.timestamp,
    )
end

Base.show(io::IO, broker::PaperBroker) = @printf(
    io, "<PaperBroker cash=%.2f positions=%d fills=%d equity=%.2f>",
    broker.cash, length(broker.positions), length(broker.fills), equity(broker)
)
