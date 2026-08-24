"""
The broker boundary.

Everything above this line reasons about weights and probabilities. Everything below it deals
in quantities, prices and order identifiers. The abstraction exists so that the paper broker
and a real one are interchangeable at the call site, and so that the live path is the same code
that has already been run ten thousand times against the simulator.

Execution does not re-decide anything. It is handed an approved size and it turns that into an
order, or it refuses because the market cannot accommodate it. It never asks whether the trade
was a good idea.
"""

"""
    Order

An instruction to the broker, as placed.
"""
struct Order
    id::String
    symbol::String
    side::OrderSide
    quantity::Float64
    order_type::OrderType
    limit_price::Union{Float64, Nothing}
    placed_at::DateTime

    function Order(;
            id::AbstractString,
            symbol::AbstractString,
            side::OrderSide,
            quantity::Real,
            order_type::OrderType = MARKET,
            limit_price::Union{Real, Nothing} = nothing,
            placed_at::DateTime,
        )
        isempty(id) && throw(ArgumentError("an order needs an identifier"))
        isempty(symbol) && throw(ArgumentError("an order needs a symbol"))
        # Quantity is unsigned and the direction is the side. A signed quantity plus a side
        # is two sources of truth for one fact, and they can disagree.
        quantity > 0 ||
            throw(ArgumentError(string("quantity must be positive, got ", quantity)))
        isfinite(quantity) || throw(ArgumentError("quantity must be finite"))
        if order_type === LIMIT
            limit_price === nothing &&
                throw(ArgumentError("a limit order needs a limit price"))
            limit_price > 0 || throw(ArgumentError("limit_price must be positive"))
        end
        return new(
            String(id), String(symbol), side, Float64(quantity), order_type,
            limit_price === nothing ? nothing : Float64(limit_price), placed_at,
        )
    end
end

is_buy(order::Order) = order.side === BUY_SIDE
signed_quantity(order::Order) = is_buy(order) ? order.quantity : -order.quantity

"""
    Fill

What actually happened, which is rarely exactly what was asked for.
"""
struct Fill
    order_id::String
    symbol::String
    side::OrderSide
    quantity::Float64
    price::Float64
    commission::Float64
    filled_at::DateTime

    function Fill(;
            order_id::AbstractString,
            symbol::AbstractString,
            side::OrderSide,
            quantity::Real,
            price::Real,
            commission::Real,
            filled_at::DateTime,
        )
        quantity > 0 || throw(ArgumentError("a fill needs a positive quantity"))
        price > 0 || throw(ArgumentError("a fill needs a positive price"))
        commission >= 0 || throw(ArgumentError("commission cannot be negative"))
        return new(
            String(order_id), String(symbol), side, Float64(quantity), Float64(price),
            Float64(commission), filled_at,
        )
    end
end

"""
    cash_flow(fill)

Signed effect on cash: a buy costs, a sell raises, and the commission is paid either way.
"""
cash_flow(fill::Fill) =
    (fill.side === BUY_SIDE ? -1 : 1) * fill.quantity * fill.price - fill.commission

"""
    OrderReceipt

What the broker did with an order.
"""
struct OrderReceipt
    order::Order
    status::OrderStatus
    fill::Union{Fill, Nothing}
    detail::String

    function OrderReceipt(order::Order, status::OrderStatus, fill, detail)
        if status === FILLED
            fill === nothing || fill.quantity ≈ order.quantity ||
                throw(ArgumentError("a filled order must be filled in full"))
            fill === nothing && throw(ArgumentError("a filled order needs a fill"))
        end
        status === REJECTED && fill !== nothing &&
            throw(ArgumentError("a rejected order cannot have a fill"))
        return new(order, status, fill, String(detail))
    end
end

was_filled(receipt::OrderReceipt) =
    receipt.status === FILLED || receipt.status === PARTIALLY_FILLED

"""
    Broker

What execution needs from a venue, paper or real.

A concrete broker defines [`place_order!`](@ref), [`cancel_order!`](@ref) and
[`broker_mode`](@ref). The mode is on the broker rather than passed in, so a paper broker
cannot be handed a live flag and quietly act on it.
"""
abstract type Broker end

place_order!(broker::Broker, ::Order, ::Quote) = throw(
    ArgumentError("$(typeof(broker)) must define place_order!"),
)
cancel_order!(broker::Broker, ::AbstractString) = throw(
    ArgumentError("$(typeof(broker)) must define cancel_order!"),
)
broker_mode(broker::Broker) = throw(
    ArgumentError("$(typeof(broker)) must define broker_mode"),
)

is_live(broker::Broker) = broker_mode(broker) === LIVE
