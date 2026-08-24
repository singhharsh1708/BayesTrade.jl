"""
What the account currently holds.

Plain state with no opinions in it. The decision engine reads it, the risk engine rules on it,
and neither is allowed to change it: positions move when a fill arrives and at no other time.
"""

"""
    Position

One open holding.

`quantity` is signed: negative is short. Everything else follows from that convention rather
than from a separate direction flag, which is one fewer thing that can contradict itself.
"""
struct Position
    symbol::String
    quantity::Float64
    average_price::Float64
    last_price::Float64
    opened_at::DateTime
    sector::String

    function Position(;
            symbol::AbstractString,
            quantity::Real,
            average_price::Real,
            last_price::Real,
            opened_at::DateTime,
            sector::AbstractString = "unknown",
        )
        isempty(symbol) && throw(ArgumentError("a position needs a symbol"))
        isfinite(quantity) || throw(ArgumentError("quantity must be finite"))
        iszero(quantity) && throw(ArgumentError("a position with no quantity is not a position"))
        average_price > 0 ||
            throw(ArgumentError(string("average_price must be positive, got ", average_price)))
        last_price > 0 ||
            throw(ArgumentError(string("last_price must be positive, got ", last_price)))
        return new(
            String(symbol), Float64(quantity), Float64(average_price),
            Float64(last_price), opened_at, String(sector),
        )
    end
end

"""
    exposure(position)

Signed market value.
"""
exposure(position::Position) = position.quantity * position.last_price

"""
    gross_exposure(position)

Market value regardless of direction, which is what a limit on concentration means.
"""
gross_exposure(position::Position) = abs(exposure(position))

is_long(position::Position) = position.quantity > 0
is_short(position::Position) = position.quantity < 0

"""
    unrealised_pnl(position)

Profit if it closed at the last price. Signed quantity makes this one expression for both
directions.
"""
unrealised_pnl(position::Position) =
    position.quantity * (position.last_price - position.average_price)

"""
    Portfolio

The account as of a moment.

`peak_equity` and `day_start_equity` are carried rather than derived because a drawdown limit
is a statement about history, and recomputing history from the current state is exactly how a
drawdown limit quietly stops binding.
"""
struct Portfolio
    equity::Float64
    cash::Float64
    positions::Dict{String, Position}
    peak_equity::Float64
    day_start_equity::Float64
    as_of::DateTime

    function Portfolio(;
            equity::Real,
            cash::Real,
            positions::Dict{String, Position} = Dict{String, Position}(),
            peak_equity::Union{Real, Nothing} = nothing,
            day_start_equity::Union{Real, Nothing} = nothing,
            as_of::DateTime,
        )
        equity > 0 || throw(ArgumentError(string("equity must be positive, got ", equity)))
        isfinite(cash) || throw(ArgumentError("cash must be finite"))
        peak = peak_equity === nothing ? Float64(equity) : Float64(peak_equity)
        start = day_start_equity === nothing ? Float64(equity) : Float64(day_start_equity)
        peak >= equity ||
            throw(ArgumentError(string("peak_equity ", peak, " is below equity ", equity)))
        start > 0 || throw(ArgumentError("day_start_equity must be positive"))
        for (symbol, position) in positions
            position.symbol == symbol || throw(
                ArgumentError(
                    string("position keyed ", symbol, " describes ", position.symbol),
                ),
            )
        end
        return new(Float64(equity), Float64(cash), positions, peak, start, as_of)
    end
end

n_positions(portfolio::Portfolio) = length(portfolio.positions)

"""
    position_weight(portfolio, symbol)

Gross exposure to one symbol as a fraction of equity.
"""
function position_weight(portfolio::Portfolio, symbol::AbstractString)
    haskey(portfolio.positions, symbol) || return 0.0
    return gross_exposure(portfolio.positions[symbol]) / portfolio.equity
end

"""
    portfolio_exposure(portfolio)

Total gross exposure as a fraction of equity. Long and short both consume it: a book that is
long one name and short another is exposed twice, not not at all.
"""
function portfolio_exposure(portfolio::Portfolio)
    total = 0.0
    for position in values(portfolio.positions)
        total += gross_exposure(position)
    end
    return total / portfolio.equity
end

"""
    sector_exposure(portfolio, sector)

Gross exposure to one sector as a fraction of equity.
"""
function sector_exposure(portfolio::Portfolio, sector::AbstractString)
    total = 0.0
    for position in values(portfolio.positions)
        position.sector == sector && (total += gross_exposure(position))
    end
    return total / portfolio.equity
end

"""
    drawdown(portfolio)

How far below its peak the account is, as a fraction.
"""
drawdown(portfolio::Portfolio) =
    max(0.0, (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity)

"""
    daily_loss(portfolio)

How far below where it started the day, as a fraction. Zero when up.
"""
daily_loss(portfolio::Portfolio) = max(
    0.0, (portfolio.day_start_equity - portfolio.equity) / portfolio.day_start_equity,
)

Base.show(io::IO, position::Position) = @printf(
    io, "<Position %s %+.4g @ %.2f mark %.2f pnl %+.2f>",
    position.symbol, position.quantity, position.average_price, position.last_price,
    unrealised_pnl(position)
)

Base.show(io::IO, portfolio::Portfolio) = @printf(
    io, "<Portfolio equity=%.2f positions=%d exposure=%.1f%% drawdown=%.1f%%>",
    portfolio.equity, n_positions(portfolio), 100 * portfolio_exposure(portfolio),
    100 * drawdown(portfolio)
)
