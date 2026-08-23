"""
Market observations.

Every type here carries the moment the information became *knowable*, which is not always
the moment it describes. Fundamentals are the clearest case: a quarter ending 31 March is
not public until the results are filed weeks later, and a backtest that uses it on 1 April
is not a backtest.

Validation lives in inner constructors, so a malformed observation cannot exist. That is
worth more than it costs: an invalid bar caught at construction names its own problem,
while one caught downstream shows up as a plausible, wrong feature.
"""

"""
    Instrument

A tradable symbol and the static facts needed to trade it.
"""
struct Instrument
    symbol::String
    exchange::Exchange
    name::Union{String, Nothing}
    sector::Union{String, Nothing}
    lot_size::Int
    tick_size::Float64

    function Instrument(
            symbol::AbstractString;
            exchange::Exchange = NSE,
            name = nothing,
            sector = nothing,
            lot_size::Integer = 1,
            tick_size::Real = 0.05,
        )
        isempty(symbol) && throw(ArgumentError("instrument needs a symbol"))
        lot_size > 0 || throw(ArgumentError("lot_size must be positive, got $lot_size"))
        tick_size > 0 || throw(ArgumentError("tick_size must be positive, got $tick_size"))
        return new(String(symbol), exchange, name, sector, Int(lot_size), Float64(tick_size))
    end
end

"""
    key(instrument)

Exchange-qualified identifier, unique where a symbol alone is not.
"""
key(instrument::Instrument) = string(slug(instrument.exchange), ':', instrument.symbol)

"""
    round_to_tick(instrument, price)

Snap a price to the exchange's tick grid, as the exchange itself would.
"""
round_to_tick(instrument::Instrument, price::Real) =
    round(round(price / instrument.tick_size) * instrument.tick_size, digits = 4)

"""
    Bar

One OHLCV candle, closed and immutable.

`timestamp` is the *closing* time of the interval. A bar is only usable by a feature once
its close has passed, which is what makes the no-look-ahead rule checkable rather than
aspirational.
"""
struct Bar
    symbol::String
    timestamp::DateTime
    open::Float64
    high::Float64
    low::Float64
    close::Float64
    volume::Float64
    interval::String

    function Bar(
            symbol::AbstractString,
            timestamp::DateTime,
            open::Real,
            high::Real,
            low::Real,
            close::Real,
            volume::Real;
            interval::AbstractString = "1d",
        )
        isempty(symbol) && throw(ArgumentError("bar needs a symbol"))
        all(>(0), (open, high, low, close)) ||
            throw(ArgumentError("$symbol: every price must be positive"))
        volume >= 0 || throw(ArgumentError("$symbol: volume cannot be negative"))
        high >= low ||
            throw(ArgumentError("$symbol: high ($high) is below low ($low)"))
        low <= open <= high ||
            throw(ArgumentError("$symbol: open ($open) lies outside [$low, $high]"))
        low <= close <= high ||
            throw(ArgumentError("$symbol: close ($close) lies outside [$low, $high]"))
        return new(
            String(symbol), timestamp, Float64(open), Float64(high),
            Float64(low), Float64(close), Float64(volume), String(interval),
        )
    end
end

"""
    typical_price(bar)

The high-low-close average, the usual proxy for where the bar actually traded.
"""
typical_price(bar::Bar) = (bar.high + bar.low + bar.close) / 3

"""
    turnover(bar)

Traded value, which is the quantity liquidity checks actually care about. A large share
count means nothing without a price.
"""
turnover(bar::Bar) = typical_price(bar) * bar.volume

"""
    true_range(bar)

The bar's own high-low range, before any gap from the previous close.
"""
true_range(bar::Bar) = bar.high - bar.low

"""
    log_return(previous, current)

Log return between two bars' closes. Log rather than simple, so returns compose additively:
chaining +50% and -50% is not zero.
"""
log_return(previous::Bar, current::Bar) = log(current.close / previous.close)

"""
    Quote

A live top-of-book snapshot.
"""
struct Quote
    symbol::String
    timestamp::DateTime
    last_price::Float64
    bid::Union{Float64, Nothing}
    ask::Union{Float64, Nothing}
    bid_quantity::Int
    ask_quantity::Int
    volume::Float64

    function Quote(
            symbol::AbstractString,
            timestamp::DateTime,
            last_price::Real;
            bid = nothing,
            ask = nothing,
            bid_quantity::Integer = 0,
            ask_quantity::Integer = 0,
            volume::Real = 0.0,
        )
        isempty(symbol) && throw(ArgumentError("quote needs a symbol"))
        last_price > 0 || throw(ArgumentError("$symbol: last_price must be positive"))
        if bid !== nothing && ask !== nothing && bid > ask
            throw(ArgumentError("$symbol: crossed book, bid $bid above ask $ask"))
        end
        return new(
            String(symbol), timestamp, Float64(last_price),
            bid === nothing ? nothing : Float64(bid),
            ask === nothing ? nothing : Float64(ask),
            Int(bid_quantity), Int(ask_quantity), Float64(volume),
        )
    end
end

"""
    mid(quote)

Midpoint of the book, or `nothing` when either side is missing.
"""
function mid(q::Quote)
    (q.bid === nothing || q.ask === nothing) && return nothing
    return (q.bid + q.ask) / 2
end

"""
    spread(quote)

Absolute bid-ask spread, or `nothing` without a two-sided book.
"""
function spread(q::Quote)
    (q.bid === nothing || q.ask === nothing) && return nothing
    return q.ask - q.bid
end

"""
    spread_bps(quote)

Spread in basis points of the mid: the honest floor on execution cost.
"""
function spread_bps(q::Quote)
    s = spread(q)
    m = mid(q)
    (s === nothing || m === nothing || m == 0) && return nothing
    return 1.0e4 * s / m
end

"""
    FundamentalSnapshot

Reported financials for one period, tagged with when they became public.

`period_end` is what the numbers describe. `reported_at` is when the market could have
known them. Features must filter on `reported_at`, and a snapshot reported before its own
period cannot be constructed at all.
"""
Base.@kwdef struct FundamentalSnapshot
    symbol::String
    period_end::Date
    reported_at::DateTime
    revenue::Union{Float64, Nothing} = nothing
    revenue_growth_yoy::Union{Float64, Nothing} = nothing
    earnings::Union{Float64, Nothing} = nothing
    earnings_growth_yoy::Union{Float64, Nothing} = nothing
    operating_margin::Union{Float64, Nothing} = nothing
    net_margin::Union{Float64, Nothing} = nothing
    roe::Union{Float64, Nothing} = nothing
    roce::Union{Float64, Nothing} = nothing
    debt_to_equity::Union{Float64, Nothing} = nothing
    free_cash_flow::Union{Float64, Nothing} = nothing
    price_to_earnings::Union{Float64, Nothing} = nothing
    price_to_book::Union{Float64, Nothing} = nothing

    function FundamentalSnapshot(
            symbol, period_end, reported_at, revenue, revenue_growth_yoy, earnings,
            earnings_growth_yoy, operating_margin, net_margin, roe, roce,
            debt_to_equity, free_cash_flow, price_to_earnings, price_to_book,
        )
        isempty(symbol) && throw(ArgumentError("snapshot needs a symbol"))
        if Date(reported_at) < period_end
            throw(
                ArgumentError(
                    string(
                        "$symbol: reported_at $(Date(reported_at)) precedes ",
                        "period_end $period_end, which is not physically possible",
                    ),
                ),
            )
        end
        if debt_to_equity !== nothing && debt_to_equity < 0
            throw(ArgumentError("$symbol: debt_to_equity cannot be negative"))
        end
        return new(
            String(symbol), period_end, reported_at, revenue, revenue_growth_yoy,
            earnings, earnings_growth_yoy, operating_margin, net_margin, roe, roce,
            debt_to_equity, free_cash_flow, price_to_earnings, price_to_book,
        )
    end
end

"""
    NewsItem

A headline with an optional pre-computed sentiment label.

Scoring happens outside the trading loop; this type carries the result of that offline work
rather than triggering it.
"""
Base.@kwdef struct NewsItem
    symbol::String
    published_at::DateTime
    headline::String
    source::Union{String, Nothing} = nothing
    url::Union{String, Nothing} = nothing
    sentiment::Union{Sentiment, Nothing} = nothing
    sentiment_confidence::Union{Float64, Nothing} = nothing
    scored_at::Union{DateTime, Nothing} = nothing

    function NewsItem(
            symbol, published_at, headline, source, url,
            sentiment, sentiment_confidence, scored_at,
        )
        isempty(symbol) && throw(ArgumentError("news item needs a symbol"))
        isempty(headline) && throw(ArgumentError("$symbol: news item needs a headline"))
        if sentiment_confidence !== nothing
            sentiment === nothing && throw(
                ArgumentError("$symbol: sentiment_confidence given without a label"),
            )
            0 <= sentiment_confidence <= 1 ||
                throw(ArgumentError("$symbol: sentiment_confidence must lie in [0, 1]"))
        end
        return new(
            String(symbol), published_at, String(headline), source, url,
            sentiment, sentiment_confidence, scored_at,
        )
    end
end

"""
    is_known_at(observation, moment)

Whether this fact could have been known at `moment`.

Defined on the types whose knowability differs from what they describe. A backtest that
does not call this is not a backtest.
"""
is_known_at(snapshot::FundamentalSnapshot, moment::DateTime) = snapshot.reported_at <= moment
is_known_at(item::NewsItem, moment::DateTime) = item.published_at <= moment
is_known_at(bar::Bar, moment::DateTime) = bar.timestamp <= moment
