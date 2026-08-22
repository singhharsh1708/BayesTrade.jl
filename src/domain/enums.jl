"""
Closed vocabularies shared across every layer.

Each is an `@enum` rather than a bare `Symbol`, so a typo is a compile-time or
constructor-time error rather than a value that silently matches nothing.
"""

"""
    TradingMode

How orders leave the system.

`BACKTEST` and `PAPER` are simulated. `LIVE` risks real capital and is never the default;
reaching it requires an explicit, deliberate configuration flag.
"""
@enum TradingMode BACKTEST PAPER LIVE

"""
    is_simulated(mode::TradingMode)

Whether orders in this mode are simulated. Only `LIVE` is not.
"""
is_simulated(mode::TradingMode) = mode !== LIVE

"""
    Action

What the decision engine wants to do.

`NO_TRADE` is distinct from `HOLD`: `HOLD` means the posterior was read and the current
exposure is right, while `NO_TRADE` means the posterior was too uncertain to act on at all.
Collapsing the two would hide the system's most important admission.
"""
@enum Action BUY SELL HOLD NO_TRADE

"""
    is_actionable(action::Action)

Whether this action can reach a broker. `HOLD` and `NO_TRADE` cannot.
"""
is_actionable(action::Action) = action === BUY || action === SELL

"""
    OrderSide

Direction of an order.
"""
@enum OrderSide BUY_SIDE SELL_SIDE

@enum OrderType MARKET LIMIT STOP_LOSS STOP_LOSS_MARKET

@enum OrderStatus PENDING OPEN PARTIALLY_FILLED FILLED CANCELLED REJECTED

"""
    is_terminal(status::OrderStatus)

Whether an order in this state will never change again.
"""
is_terminal(status::OrderStatus) =
    status === FILLED || status === CANCELLED || status === REJECTED

"""
    Regime

Latent market states the regime model assigns probabilities to.

Direction and volatility are separate axes, and the models treat them separately: a flat
five-state chain would conflate them, and since bear markets are usually volatile it would
spend its capacity relearning that correlation instead of the transitions.
"""
@enum Regime BULL BEAR SIDEWAYS HIGH_VOLATILITY LOW_VOLATILITY

@enum Sentiment POSITIVE NEUTRAL NEGATIVE

"""
    ModelName

The six modules the fusion layer knows how to weight.
"""
@enum ModelName MOMENTUM VOLATILITY FUNDAMENTAL SENTIMENT REGIME PORTFOLIO

@enum RiskCheckStatus PASS FAIL SKIPPED

@enum Exchange NSE BSE

"""
    NoTradeReason

Why the decision engine declined to act.

Declining is an outcome the system is supposed to reach often. Recording which condition
triggered it is what makes the refusal auditable rather than a shrug.
"""
@enum NoTradeReason begin
    EDGE_TOO_SMALL
    UNCERTAINTY_TOO_HIGH
    LOSS_PROBABILITY_TOO_HIGH
    MODEL_DISAGREEMENT
    INSUFFICIENT_DATA
    STALE_POSTERIOR
end

"""
    slug(value)

Lower-case string form, for configuration, logs and file formats.

Defined explicitly rather than relying on `string`, so renaming an enum member is a visible
break in the serialised form rather than a silent one.
"""
slug(value::Union{TradingMode, Action, Regime, Sentiment, ModelName, RiskCheckStatus, NoTradeReason, OrderType, OrderStatus}) =
    lowercase(string(value))

slug(value::OrderSide) = value === BUY_SIDE ? "buy" : "sell"
slug(value::Exchange) = string(value)
