"""
Turning a posterior into an intent.

The decision engine reads a fused predictive and the limits, and says what it would like to do
and how much of the account it would like to use. It does not place orders and it does not get
the last word: everything it produces is a proposal that the risk engine rules on.

Declining is a first-class outcome and is expected to be the common one. Every refusal names
the condition that caused it, so a quiet week is auditable rather than a shrug.
"""

"""
    TradeIntent

What the decision engine would like to do, and why.

`target_weight` is a fraction of equity and is always non-negative; the direction lives in
`action`. Keeping them apart means a sizing bug cannot silently become a direction bug.
"""
struct TradeIntent
    symbol::String
    as_of::DateTime
    horizon_bars::Int
    action::Action
    target_weight::Float64
    reason::Union{NoTradeReason, Nothing}
    evidence::Dict{Symbol, Float64}

    function TradeIntent(;
            symbol::AbstractString,
            as_of::DateTime,
            horizon_bars::Integer,
            action::Action,
            target_weight::Real = 0.0,
            reason::Union{NoTradeReason, Nothing} = nothing,
            evidence::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
        )
        isempty(symbol) && throw(ArgumentError("an intent needs a symbol"))
        horizon_bars >= 1 ||
            throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
        weight = Float64(target_weight)
        (isfinite(weight) && weight >= 0) ||
            throw(ArgumentError(string("target_weight must be finite and non-negative")))
        if is_actionable(action)
            weight > 0 ||
                throw(ArgumentError(string(slug(action), " with no size is not an intent")))
            reason === nothing ||
                throw(ArgumentError("an actionable intent cannot also carry a refusal"))
        else
            iszero(weight) ||
                throw(ArgumentError(string(slug(action), " cannot carry a size")))
        end
        return new(
            String(symbol), as_of, Int(horizon_bars), action, weight, reason, evidence,
        )
    end
end

is_actionable(intent::TradeIntent) = is_actionable(intent.action)

"""
    decline(prediction, reason; evidence)

A refusal that records what it was looking at when it refused.
"""
decline(
    prediction::FusedPrediction, reason::NoTradeReason;
    evidence::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
) = TradeIntent(
    symbol = prediction.symbol, as_of = prediction.as_of,
    horizon_bars = prediction.horizon_bars, action = NO_TRADE,
    reason = reason, evidence = evidence,
)

"""
    downside(prediction, level)

The magnitude of the adverse move at a given tail probability, in return units.

Read off the predictive itself rather than assumed from a volatility, so a skewed or
heavy-tailed posterior sizes smaller without anything special being written for that case.
"""
function downside(prediction::FusedPrediction, level::Float64)
    lower = quantile(prediction.distribution, level)
    upper = quantile(prediction.distribution, 1 - level)
    return max(abs(lower), abs(upper), 1.0e-6)
end

"""
    size_by_risk(prediction, limits; tail = 0.05)

How much of the account to put at risk, from the budget and the posterior's own tail.

`risk_budget_per_trade` is what may be lost if the tail move happens, so the weight is the
budget divided by that move. A wider posterior therefore sizes smaller with no separate rule
for it, which is the whole reason the models emit distributions rather than point forecasts.
"""
function size_by_risk(
        prediction::FusedPrediction, limits::RiskLimits; tail::Float64 = 0.05,
    )
    adverse = downside(prediction, tail)
    return min(limits.risk_budget_per_trade / adverse, limits.max_position_weight)
end

"""
    decide(prediction, limits; now, max_age, tail)

Rule on one fused prediction.

The gates are read from the limits rather than written here, so the threshold a trade was
judged against is recorded alongside the limits in force at the time rather than buried in
whatever code happened to be deployed.

The order of the gates is deliberate. The direction-independent ones come first, so the reason
recorded is the most fundamental one; the tail gate comes after the direction is known,
because which tail is the dangerous one depends on which way the trade would go.
"""
function decide(
        prediction::FusedPrediction, limits::RiskLimits;
        now::Union{DateTime, Nothing} = nothing,
        max_age::Period = Day(3),
        tail::Float64 = 0.05,
    )
    0 < tail < 0.5 || throw(ArgumentError(string("tail must lie in (0, 0.5), got ", tail)))

    probability_up = probability_positive(prediction)
    share = epistemic_share(prediction)
    confidence = 1 - share
    evidence = Dict{Symbol, Float64}(
        :probability_positive => probability_up,
        :epistemic_share => share,
        :confidence => confidence,
        :expected_return => mean(prediction),
        :predictive_sd => std(prediction),
        :n_models => Float64(n_models(prediction)),
    )

    if now !== nothing && now - prediction.as_of > max_age
        return decline(prediction, STALE_POSTERIOR; evidence = evidence)
    end
    if share > limits.max_model_uncertainty
        return decline(prediction, UNCERTAINTY_TOO_HIGH; evidence = evidence)
    end
    if confidence < limits.min_confidence
        return decline(prediction, MODEL_DISAGREEMENT; evidence = evidence)
    end
    action = if probability_up >= limits.min_probability_positive
        BUY
    elseif probability_up <= 1 - limits.min_probability_positive
        SELL
    else
        NO_TRADE
    end
    action === NO_TRADE &&
        return decline(prediction, EDGE_TOO_SMALL; evidence = evidence)

    # Measured in the direction of the trade being considered. A large loss on a long is a
    # large move down; on a short it is a large move up. A gate that only ever looked
    # downward would wave through exactly the short whose upside tail should stop it, and
    # would refuse bearish longs for a risk they do not carry.
    loss_probability = action === BUY ?
        probability_below(prediction.distribution, -limits.large_loss_threshold) :
        probability_above(prediction.distribution, limits.large_loss_threshold)
    evidence[:probability_large_loss] = loss_probability
    if loss_probability > limits.max_probability_large_loss
        return decline(prediction, LOSS_PROBABILITY_TOO_HIGH; evidence = evidence)
    end

    weight = size_by_risk(prediction, limits; tail = tail)
    evidence[:target_weight] = weight
    evidence[:downside] = downside(prediction, tail)
    return TradeIntent(
        symbol = prediction.symbol, as_of = prediction.as_of,
        horizon_bars = prediction.horizon_bars, action = action,
        target_weight = weight, evidence = evidence,
    )
end

Base.show(io::IO, intent::TradeIntent) = @printf(
    io, "<TradeIntent %s %s %s%.2f%%>",
    intent.symbol, slug(intent.action),
    intent.reason === nothing ? "" : string(slug(intent.reason), " "),
    100 * intent.target_weight
)
