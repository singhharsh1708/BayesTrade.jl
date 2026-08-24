"""
Deterministic risk limits.

Plain numbers with no learned parameters. That is the point: the risk engine's behaviour
must be reproducible from this object alone, so an audit of a past trade can re-derive the
ruling without re-running a model.

Limits are fractions of equity rather than currency amounts, so the same configuration
behaves identically on a one lakh account and a one crore account.

The cross-checks in the constructor matter more than the individual bounds. Each rejected
combination below is a plausible typo that would otherwise be discovered by a trade.
"""

"""
    RiskLimits

Hard constraints the risk engine enforces, and the gates the decision engine reads.

The probability gates live here rather than in the decision engine so that the threshold a
trade was judged against is recorded with the limits that were in force, not buried in
whatever code happened to be deployed.
"""
Base.@kwdef struct RiskLimits
    version::VersionNumber = v"0.1.0"

    max_position_weight::Float64 = 0.05
    max_sector_exposure::Float64 = 0.25
    max_portfolio_exposure::Float64 = 0.6
    max_open_positions::Int = 10

    max_daily_loss::Float64 = 0.02
    max_drawdown::Float64 = 0.15
    risk_budget_per_trade::Float64 = 0.005
    stop_loss_atr_multiple::Float64 = 2.5

    min_probability_positive::Float64 = 0.6
    max_probability_large_loss::Float64 = 0.15
    large_loss_threshold::Float64 = 0.05
    min_confidence::Float64 = 0.55
    max_model_uncertainty::Float64 = 0.6

    min_daily_turnover::Float64 = 5.0e7
    max_participation_rate::Float64 = 0.02
    max_annualised_volatility::Float64 = 0.6

    function RiskLimits(
            version, max_position_weight, max_sector_exposure, max_portfolio_exposure,
            max_open_positions, max_daily_loss, max_drawdown, risk_budget_per_trade,
            stop_loss_atr_multiple, min_probability_positive, max_probability_large_loss,
            large_loss_threshold, min_confidence, max_model_uncertainty,
            min_daily_turnover, max_participation_rate, max_annualised_volatility,
        )
        fractions = (
            :max_position_weight => max_position_weight,
            :max_sector_exposure => max_sector_exposure,
            :max_portfolio_exposure => max_portfolio_exposure,
            :max_daily_loss => max_daily_loss,
            :max_drawdown => max_drawdown,
            :risk_budget_per_trade => risk_budget_per_trade,
            :max_probability_large_loss => max_probability_large_loss,
            :large_loss_threshold => large_loss_threshold,
            :max_model_uncertainty => max_model_uncertainty,
            :max_participation_rate => max_participation_rate,
        )
        for (name, value) in fractions
            0 < value <= 1 ||
                throw(ArgumentError("$name must lie in (0, 1], got $value"))
        end
        max_open_positions > 0 ||
            throw(ArgumentError("max_open_positions must be positive, got $max_open_positions"))
        stop_loss_atr_multiple > 0 ||
            throw(ArgumentError("stop_loss_atr_multiple must be positive"))
        min_daily_turnover >= 0 ||
            throw(ArgumentError("min_daily_turnover cannot be negative"))
        max_annualised_volatility > 0 ||
            throw(ArgumentError("max_annualised_volatility must be positive"))
        0 <= min_confidence <= 1 ||
            throw(ArgumentError("min_confidence must lie in [0, 1], got $min_confidence"))

        # A gate at or below a coin flip is not a gate.
        # Strictly above a half. At exactly a half every prediction is either a buy or a
        # sell, the no-trade band vanishes, and a coin becomes a trading signal.
        0.5 < min_probability_positive < 1 || throw(
            ArgumentError(
                string(
                    "min_probability_positive must lie in (0.5, 1), got ",
                    min_probability_positive,
                ),
            ),
        )

        max_position_weight <= max_sector_exposure || throw(
            ArgumentError(
                string(
                    "max_position_weight ($max_position_weight) exceeds ",
                    "max_sector_exposure ($max_sector_exposure), so a single position ",
                    "could never fill its own allowance",
                ),
            ),
        )
        max_sector_exposure <= max_portfolio_exposure || throw(
            ArgumentError(
                string(
                    "max_sector_exposure ($max_sector_exposure) exceeds ",
                    "max_portfolio_exposure ($max_portfolio_exposure)",
                ),
            ),
        )
        max_daily_loss <= max_drawdown || throw(
            ArgumentError(
                string(
                    "max_daily_loss ($max_daily_loss) exceeds max_drawdown ",
                    "($max_drawdown), so the drawdown halt could never fire first",
                ),
            ),
        )
        risk_budget_per_trade <= max_daily_loss || throw(
            ArgumentError(
                string(
                    "risk_budget_per_trade ($risk_budget_per_trade) exceeds ",
                    "max_daily_loss ($max_daily_loss), so one losing trade could breach ",
                    "the daily limit on its own",
                ),
            ),
        )

        return new(
            version, max_position_weight, max_sector_exposure, max_portfolio_exposure,
            max_open_positions, max_daily_loss, max_drawdown, risk_budget_per_trade,
            stop_loss_atr_multiple, min_probability_positive, max_probability_large_loss,
            large_loss_threshold, min_confidence, max_model_uncertainty,
            min_daily_turnover, max_participation_rate, max_annualised_volatility,
        )
    end
end

"""
    max_concurrent_position_weight(limits)

Portfolio exposure implied if every open slot were filled to the per-position cap.
"""
max_concurrent_position_weight(limits::RiskLimits) =
    min(1.0, limits.max_position_weight * limits.max_open_positions)

"""
    is_position_cap_binding(limits)

Whether the per-position cap binds before the portfolio cap does.

If it does not, the open-position count is the constraint that actually shapes the book,
which is worth knowing before blaming the position cap for the shape.
"""
is_position_cap_binding(limits::RiskLimits) =
    max_concurrent_position_weight(limits) >= limits.max_portfolio_exposure

"""
    CONSERVATIVE

Tighter limits, intended as the starting point for anything touching real capital.
"""
const CONSERVATIVE = RiskLimits(
    max_position_weight = 0.03,
    max_sector_exposure = 0.15,
    max_portfolio_exposure = 0.4,
    max_open_positions = 8,
    max_daily_loss = 0.015,
    max_drawdown = 0.1,
    risk_budget_per_trade = 0.003,
    min_probability_positive = 0.7,
    max_probability_large_loss = 0.1,
    min_confidence = 0.65,
    max_model_uncertainty = 0.5,
)
