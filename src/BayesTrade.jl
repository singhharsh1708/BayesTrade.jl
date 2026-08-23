"""
    BayesTrade

A modular, uncertainty-aware Bayesian trading system.

The system does not emit `BUY`. It emits a posterior over the forward return, and a
deterministic risk engine decides whether that posterior justifies taking the trade at all.
Declining because the uncertainty is too high is a first-class outcome.

See `docs/ARCHITECTURE.md` for the component map and `docs/ROADMAP.md` for build order.
"""
module BayesTrade

using Dates
using Distributions
using LinearAlgebra
using Printf
using Random
using SHA
using SpecialFunctions
using Statistics
using StatsBase

include("domain/enums.jl")
include("domain/market.jl")
include("domain/probabilistic.jl")
include("config/limits.jl")
include("config/settings.jl")
include("models/interface.jl")

export TradingMode, BACKTEST, PAPER, LIVE, is_simulated
export Action, BUY, SELL, HOLD, NO_TRADE, is_actionable
export OrderSide, BUY_SIDE, SELL_SIDE
export OrderType, MARKET, LIMIT, STOP_LOSS, STOP_LOSS_MARKET
export OrderStatus, PENDING, OPEN, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, is_terminal
export Regime, BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY
export Sentiment, POSITIVE, NEUTRAL, NEGATIVE
export ModelName, MOMENTUM, VOLATILITY, FUNDAMENTAL, SENTIMENT, REGIME, PORTFOLIO
export RiskCheckStatus, PASS, FAIL, SKIPPED
export Exchange, NSE, BSE
export NoTradeReason
export EDGE_TOO_SMALL, UNCERTAINTY_TOO_HIGH, LOSS_PROBABILITY_TOO_HIGH
export MODEL_DISAGREEMENT, INSUFFICIENT_DATA, STALE_POSTERIOR
export slug

export Instrument, key, round_to_tick
export Bar, typical_price, turnover, true_range, log_return
export Quote, mid, spread, spread_bps
export FundamentalSnapshot, NewsItem, is_known_at

export ModelVersion, identifier
export student_t, CredibleInterval, credible_interval, width
export probability_above, probability_below, probability_positive, probability_loss_exceeds
export LabelledCategorical, probability_of, most_likely, normalised_entropy
export uniform_categorical, normalise
export ProbabilisticResult, epistemic_share

export RiskLimits, CONSERVATIVE, max_concurrent_position_weight, is_position_cap_binding
export Secret, reveal
export BrokerSettings, has_credentials, has_session
export DataSettings, ExecutionSettings, total_cost_bps, round_trip_cost
export Settings, load_settings, describe, doctor, is_live

export ProbabilisticModel, NotFittedError, FitState
export fit_state, model_name, model_semver, model_version, parameters, uncertainty
export fit!, update!, predict, is_fitted, n_observations
export mark_fitted!, reset!, require_fitted, params_hash, stable_hash

end # module
