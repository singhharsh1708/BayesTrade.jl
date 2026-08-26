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
using JSON3
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
include("domain/features.jl")
include("config/limits.jl")
include("config/settings.jl")
include("models/interface.jl")
include("data/calendar.jl")
include("data/processes.jl")
include("data/synthetic.jl")
include("data/store.jl")
include("data/csv_io.jl")
include("data/quality.jl")
include("data/sources.jl")
include("features/window.jl")
include("features/base.jl")
include("features/engine.jl")
include("features/price.jl")
include("features/momentum.jl")
include("features/volatility.jl")
include("features/volume.jl")
include("features/labels.jl")
include("features/presets.jl")
include("inference/online/linear.jl")
include("inference/online/variance.jl")
include("inference/online/regime.jl")
include("models/scaling.jl")
include("models/return_model.jl")
include("models/volatility_model.jl")
# After return_model.jl: require_later and HorizonMismatchError are defined there and are
# generic over ProbabilisticModel.
include("models/regime_model.jl")
include("fusion/pool.jl")
include("fusion/reliability.jl")
include("fusion/fuse.jl")
include("inference/posterior/calibration.jl")
include("inference/offline/walk_forward.jl")
include("models/persistence.jl")
include("models/report.jl")
# After calibration.jl: a replay reports the calibration of what it believed.
include("portfolio/state.jl")
include("decision/engine.jl")
include("risk/engine.jl")
include("feed/hours.jl")
include("feed/stream.jl")
include("execution/broker.jl")
include("execution/paper.jl")
# Before session/paper.jl: the session carries its last reconciliation as a field, so the
# type has to exist by then. The method that reconciles a session resolves at the call.
include("ops/reconciliation.jl")
include("ops/rebuild.jl")
# After execution/broker.jl: the Kite client speaks in Orders and Quotes.
include("kite/protocol.jl")
include("kite/session.jl")
# After data/sources.jl and feed/hours.jl: the Groww source is a BarSource and stamps its
# bars in exchange-local time. History only; it has no order path at all.
include("groww/session.jl")
include("groww/source.jl")
include("backtest/replay.jl")
include("report/dashboard.jl")
include("report/page.jl")
include("session/paper.jl")
# After session/paper.jl: health inspects a session, and the session calls health at run
# time, so the mutual reference resolves at the call rather than at definition.
include("ops/health.jl")
include("ops/recovery.jl")
include("ops/provenance.jl")
include("ops/model_health.jl")
include("ops/reconcile_session.jl")

export TradingMode, BACKTEST, PAPER, LIVE, is_simulated
export Action, BUY, SELL, HOLD, NO_TRADE, is_actionable
export OrderSide, BUY_SIDE, SELL_SIDE
export OrderType, MARKET, LIMIT, STOP_LOSS, STOP_LOSS_MARKET
export OrderStatus, PENDING, OPEN, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, is_terminal
export Regime, BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY
export Sentiment, POSITIVE, NEUTRAL, NEGATIVE
export ModelName, MOMENTUM, VOLATILITY, FUNDAMENTAL, SENTIMENT, REGIME, PORTFOLIO
export RiskCheckStatus, PASS, FAIL, SKIPPED
export Agreement, AGREEMENT_HIGH, AGREEMENT_MEDIUM, AGREEMENT_LOW
export disagreement_share, model_agreement, AGREEMENT_BOUNDS
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
export PACKAGE_VERSION

export BARS_PER_YEAR, ANNUALISER, NSE_CLOSE
export is_trading_day, trading_days, session_close, annualise, deannualise
export TradingCalendar, CalendarCoverageError, covers, is_full_session
export nse_calendar, load_calendar, NSE_HOLIDAYS_2025, NSE_HOLIDAYS_2026, NSE_MUHURAT
export NSE_SOURCES

export ReturnProcess, ProcessPath, simulate, process_parameters
export GaussianReturns, AR1Returns, innovation_scale
export RegimeSwitchingReturns, n_regimes, stationary_distribution
export StochasticVolatilityReturns, log_volatility_mean

export BarShape, SyntheticSeries, generate_series
export closes, true_log_returns, true_volatility, true_states
export realised_log_returns, bars_until

export BarStore, InMemoryBarStore, Coverage, IntervalConflictError
export upsert!, symbols, coverage, history, load_range, latest, bar_count
export clear!, all_bars, align

export BarFileError, write_bars, read_bars, save_store, load_store

export Severity, INFO, WARNING, ERROR
export QualityIssue, QualityReport, validate_bars, is_usable, summarise
export of_severity, errors, warnings, split_ratio

export BarSource, DataSourceError, SymbolNotFoundError, TransientSourceError
export source_name, supported_intervals, supports, fetch_bars
export SyntheticSource, seed_for, RetryingSource, delay_for
export IngestionReport, ingest!, total_written, succeeded, failed, is_complete

export FeatureVector, staleness, is_stale, feature_names, require, design_row, subset
export BarWindow, window_symbol, current, window_as_of, turnovers, tail
export Feature, FeatureSet, feature_name, columns, lookback, compute, required_bars
export evaluate
export empty_vector
export FeatureEngine, warmup_bars, features_at, walk, first_complete_at
export LogReturn, SimpleReturn
export MovingAverage, PriceToMovingAverage, MovingAverageSpread, Momentum
export RelativeStrengthIndex, PriceZScore, DrawdownFromHigh, TrendSlope, TrendQuality
export is_flat, regress
export RealisedVolatility, EwmaVolatility, ParkinsonVolatility, GarmanKlassVolatility
export DownsideVolatility, VolatilityRatio, AverageTrueRange
export RelativeVolume, VolumeZScore, MedianTurnover, AmihudIlliquidity
export Label, TrainingExample, forward_label, build_training_set, is_positive
export minimal_feature_set, default_feature_set

export NormalInverseGammaPrior, weakly_informative_prior, expected_noise_variance
export BayesianLinearModel, n_features, n_absorbed, effective_sample_size
export posterior_precision, posterior_shape, posterior_rate, coefficients
export noise_variance, residual_scale, coefficient_covariance, coefficient_std
export predict_mean, state, load_state!, solve_precision

export InverseGammaPrior, variance_prior, DiscountedVarianceFilter
export DEFAULT_DISCOUNTS, n_components, n_skipped, discount_grid, discount_weights
export expected_discount, discount_entropy, discount_disagreement, steady_state_shape
export evolved_shape, evolved_rate, expected_volatility, expected_log_volatility
export volatility_uncertainty, plugin_variance, variance_inflation
export observe_variance!, skip_observation!, return_predictive, variance_posterior
export predict_realised_variance, volatility_interval

export RegimePrior, RegimeParameters, RegimeFilter, REGIME_STATES, N_REGIMES
export regime_transition, regime_shape, estimate_regime_parameters, emission
export regime_probabilities, regime_belief, regime_confidence, most_likely_regime
export propagate, observe_return!, fit_filter!, horizon_weights, predict_return
export variance_decomposition, transition_matrix, n_states

export RegimeSource, BarReturnSource, MarketRegimeModel, MIN_REGIME_ROWS

export OpinionPool, disagreement
export ModelReliability, reliabilities, reliability_belief, mean_log_scores, score!
export n_models
export FusedPrediction, fuse, score_fusion!, epistemic_share
export ReplayRecord, ReplayConfig, ReplayReport, replay

export Position, Portfolio, exposure, gross_exposure, is_long, is_short, unrealised_pnl
export n_positions, position_weight, portfolio_exposure, sector_exposure, drawdown, daily_loss

export TradeIntent, decide, decline, downside, size_by_risk
export RiskCheck, RiskRuling, review, approved, failures, was_reduced, passed, failed

export Broker, Order, Fill, OrderReceipt, place_order!, cancel_order!, broker_mode, is_live
export is_buy, signed_quantity, cash_flow, was_filled
export PaperBroker, PaperCosts, fill_price, commission, tradeable_quantity, apply!
export mark_to_market!, equity, portfolio, order_from_ruling, next_order_id!

export MarketHours, IST_OFFSET, exchange_time, is_open, same_session, session_bounds
export TickSource, ReplayTickSource, next_tick!, source_symbols, exhausted
export FeedHealth, is_stale, silence, accept!, mark_stale!, arrived_after_silence
export BarAggregator, bucket_of, has_open_bar, push_tick!, build_bar, flush!
export FeedSession, handle_tick!, close_session!, run_feed!

export KiteSegment, NSE_CM, BSE_CM, NSE_FO, CDS, BSE_CDS, MCX_FO, OTHER_SEGMENT
export KiteTick, DepthEntry, KiteProtocolError, segment_of, price_divisor
export parse_packet, parse_frame, to_quote
export KiteCredentials, KiteSession, KiteRequest, KiteResponse, KiteError
export credentials_from_env, login_url, session_checksum, is_authenticated
export authorisation, build_request, kite_call, authenticate!, kite_quote, order_params

export dashboard_payload, write_dashboard, DASHBOARD_SCHEMA_VERSION
export dashboard_page, write_dashboard_page, serve_dashboard

export PaperTradingSession, SessionCounters, SESSION_SCHEMA_VERSION
export on_tick!, on_bar!, close_bar!, session_report, record!

export GrowwCredentials, GrowwSession, GrowwRequest, GrowwResponse, GrowwError
export GROWW_API_ROOT, GROWW_READ_PATHS, GROWW_INTERVALS, GROWW_MAX_WINDOW_DAYS
export groww_credentials_from_env, access_checksum, groww_call, token_expired, has_secret
export GrowwSource, MalformedBarError, vendor_symbol, window_chunks
export groww_transport, connect_groww

export HealthStatus, SystemHealth, check_health, may_trade, healthy, problems
export JournalState, read_journal, resume!, already_handled
export RebuiltAccount, rebuild_account, rebuild_report, REBUILD_EQUITY_TOLERANCE
export RunManifest, run_manifest, manifest_payload, run_label, MANIFEST_SCHEMA_VERSION
export ModelCheck, ModelHealth, assess_model_health, model_health_report
export ExternalPosition, ExternalOrder, VenueSnapshot, AccountSource, fetch_account
export LocalAccount, local_account, Discrepancy, Reconciliation, reconcile
export ReconciliationStatus, RECONCILE_MATCHED, RECONCILE_MISMATCHED, RECONCILE_UNAVAILABLE
export ReconciliationTolerances, reconciled, may_open_new_positions
export reconciliation_record, reconciliation_report
export reconcile!
export model_problems, trustworthy, MODEL_HEALTH_BOUNDS

export FeatureScaler, fit_scaler, transform, transform_row, unscale_coefficients
export BayesianReturnModel, HorizonMismatchError, design_columns, response_scale
export can_predict, coefficient_report, restore!
export ResponseScalePolicy, ConstantScale, VolatilityScale
export policy_columns, policy_parameters, default_residual_scale

export VarianceSource, SquaredReturnSource, BayesianVolatilityModel
export source_columns, source_name, observe, absorb!, feature_names, predictive_df

export CoveragePoint, ReliabilityBin, CalibrationReport, coverage_error, reliability_gap
export interval_calibration_error, is_overconfident, assess
export probability_integral_transform, kolmogorov_smirnov_uniform
export coverage_curve, reliability_curve, expected_calibration_error, brier_score

export WalkForwardConfig, PredictionRecord, walk_forward, outcomes, predictives
export went_up, prediction_error

export ModelFileError, save_model, load_model
export FitReport, fit_return_model, summarise

end # module
