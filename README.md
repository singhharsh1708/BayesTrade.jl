# BayesTrade.jl

A modular, uncertainty-aware algorithmic trading system.

> We don't predict the future. We quantify it.

BayesTrade does not emit `BUY`. It emits a posterior — expected return, predictive
variance, a credible interval, the probability of a large loss, and the model uncertainty
behind all of it — and then a **deterministic** risk engine decides whether that posterior
justifies taking the trade at all. `NO_TRADE` because uncertainty is too high is a
first-class outcome, not a failure.

## Design principles

| Principle | What it means here |
| --- | --- |
| Prediction is separate from risk | Bayesian models estimate; a rule-based risk engine authorises. The risk engine always wins. |
| Uncertainty is never hidden | Every model output carries a variance and a credible interval, all the way to the dashboard. |
| Probabilities must be calibrated | Backtests report calibration error alongside Sharpe. An uncalibrated 0.76 is worthless. |
| Nothing expensive in the hot path | Live updates are conjugate, Kalman or recursive. Sampling runs offline and ships parameters. |
| Everything is versioned | Every prediction records the model version that produced it. |
| Paper by default | Live trading requires an explicit, deliberate configuration flag. |

## Why Julia

The system is mostly closed-form probability: conjugate updates, Kalman-style filtering, a
hidden Markov forward recursion, and mixtures. Julia expresses that directly, and
`Distributions.jl` supplies the predictive families with correct `cdf`, `quantile` and
`logpdf` already checked by people other than us.

The one place Julia costs something is the broker: Zerodha ships an official Python SDK and
no Julia one, so the Kite client here is written against the REST and WebSocket protocols
directly. That work is confined to the execution layer, behind a `Broker` interface that
paper trading and backtesting satisfy without it.

## Architecture

```
market data ──▶ feature engine ──▶ modular Bayesian models ──▶ Bayesian fusion
                                                                     │
                                                                     ▼
                                              decision engine ──▶ risk engine ──▶ execution
```

Six independently replaceable models (price/momentum, volatility, fundamentals,
news/sentiment, market regime, portfolio context) each produce a probabilistic result. A
hierarchical fusion layer weights them by learned reliability rather than averaging them,
and emits a single posterior over the forward return.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the component map,
[docs/MODELS.md](docs/MODELS.md) for the probabilistic models, and
[docs/ROADMAP.md](docs/ROADMAP.md) for build order.

## Getting started

Requires Julia 1.10 or later.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

## Status

Early. Track progress in [docs/ROADMAP.md](docs/ROADMAP.md). Nothing here is investment
advice, and nothing here should be pointed at live capital.

## Licence

MIT. See [LICENSE](LICENSE).
