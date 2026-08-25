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

The one place Julia costs something is the broker: neither Zerodha nor Groww ships a Julia
SDK, so both clients here are written against the documented HTTP protocols directly. That
work is confined to the edges, behind a `Broker` interface that paper trading and backtesting
satisfy without it and a `BarSource` interface that any vendor can be adapted to.

Historical data comes from `GrowwSource`, which reads NSE candles and nothing else: its client
can build exactly two request paths, and neither is an order endpoint. See
`docs/RUNBOOK.md` for credentials and the correctness rules it enforces on vendor data.

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

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Running it

Two entry points, both on generated data. Neither needs credentials, a network or a broker.

```sh
# replay the pipeline over history and print what it believed and decided
julia --project=. examples/backtest.jl          # both markets
julia --project=. examples/backtest.jl edge     # just the one with an edge

# drive the long-running session a tick at a time, as a live feed would
julia --project=. examples/paper_session.jl
julia --project=. examples/paper_session.jl 1400 journal.jsonl
```

`backtest.jl` runs two markets deliberately: one whose returns carry genuine
autocorrelation, and one with none. The system trades the first and declines
every bar of the second. Showing only the second makes it look inert; showing
only the first makes it look like a backtest.

`paper_session.jl` is the same code path a live session would run. The only
difference is where the ticks come from, which is the point: nothing about the
decision, risk or execution path changes when the feed becomes real.

With a journal path it appends one JSON line per decision as it happens, so a
session that is killed keeps everything up to the last line.

## Going live

Not yet, and not by accident. Two independent environment variables have to
agree before anything can reach a venue:

```sh
export BAYESTRADE_TRADING_MODE=live
export BAYESTRADE_ALLOW_LIVE_TRADING=true
export KITE_API_KEY=...
export KITE_API_SECRET=...
```

One of them set alone does nothing. Beyond that, `src/kite/` has never spoken to
Zerodha: the tick protocol is tested against frames laid out byte by byte and
the REST client against a stub transport, so the first real call will be the
first real test.

## Status

Early. Track progress in [docs/ROADMAP.md](docs/ROADMAP.md). Nothing here is investment
advice, and nothing here should be pointed at live capital.

## Licence

MIT. See [LICENSE](LICENSE).
