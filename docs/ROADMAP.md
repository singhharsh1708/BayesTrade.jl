# Roadmap

Phases land in order. Each phase is three to four pull requests. A phase is done when its
tests pass in CI and the capability is reachable from the CLI or the API.

| Phase | Capability | Status |
| --- | --- | --- |
| 1 | Package architecture, domain types, config, model interfaces | done |
| 2 | Historical data ingestion (synthetic + vendor adapter) | done |
| 3 | Leak-free feature generation | done |
| 4 | Bayesian price/return model | done |
| 5 | Bayesian volatility model | done |
| 6 | Market regime model | done |
| 7 | Bayesian fusion layer | done |
| 8 | Event-driven backtester and walk-forward | done |
| 9 | Deterministic risk engine | done |
| 10 | Paper broker | done |
| 11 | Live market feed and tick aggregation | done |
| 12 | Dashboard payload (JSON contract; no page in-tree) | done |
| 13 | Zerodha Kite: binary tick protocol and REST client | done (no live credentials) |
| 14 | Extended paper trading | pending |
| 15 | Small live capital, only if 14 justifies it | pending |

## Definition of done for the MVP

```julia
BayesTrade.backtest()
```

reports CAGR, Sharpe, Sortino, max drawdown, win rate, average P(profit), calibration error
and trade count, alongside the model version and the risk limits that were in force. And
`BayesTrade.paper()` runs the same decision path against live data through the paper broker.

## Non-goals

Beating a benchmark on a single backtest. Any result not accompanied by a calibration check
and a walk-forward is treated as unvalidated.

## History

Phases 1 to 6 were first built in Python and then ported here. The port kept every
statistical test and the decisions worth keeping, and dropped roughly six hundred lines of
hand-rolled distribution code that `Distributions.jl` already provides correctly.
