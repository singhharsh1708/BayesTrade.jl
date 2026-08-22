# Architecture

## The one-sentence version

Market data becomes leak-free features, features become per-model posteriors, posteriors are
fused into a single predictive distribution over the forward return, that distribution
becomes a structured decision, and a deterministic risk engine decides whether the decision
is allowed to reach a broker.

## Two inference paths

The trading loop must never block on expensive inference, so inference is split.

**Hot path (per bar, bounded time):** conjugate updates, recursive filtering, cached
posterior parameters. Closed form only.

**Cold path (scheduled, offline):** sampling or variational inference over history,
posterior diagnostics, calibration checks. Produces a versioned parameter bundle that the
hot path loads. A cold-path failure degrades the system to stale-but-valid parameters; it
never stalls trading.

## Component responsibilities

| Module | Owns | Must not |
| --- | --- | --- |
| `domain` | Immutable value types and the invariants that make an invalid one unrepresentable | Perform I/O, or encode policy such as accounting rules or thresholds |
| `config` | Resolved settings, risk limits, trading-mode guard | Be read at load time |
| `data` | Ingestion, storage, quality checks | Know about models |
| `features` | Point-in-time feature construction | See any bar after `t` |
| `models` | One posterior each, from features alone | Know about portfolio state or orders |
| `inference` | Online filters and offline fits | Decide anything |
| `fusion` | Combining model posteriors and their reliabilities | Apply risk limits |
| `decision` | Turning a posterior into an intent | Place orders |
| `risk` | Hard constraints and the kill switch | Consult a model, or an LLM |
| `execution` | Broker I/O and order lifecycle | Re-decide anything |
| `backtest` | Deterministic replay of the same code path as live | Use future data |

The single most important invariant: **`risk` can veto `decision`, and `decision` can never
override `risk`.** The risk engine is plain deterministic code with no learned parameters,
so its behaviour is auditable and reproducible from the limits alone.

## Distributions

Predictive families come from `Distributions.jl`. `Normal`, `TDist`, `LogNormal` and
`MixtureModel` already have correct `cdf`, `quantile`, `logpdf` and moments, including the
bisection inverse a mixture needs, and reimplementing them would be a large surface of
arithmetic nobody else has checked.

What this package adds is the part `Distributions.jl` has no opinion about: which model
produced a prediction, when it was valid, and how much of its spread is reducible.

`ProbabilisticResult` carries `epistemic_variance` alongside the distribution rather than
inside it. The split between reducible and irreducible uncertainty is a statement about
*why* the distribution is that wide, not a property of the distribution, and keeping it
outside leaves the `Distributions.jl` types plain.

## Where an LLM is allowed

Interpreting news text, summarising market conditions, phrasing explanations, and
investigating anomalies. Never in the path that sizes or authorises a trade.

## Reproducibility

Every prediction is stored with the `ModelVersion` that produced it and the feature vector
it consumed, so any historical trade can be re-derived exactly. Random seeds are explicit
inputs, never implicit global state.
