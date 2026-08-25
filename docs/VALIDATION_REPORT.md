# BayesTrade system validation report

Sections 26, 27 and 28 of the validation brief. Everything below is measured; where it is not, it
says so.

Reproduce:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=validation validation/baseline.jl
julia --project=validation validation/calibration.jl
julia --project=validation validation/benchmarks.jl
julia --project=validation validation/mutations.jl
```

Companion documents: `VALIDATION_BASELINE.md` (sections 1 to 8),
`VALIDATION_CALIBRATION.md` (sections 9 to 19), `PRE_LIVE_AUDIT.md` (the earlier audit).

## 1. Architecture

Unchanged by this exercise, which was the brief's constraint. No model added, no model replaced,
no strategy introduced. Two diagnostics were added (`disagreement_share` / `model_agreement`, and
`assess_model_health`) and neither gates a trade.

## 2 to 3. Data and features

The pipeline was traced boundary by boundary. One defect found and fixed.

**Bar accepted infinite prices.** `all(>(0), prices)` is true for `Inf`, so an infinite price
satisfied positivity, then satisfied `high >= low` and `low <= open <= high`, and constructed
without complaint. It would have propagated into features and posteriors without ever raising.
The identical defect was found and fixed for `Quote` in the earlier audit and missed here, so an
infinite price could not enter through the live feed but could enter through the store.

## 4. Bayesian return model

Checked against its closed forms rather than for absence of crashes. Posterior matches the
textbook normal-inverse-gamma posterior to a relative 1e-10. Sequential absorption equals batch,
in order or shuffled. Epistemic uncertainty falls 50x between 10 and 2,000 observations while the
total settles on the noise floor. Degrees of freedom track `2a_0 + n`. A coefficient is recovered
to 0.005 by 5,000 rows. Forgetting bounds the effective sample size at `1/(1-lambda)`.

**V7**: `b_n` carries the prior's ridge penalty `beta' L_0 beta`, so where a coefficient is large
against `coefficient_scale` the model reports signal as noise — 4.13x the true noise variance at
the default scale in the worst case tried. It does not bite in this system because features are
standardised with statistics frozen at fitting time; `E[sigma^2]` is 0.9996 of the empirical
residual variance on the baseline configuration. **The standardiser is therefore load-bearing**,
and that is now asserted: raw, the fitted coefficient comes back at 10.4% of what the data says;
standardised, 99.8%.

## 5 to 6. Volatility and regime

Volatility distinguishes quiet from violent by 4x, absorbs a shock with most of the move inside
twenty bars, gives it back on collapse, and shifts its grid toward faster forgetting when the
level keeps moving. A flat bar is an ordinary observation. A skipped bar ages the filter and is
measurably different from substituting a zero return.

The regime chain identifies states by what they look like rather than by index, switches within a
hundred bars, and holds a valid simplex over four thousand bars.

## 7 to 8. Fusion and disagreement

The pool widens when models disagree rather than averaging the conflict away. Weights survive two
thousand rounds of pressure without reaching zero — a model on zero weight can never earn its way
back. With nothing weightable at all it falls back to uniform, not to whichever model is first.

`disagreement_share` and `model_agreement` are exposed in the decision journal and the dashboard.
**Nothing branches on them.** Agreement measures location, not confidence: a confident model and
a useless one that agree on the mean report AGREEMENT_HIGH, correctly.

## 9 to 11. Calibration

**The direction of miscalibration is set by the generating process, not by the configuration.**
Forty-five runs across five generators, three feature sets and three model sets: constant-
volatility processes calibrate to within half a percent; processes whose volatility moves on its
own come out underconfident. Nothing in the sweep is overconfident. This supersedes finding V1.

The volatility model halves the error on the stochastic-volatility generator (+0.1247 to +0.0522),
which is the clearest evidence in this exercise that a component earns its place.

By regime, on ex-ante splits only: **overconfident in quiet markets** by about a point of coverage,
calibrated in violent ones. Aggregate calibration hides it.

## 12 to 13. Risk and execution

The baseline ran 1,478 decisions through the risk engine and **no gate failed once** — evidence
it was never reached, not evidence it works. 72 adversarial assertions now reach every gate.

**V4 (HIGH, fixed)**: `session_report` reported the drawdown at the final bar under the name
`drawdown_pct`, beside `return_pct`. `examples/paper_session.jl` printed it as "peak drawdown".
On the baseline run the true maximum was **0.944% against the 0.412% being reported**.

**V6 (fixed)**: the dashboard ruled with three fewer gates than the live session, so its decisions
panel was quietly optimistic.

## 14 to 17. Backtester and baselines

One harness, same bars, same costs, no tuning of BayesTrade.

**V9 (HIGH, characterisation)**: on three generators out of four, BayesTrade took **zero
positions across 3,000 bars**. On the gaussian generator that is correct — no edge exists and
`edge_too_small` was the reason on every bar. On regime switching and stochastic volatility it is
a limitation: both contain exploitable structure. The edge gate compares the predicted move
against the predictive spread, so where volatility is high the edge never clears.

**The system trades only where returns are strongly and stably autocorrelated.**

**V10 (characterisation)**: where it does trade, Sharpe 3.92, Sortino 8.64, max drawdown 0.58%,
profit factor 3.32 — at 0.83% annualised volatility against buy-and-hold's 25%, and 3.3% CAGR
against momentum's 21.3%. It is not beating momentum on return; it is taking a thirtieth of the
risk. Both numbers belong in any summary.

Cost sensitivity: Sharpe 4.21 / 3.92 / 3.34 / 1.55 at 0.5x / 1x / 2x / 5x. The edge does not
depend on perfect execution.

## 18 to 19. Stress and chaos

Fourteen scenarios, held to the brief's standard: when the system cannot establish that its inputs
and state are trustworthy, it does not trade. Tenfold volatility burst, 50% gap, year-long outage,
duplicate bars, out-of-order bars, zero volume, non-finite price, sudden regime change, unwritable
journal, health with nothing to measure, a model that cannot fit, mid-run crash, repeated crashes,
corrupt journal line.

## 20. Reproducibility

`run_manifest` produces a content-hashed identifier over code version, dataset, configuration,
seed, features and models. Order-insensitive, so a reordered feature list is the same run. The
clock is recorded and deliberately excluded from the hash: a run repeated tomorrow with the same
inputs is the same run.

## 21. Mutation testing

**Do not trust the number 81,828.** `validation/mutations.jl` holds twenty deliberate defects.

**20 caught, 0 survived.**

The number that matters more: **seven of the twenty survived the first version of the suite that
was written to catch them.** Each needed a specific weakness closed:

| Mutation | Why it first survived |
|---|---|
| walk-forward embargo | a clean-versus-poisoned comparison cannot see a leak that widens both runs identically |
| negative-rate clamp | the guard fires only on a rounding accident, which is not reproducible across platforms |
| prior mean, twice | every prior the package builds has mean zero, leaving that path untested |
| filtered variance shape | `predictive_df` reads the evolved shape, so a test of it never touched the filtered one |
| unweightable pool fallback | reachable only by setting the log weights directly |
| grid re-weighting | the mutation still produces a plausible expected discount; only accumulation distinguishes it |

Every one of those is a claim nothing was checking, in a suite that was green.

## 22. Performance

Full decision path ~80 us per bar (predict 12–16 us per model, fuse 1.6, decide 1.9, risk 0.3).
Against a one-minute bar that is five orders of magnitude of headroom. Nothing is optimised and
nothing should be.

**V2 (open)**: feature generation allocates 11.9 GB for 10,000 bars against 32 MB for 500, while
time stays near linear. Repeated allocation rather than repeated work. Not a correctness defect;
it caps how large a validation dataset can be.

## 23. Numerical stability

Zero variance, 1e-12 to 1e12 magnitudes, one observation, none at all, 50,000-bar sequences, and
non-finite features refused at the boundary. The probabilistic models fail safely rather than
producing nonsense.

## 24 to 25. Observability

`check_health` gates on the machinery; `assess_model_health` gates on the mathematics — a model
whose intervals have stopped covering, whose posterior has gone numerically strange, or that has
too little evidence is not something that should be sizing positions, and none of that shows up
as an unhealthy process. Both fail closed: a check that cannot measure its input fails.

`MODEL_HEALTH_BOUNDS` are conventional and stated in one place. They are set where a person would
call the behaviour clearly wrong, not where the current system happens to sit.

## 26. Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| No known HIGH-severity correctness defects | **met** | V4 and the `Bar` infinity fixed; V9 and V10 are characterisations, not defects |
| No look-ahead | **met** | 9,011 adversarial assertions; both leak mutations caught |
| No unexplained state divergence | **met** | streaming and offline paths agree; restart replays to the same state |
| No unresolved accounting errors | **met** | paper broker accounting asserted; drawdown corrected |
| No critical numerical instability | **met** | section 23 |
| Risk engine passes adversarial tests | **met** | 72 assertions, 5 mutations caught |
| Paper broker passes accounting tests | **met** | slippage, commission, participation cap all mutation-verified |
| Calibration meets predefined criteria | **partially met** | calibrated on constant-volatility processes; underconfident where volatility moves; overconfident in quiet markets by ~1 point. **No threshold is asserted, because none is statistically justified** — flagged for review as the brief requires |
| Backtest is reproducible | **met** | `run_manifest`; deterministic harness |
| Baselines benchmarked | **met** | section 16, honest harness, no tuning |
| Stress tests fail safely | **met** | fourteen scenarios |
| Mutation testing acceptable | **met** | 20 of 20 |

## 27. Remaining blockers

From `PRE_LIVE_AUDIT.md`, unchanged by this exercise:

1. No live broker exists. `subtypes(Broker) == [PaperBroker]`.
2. No position reconciliation against a venue.
3. `resume!` does not rebuild the position book.
4. The calendar covers 2025 and 2026 only.

Added by this exercise:

5. **V9** — the system abstains on three of four generators. Before live money, it needs to be
   established whether that is the intended behaviour or a threshold that is set too high, and
   the answer needs real data rather than another synthetic sweep.
6. **V2** — quadratic allocation in feature generation caps dataset size.

Still impossible to close by writing code: **Phase F** (live feed validation) and **Phase H**
(calibration against real returns) need a live market data connection.

## 28. Go / no-go

**Verify: pass.** The arithmetic is what it claims to be, the leaks are absent and the absence is
mutation-verified, the guards fire, and the failure behaviour is safe.

**Calibrate: qualified pass.** Calibration is measured, its direction is explained, and the driver
is identified. It is not calibrated everywhere, and where it is not is documented rather than
adjusted away.

**Benchmark: pass, with the result stated plainly.** Against honest baselines on the same costs,
BayesTrade is a low-volatility, high-Sharpe, low-return strategy that abstains from most markets.

**Stress test: pass.**

**Real money: no-go**, unchanged, and this exercise did not attempt to change it. External broker
disabled, real orders impossible, live credentials not required, paper trading and backtesting and
research all available.

The question the brief sets is not whether the backtest made money. It is whether the system's
calculations, probabilities, uncertainty, risk controls, execution simulation and failure
behaviour can be trusted enough to justify the next validation stage.

**Yes** — with V9 named as the thing that most needs a real-data answer, and with the evidence
being the twenty mutations rather than the 81,828 assertions.
