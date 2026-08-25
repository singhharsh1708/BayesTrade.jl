# Calibration, baselines and stress

Sections 9 to 19 of the validation brief. Reproduce with:

```bash
julia --project=validation validation/calibration.jl
julia --project=validation validation/benchmarks.jl
```

Raw numbers in `validation/results/calibration.json` and `benchmarks.json`.

## Section 10 — what actually drives the miscalibration

The baseline reported the system underconfident; the dashboard run reported it overconfident.
Finding V1 called this configuration-dependent. **That was wrong.** Sweeping five generators, three
feature sets and three model sets, forty-five runs in all:

| Grouped by | overconfident | underconfident | calibrated |
|---|---|---|---|
| `ar1_strong` | 0 | 0 | 9 |
| `ar1_weak` | 0 | 0 | 9 |
| `gaussian` | 0 | 0 | 9 |
| `regime_switching` | 0 | 7 | 2 |
| `stochastic_vol` | 0 | **9** | 0 |
| lean features | 0 | 5 | 10 |
| standard features | 0 | 5 | 10 |
| wide features | 0 | 6 | 9 |
| return only | 0 | 6 | 9 |
| return + volatility | 0 | 6 | 9 |
| all three | 0 | 4 | 11 |

**It is the generator, not the configuration.** Features and model sets barely move it. Where the
data-generating process has constant volatility, the system is calibrated to within half a
percent. Where volatility moves on its own, it is underconfident, and badly.

Nothing in the sweep is overconfident. The dashboard's overconfident reading came from a
different generator and seed, not from a different configuration, and V1's framing should be read
as superseded by this.

### The volatility model earns its place, measurably

On the stochastic-volatility generator, adding it halves the error:

| Models | signed interval error |
|---|---|
| return only | **+0.1247** |
| return + volatility | +0.0522 |
| all three | +0.0535 |

That is the clearest evidence in this exercise that a component is doing what it was added for.

## Section 11 — calibration by regime

Every split is on something known **before** the bar. Bucketing by the realised return conditions
on the quantity being predicted: each bucket's outcomes are truncated on one side and a symmetric
interval mis-covers them by construction. The first version of this table did exactly that and
showed a striking asymmetry between rising and falling markets. It was an artefact, and it
disappeared under an honest split.

| Bucket | n | signed error | verdict |
|---|---|---|---|
| low volatility | 1,148 | **-0.0101** | overconfident |
| mid volatility | 1,182 | +0.0104 | underconfident |
| high volatility | 1,148 | -0.0024 | calibrated |
| predicted up | 3,089 | -0.0001 | calibrated |
| predicted down | 389 | -0.0045 | calibrated |

The one that matters: **the system is overconfident in quiet markets**, by about one percentage
point of coverage. Aggregate calibration hides it, which is the reason section 11 exists. It is
the safer direction than the reverse (being overconfident in a violent market would be the
dangerous one) but it is systematic and it is now measured.

## Sections 14 to 17 — baselines

Same bars, same costs (3bp commission, 5bp slippage), same metrics, and BayesTrade running the
package defaults with no tuning. 3,000 bars.

**AR1, genuine predictability**

| strategy | CAGR | Sharpe | Sortino | max DD | vol | turnover | exposed |
|---|---|---|---|---|---|---|---|
| buy and hold | 6.6% | 0.38 | 0.55 | 53.1% | 25.4% | 1 | 100% |
| random | -26.3% | -1.07 | -1.44 | 98.0% | 25.4% | 2,931 | 100% |
| momentum(20) | **21.3%** | 0.89 | 1.33 | 47.9% | 25.2% | 449 | 99% |
| moving average | 0.6% | 0.12 | 0.18 | 51.9% | 18.1% | 84 | 51% |
| **BayesTrade** | 3.3% | **3.92** | **8.64** | **0.58%** | 0.83% | 73 | 61% |

**Gaussian, regime switching and stochastic volatility: BayesTrade took zero positions.** Not a
small number. Zero, on all three.

## What this says about where the system works

### V9 — it abstains on three generators out of four — HIGH, characterisation

On the gaussian generator that is correct and worth saying plainly: there is no edge, and the
system declining every bar is the decision engine doing its job. `edge_too_small` was the reason
on every one.

On regime switching and stochastic volatility it is a limitation. Both contain exploitable
structure, both are markets a trading system is supposed to have an opinion about, and the answer
was silence for three thousand bars.

The mechanism is not a bug: the edge gate compares the predicted move against the predictive
spread, and where volatility is high the spread is high, so the edge never clears. Whether that
threshold is right is a modelling decision and not one this exercise should make. What it means
today is that **the system trades only where returns are strongly and stably autocorrelated**, and
any expectation set from the AR1 column above should be read with that beside it.

### V10 — where it does trade, it trades very small — MEDIUM, characterisation

Sharpe 3.92 and a maximum drawdown of 0.58% are excellent numbers, and they come with an
annualised volatility of 0.83% against buy-and-hold's 25%. It is not beating momentum on return
(3.3% against 21.3%); it is taking roughly a thirtieth of the risk.

That is a defensible profile and it is not the one a reader of the Sharpe column alone would
picture. Both numbers belong in any summary of this system.

## Section 15 — cost sensitivity

The AR1 positions, repriced:

| costs | CAGR | Sharpe | max DD |
|---|---|---|---|
| 0.5x | 3.55% | 4.21 | 0.54% |
| 1x | 3.30% | 3.92 | 0.58% |
| 2x | 2.79% | 3.34 | 0.70% |
| 5x | 1.29% | 1.55 | 1.63% |

The edge degrades smoothly and survives at five times the assumed cost. It does not depend on
perfect execution.

**A defect in the first version of this harness is worth recording.** It recorded gross returns
and applied costs only to the equity curve, so every risk-adjusted number was blind to costs and
the sweep reported an identical Sharpe of 4.494 at every cost multiple. Returns are now recorded
net. A cost sensitivity study that cannot see costs is worse than none, because it produces a
confident answer.

## Sections 18 and 19 — stress and chaos

`test/test_stress_chaos.jl`, 54 assertions, in the suite.

Held to the standard the brief sets: when the system cannot establish that its inputs and state
are trustworthy, it must not trade.

| Scenario | Behaviour |
|---|---|
| tenfold volatility burst | trades smaller or stops; never the same size through ten times the spread |
| 50% price gap | absorbed, equity finite, nothing rejected |
| year-long outage | halts rather than treating the next bar as though nothing happened |
| duplicate bars | absorbed once; the store is keyed by timestamp |
| out-of-order bars | dropped, not reordered behind decisions already taken |
| zero volume | flagged by the quality check rather than passing as a tradeable price |
| non-finite or non-positive price | refused at the `Quote` boundary |
| sudden regime change | halts across the discontinuity and resumes |
| unwritable journal | halts, and the process survives |
| health with nothing to measure | fails closed |
| model that cannot fit | stays unfitted, predicts nothing, trades nothing |
| crash mid-run | resumes with zero duplicate fills |
| repeated crashes | converge; work is done once |
| corrupt journal line | the record survives |
