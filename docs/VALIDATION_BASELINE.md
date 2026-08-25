# Validation baseline

Recorded before any code was changed, so that every later claim of improvement or regression has
something to be measured against. Reproduce with:

```bash
julia --project=validation -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=validation validation/baseline.jl
```

Everything is deterministic: one seed, one generated series, one feature set. A validation number
that moves between runs for reasons nobody controls is not a measurement, and the first time it
shifts somebody spends a day looking for a regression that never happened.

Machine: Apple Silicon, 8 CPU threads, Julia 1.10.10. Raw numbers in
`validation/results/baseline.json`.

## Test suite

| | |
|---|---|
| Tests | 51,742 |
| Duration | ~2m 26s |
| Static analysis | Aqua and JET, run first |
| CI matrix | Julia 1.10 and 1.12 |

Green. That is the starting point of this exercise, not the conclusion of it: three of the four
findings below sit in code the suite covers and passes.

## Feature generation

| Dataset | Bars | Examples | Time | Throughput | Allocated |
|---|---|---|---|---|---|
| small | 500 | 479 | 0.005 s | 110k bars/s | 32 MB |
| medium | 2,000 | 1,979 | 0.032 s | 63k bars/s | 431 MB |
| large | 10,000 | 9,979 | 0.581 s | 17k bars/s | **11,963 MB** |

Time is roughly linear. Memory is not: 4x the bars costs 13x the allocation, and 5x more costs
28x. See finding V2.

## Per-bar latency

Measured one stage at a time, because a total says the system is slow and nothing about where.
These are the stages that run on every bar of a live session.

| Stage | Median | Allocated |
|---|---|---|
| `predict` momentum | 15.96 us | 24,552 B |
| `predict` volatility | 13.38 us | 22,960 B |
| `predict` regime | 12.17 us | 20,000 B |
| `update!` regime | 11.08 us | 3,280 B |
| `update!` momentum | 6.04 us | 2,592 B |
| `update!` volatility | 4.92 us | 1,456 B |
| `decide` | 1.92 us | 4,192 B |
| `fuse` | 1.62 us | 10,256 B |
| `review` (risk) | 0.29 us | 856 B |

Total decision path is roughly 80 us per bar. Against a one-minute bar that is five orders of
magnitude of headroom, so nothing here is a bottleneck and nothing should be optimised yet.

## Calibration

1,478 out-of-sample predictions from a walk-forward with a one-bar embargo.

| | |
|---|---|
| Interval error | 0.749% (**underconfident**) |
| Direction ECE | 2.758% |
| Brier score | 0.2324 |
| PIT KS | 0.0161 |
| Mean log score | +2.7837 |
| Sharpness | 1.4830% |
| Bias | +0.0186% |

| Nominal | Empirical | Error |
|---|---|---|
| 50% | 51.1% | +1.1% |
| 60% | 62.2% | +2.2% |
| 70% | 70.0% | +0.0% |
| 80% | 80.0% | -0.0% |
| 90% | 89.1% | -0.9% |
| 95% | 94.0% | -1.0% |
| 99% | 99.0% | -0.0% |

Note this contradicts the dashboard run, which reported overconfident on the same generator with
a different feature set and model set. See finding V1: the direction of miscalibration is
configuration-dependent, so "is it calibrated" has no single answer yet.

## Decisions

Over 1,478 scored bars, at default `RiskLimits`:

| Action | Count |
|---|---|
| no_trade | 757 |
| buy | 381 |
| sell | 340 |

Every refusal was `edge_too_small`. No other decision gate fired once, and no risk gate failed
at all in this configuration, which means most of the risk engine is untested by this baseline
rather than proven by it. That is what section 12 of the brief is for.

## Paper session

2,000 ticks through the full live path.

| | |
|---|---|
| Fills | 571 |
| Reduced by risk | 268 |
| Vetoed | 154 |
| Declined | 754 |
| Rejected | 0 |
| Refits | 30 |
| Settled | 1,478 |
| Stale bars | 0 |
| Halted bars | 0 |
| Return | +16.41% |
| Reported "drawdown" | 0.41% |
| Throughput | 633 bars/s |

Pool weights ended at momentum 0.9905, volatility 0.0088, regime 0.0008. See finding V3.

## Findings

Recorded here, fixed in later sections. Nothing was changed for this baseline.

### V1 — the direction of miscalibration is configuration-dependent — MEDIUM

The dashboard run reported **overconfident** (interval error 1.62%, every coverage point below
nominal). This baseline, same generator and seed family, different feature set and a third model,
reports **underconfident** (0.749%, the low levels above nominal and the high levels below).

Both cannot be a property of "the system". Whatever is driving it is a property of a particular
configuration, and section 10 must find which before anything is adjusted. Inflating variance to
fix the first run would have broken the second.

### V2 — feature generation allocates quadratically — MEDIUM

11.9 GB to build features for 10,000 bars, against 32 MB for 500. Time stays near linear, so this
is repeated allocation rather than repeated work: something is copying a window per bar instead
of viewing it. Ten years of one-minute bars is roughly a million rows, which at this rate does not
fit in any machine. Not a correctness defect, and not worth optimising until sections 2 to 14 have
finished changing the code, but it caps how large a validation dataset can be.

### V3 — the pool collapses onto one model — MEDIUM

Reliability weights finished at 0.9905 / 0.0088 / 0.0008. A linear opinion pool that has put
99% on one component is a single model with two decorations, and the diversification the fusion
layer exists to provide is gone. This may be correct behaviour, since on an AR(1) generator the
momentum model genuinely is the right one, but it needs to be shown rather than assumed, and the
behaviour under a generator that favours no single model needs measuring. Section 7.

### V4 — `session_report` reports the wrong drawdown — HIGH

```julia
"drawdown_pct" => 100 * max(0.0, (session.peak_equity - value) / session.peak_equity)
```

That is the drawdown **at the final bar**, not the maximum drawdown over the run. The key is named
`drawdown_pct` and sits next to `return_pct` in the same report, so it reads as the run's drawdown
and will be compared against one. This baseline reports 0.41% beside a return of 16.41%, which
describes a risk profile the run may not have had.

Anyone sizing on that number is sizing on a number that means something else. Peak equity is
already tracked, so a running maximum costs one field. Fixed in the section 13 pass.
