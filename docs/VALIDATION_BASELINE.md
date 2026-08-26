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

### V2 — feature generation allocates quadratically — MEDIUM, **fixed**

11.9 GB to build features for 10,000 bars, against 32 MB for 500. Time stays near linear, so this
is repeated allocation rather than repeated work: something is copying a window per bar instead
of viewing it. Ten years of one-minute bars is roughly a million rows, which at this rate does not
fit in any machine. Not a correctness defect, and not worth optimising until sections 2 to 14 have
finished changing the code, but it caps how large a validation dataset can be.

**Cause, found by profiling rather than guessed at.** It was not the features. `walk` is linear
and stays at 11.3 KB per bar at every size tried; `features_at` and `history` are constant per
call. All of it was in `forward_label`:

```julia
ahead = Bar[bar for bar in load_range(store, symbol, entry.timestamp) if ...]
holding = view(ahead, 1:horizon_bars)
```

`load_range` returns every bar from a moment to the end of history, materialised into a fresh
array, and the label keeps `horizon_bars` of them. For row *i* of *n* that copies *n − i* bars,
which sums to O(n²). The only forward read the store offered was an unbounded one, so a caller
wanting three bars had no way to ask for three bars.

**Fix.** `upcoming(store, symbol; after, count)`, the mirror of `history(...; as_of, count)`.
Bisects the same sorted stamps and collects at most `count`.

| Bars | before | after | |
|---|---|---|---|
| 500 | 32.4 MB | **5.9 MB** | 5.5x |
| 2,000 | 430.6 MB | **24.7 MB** | 17x |
| 10,000 | 11,963.4 MB | **125.0 MB** | **96x** |

Time fell with it, 0.581 s to 0.052 s at 10,000 bars, and the scaling is now linear: 4x the bars
costs 4.2x the memory, 5x costs 5.1x.

**Every label is bit-identical.** The test keeps the previous implementation as its reference and
compares `realised_at`, `forward_log_return` and both excursions with `===` across three horizons
and 22,076 assertions. Leak-freedom is unaffected: a label is *defined* by what happens after it,
which is the property a feature must not have, and the adversarial look-ahead suite still passes.

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

### V5 — an invariance test cannot see a symmetric leak — MEDIUM, test weakness

Found by mutating the code the adversarial look-ahead suite is meant to protect.

Deleting the walk-forward embargo (`train_stop = index - embargo - 1` becomes
`index - 1`) left **all 8,078 assertions passing**. The clean-versus-poisoned construction
compares two runs at the same index, and widening the training window widens it identically on
both sides, so nothing moves.

That is a real limit of the technique, not a one-off. Any leak that is a property of the
*algorithm* rather than of the *data* is invisible to it, and the suite needed a direct temporal
statement instead:

```julia
@test record.train_realised_through < record.as_of
```

With that added, the embargo mutation fails immediately. Recorded here because the same blind
spot applies to every invariance test in the file: they prove no *data* leaked, and say nothing
about whether the window was drawn correctly in the first place.

### V6 — the dashboard ruled with fewer gates than the live path — MEDIUM, fixed

`dashboard_payload` called `review(intent, book, limits)` with no volatility and no turnover,
while `PaperTradingSession` supplies both. Three of the nine gates were therefore SKIPPED in the
panel labelled "every bar the system acted on or refused to, with the gate that stopped it".

A ruling computed with fewer gates is a weaker ruling, so the panel was quietly optimistic: it
could show a trade approved that the live path would have vetoed on volatility.

Fixed by supplying `annualise(std(prediction))`, exactly as the session does. Turnover cannot be
supplied there, because a replay record carries the prediction and not the bar, so the liquidity
gate stays visibly SKIPPED rather than being invented. Section 24 should decide whether the
replay record ought to carry enough of the bar to close that last gap.

### V4 — resolved

`max_drawdown_pct` and `current_drawdown_pct` are now reported separately, neither named so the
other could be mistaken for it, and `drawdown_pct` is gone rather than silently redefined.

On the baseline run the numbers are **0.944% maximum against 0.412% current**: the figure that
was being printed, and that `examples/paper_session.jl` labelled "peak drawdown", understated
the worst trough by more than half. Regression test confirmed to fail against the old code.

### Risk engine mutation results

Section 12's suite was checked against five deliberate defects. All five caught:

| Mutation | Result |
|---|---|
| reducing trades charged as opening | 5 failures |
| ceiling becomes a floor (`allowed = headroom`) | 8 errors |
| account-level breach no longer stops the ruling | 2 failures |
| liquidity boundary `>=` becomes `>` | 1 failure |
| kill switch inverted | 15 failures, 5 errors |

### V7 — the prior's ridge penalty enters the noise estimate — MEDIUM, documented not changed

Found while checking the conjugate regression against its closed forms. Not a coding error: the
implementation matches the normal-inverse-gamma posterior to floating point.

```
b_n = b_0 + (y'y + m_0' L_0 m_0 - m_n' L_n m_n) / 2
```

With `m_0 = 0` that difference is **not** the residual sum of squares. It is the residual sum of
squares *plus* the ridge penalty the prior charges the fitted coefficients, `beta' L_0 beta`, and
`E[sigma^2] = b_n / a_n` carries it. Where a coefficient is large against `coefficient_scale`,
the penalty dominates and the model reports signal as noise.

Measured on 2,000 rows with true noise 0.01 and a coefficient of 0.4:

| `coefficient_scale` | `L_0` | `E[sigma^2]` / true | predictive sd |
|---|---|---|---|
| 0.25 | 16.0 | 13.6x | 0.0369 |
| 0.5 (default) | 4.0 | 4.13x | 0.0203 |
| 1.0 | 1.0 | 1.74x | 0.0132 |
| 5.0 | 0.04 | 0.97x | 0.0099 |
| 20.0 | 0.0025 | 0.94x | 0.0097 |

The inflation tracks `beta' L_0 beta / SSE` exactly.

**It does not bite in this system**, and the reason is worth knowing: the model standardises its
features with statistics frozen at fitting time. On the baseline configuration `E[sigma^2]` is
0.9996 of the empirical residual variance at every `coefficient_scale` tried.

That makes the standardiser load-bearing rather than preprocessing, which is now asserted. On a
raw return column (sd near 0.015) the prior precision of 4 swamps an `X'X` of about 0.45, and the
fitted coefficient comes back at **10.4% of what the data says** while the model reports no
particular difficulty. Standardised, it recovers **99.8%**.

Two consequences for the calibration work in sections 9 to 11:

1. A synthetic generator whose coefficients are large in raw units puts the model in the
   prior-dominated regime, and every calibration number then measures the prior rather than the
   model. Generators have to be checked against this before their calibration output means
   anything.
2. Removing or bypassing the scaler, for speed or for simplicity, silently moves the model into
   that regime. The test now fails if anyone does.

No change to the model. Section 10 says find the cause before adjusting, and the cause turns out
to be a property of the prior that the existing design already handles.

### Return model mutation results

| Mutation | Result |
|---|---|
| `a_n = a_0 + n` instead of `a_0 + n/2` | 20 failures |
| epistemic term dropped from the predictive scale | 4 failures |
| negative-rate clamp removed | 1 failure, 1 error |
| prior mean dropped from the rate | 2 failures |
| prior mean dropped from the coefficients | 2 failures |

The last three survived the first version of the suite. The prior-mean pair survived because
every prior the package builds has mean zero, leaving that path untested; a prior with an opinion
now covers it.

The clamp took two attempts and the first one was a mistake worth recording. It built a design of
magnitude 1e6 whose response lay exactly on the fitted plane and asserted the arithmetic would
cancel to a negative rate. It did on Apple Silicon (`-0.5`) and did not on the CI runners, because
the order of summation inside the solve depends on the BLAS and its thread count. **A test whose
premise is a rounding accident is a flaky test, however real the guard it covers**, and it passed
on the pull request that introduced it before failing on the next one. The second attempt still failed on Julia 1.12, for the
same reason at a smaller scale: it fitted three hundred random rows and halved the response sum
of squares, and the solve behind it was still large enough for the platform to matter.

The third attempt has no arithmetic in its premise at all. Two observations entered by hand, the
response sum of squares set to zero, and an unclamped rate of -0.99999999999 on every platform.
The mutation that removes the clamp is still caught.

The third attempt still failed, and the cause was not the test. An earlier edit had spliced a
replacement block into the file at the wrong offset and left the original, flaky testset in place
below it. Both ran; the deterministic one passed everywhere and the duplicate kept failing on
1.12, which is what the CI log had been reporting all along. Removed, and both Julia versions now
run locally before anything is pushed.

Two lessons, and the second cost more than the first:

1. **A defensive branch that only fires on a rounding accident cannot be tested by reproducing
   the accident.** Put the state into the shape the branch defends against, directly.
2. **Read the failing line, not the failing name.** Three fixes went to a testset with the right
   title while the failure was coming from a second copy of it forty lines further down.

## Sections 5 to 8

### V8 — the agreement label measures location, not confidence — informational

Section 8 asked for model disagreement to be first-class. `disagreement_share(prediction)` is the
share of predictive variance attributable to the models disagreeing, and `model_agreement`
labels it HIGH, MEDIUM or LOW. Both are exposed in the decision journal and the dashboard payload.

The bounds (0.10 and 0.35) are conventional, not derived, and **nothing branches on the label**.
The brief is explicit that trading behaviour must not change on it without statistical
justification, and none has been established.

One thing worth writing down, found by a test that failed for the right reason. A model pairing a
confident forecast with a useless one (`sd = 5.0` against `sd = 0.010`) reports **AGREEMENT_HIGH**,
because the two agree about *where* the return will be and differ only in how sure they are.
Agreement is about location; confidence is a separate axis. A reader who takes a high agreement
label as "the models are confident" has read it backwards, and the test now says so in place.

### Mutation results, sections 5 to 8

| Mutation | Result |
|---|---|
| filtered shape discounted along with the data | 7 failures |
| evolved shape discounted along with the data | 2 failures |
| grid re-weighting frozen | 275 failures |
| disagreement dropped from the fused epistemic variance | 1 failure |
| zero-total weights collapse onto the first model | 2 failures |
| the regime chain never propagates | 1 failure |
| a skipped bar invents a zero return | 2 failures |
| **variance rate floor removed** | **0 — equivalent mutant** |

Three of these survived the first version of the suite and needed it strengthened.

**The filtered and evolved shapes are different functions**, answering "how volatile is it now"
and "how volatile will the next bar be". `predictive_df` reads only the evolved one, so a
mutation to the filtered one survived a test that read only `predictive_df`. Both are pinned now.

**The all-models-unusable branch was untested.** When every log weight has gone to minus infinity
the pool falls back to uniform; collapsing onto whichever model happens to be first would be an
arbitrary choice presented as a decision. Reachable in a test only by setting the log weights
directly, which is what the test does.

**The variance rate floor is an equivalent mutant, and by design.** The prior sits outside the
discounted statistics, so the rate is the prior's rate plus something non-negative and the clamp
can never bind. The module docstring already claimed this; it is now asserted rather than
asserted-about. Recorded here so it is not re-reported as a surviving mutation later.
