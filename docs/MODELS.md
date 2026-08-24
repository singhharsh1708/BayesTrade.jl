# Models

## The contract

```julia
fit!(model, observations)                              # batch, offline, may be expensive
update!(model, observation)                            # one recursive step, closed form
predict(model, features; symbol, as_of, horizon_bars)  # a ProbabilisticResult
uncertainty(model)                                     # a scalar the fusion layer can read
```

Predicting before fitting throws `NotFittedError` rather than returning a prior-only guess. A
prior-only guess looks like a prediction and pollutes the record with beliefs the model never
formed.

Julia has no field inheritance, so fitted state is composed rather than inherited: each model
holds a `FitState` and exposes it through `fit_state`. That is more honest than inheritance
would have been, since the shared state is a visible field rather than something a subclass
silently acquires.

## The Bayesian return model

A conjugate linear regression from point-in-time features to the forward log return.

```
beta | sigma^2 ~ Normal(m0, sigma^2 V0)
sigma^2        ~ InverseGamma(a0, b0)
```

Everything is closed form. Absorbing an observation is a rank-one update to a precision
matrix; predicting is one triangular solve. That is the whole reason for choosing a conjugate
prior over something more expressive: it runs inside the trading loop, so it may not sample,
iterate to convergence, or take a variable amount of time.

### The predictive is Student-t

```
y* ~ StudentT(df = 2·a_n, loc = x'm_n, scale² = (b_n/a_n)(1 + x'Λ_n⁻¹x))
```

This is what an unknown noise variance implies. A normal in its place would understate the
probability of a large loss, which is the one quantity this system must not understate.

The scale splits into a part the data cannot remove and a part that shrinks with evidence:

| Term | Meaning |
| --- | --- |
| `1` | Observation noise, irreducible |
| `x'Λ_n⁻¹x` | Coefficient uncertainty, shrinks as data arrives |

`predict` returns both, and `ProbabilisticResult` carries the epistemic share alongside the
distribution rather than inside it — whether a spread is reducible is a statement about why
the distribution is that wide, not a property of the distribution.

### Standardisation is a fitted parameter

Standardising with statistics computed over the whole sample leaks the future: the mean and
scale of a feature over 2022–2026 are not knowable in 2023. `FeatureScaler` is estimated on
the training window, **stored with the model**, and applied unchanged afterwards.

A consequence worth knowing: a batch fit over 1000 rows and a 100-row fit followed by 900
updates do **not** agree, because they standardise against different windows. Only the first
100 rows were knowable when the second model was fitted. That is correct behaviour, and there
is a test asserting it rather than papering over it.

### The horizon is part of the identity

A model fitted on 1-bar returns refuses to answer about 20-bar returns. Silently answering
would look reasonable and be wrong by a factor of four in the predictive scale — exactly the
quantity the risk engine reads.

### Forgetting

`forgetting < 1` decays observations geometrically, so the effective sample size converges to
`1/(1−λ)` rather than growing. A model fitted on five years of history is usually wrong about
this month, and a posterior that has hardened around old data cannot notice.

Only the data decays. The prior is kept separate and is not discounted.

### The response scale is a dispatch point

`response_scale(model, features)` returns `1.0`. The plain model assumes a constant noise
scale; a model that knows better defines its own method, and the conjugate update, the version
hash and the predictive all follow unchanged. That is how the volatility-scaled variant arrives
in Phase 5 without a second code path.

## The Bayesian volatility model

A conjugate inverse-gamma filter over the variance of one bar, and a posterior over how fast
to forget it.

```
r | sigma^2 ~ Normal(mu, sigma^2)      z = (r - mu)^2

S <- delta S + w                       Q <- delta Q + w z
a  = a0 + S / 2                        b  = b0 + Q / 2
```

Volatility moves, so old evidence has to decay. How fast is the whole question, and it is not
answerable in advance: the right memory length in a quiet market is not the right one in a
violent one, and nothing announces which market today is. The usual answer is to pick a decay
rate by hand. This runs the same filter at memory lengths of 10, 20, 40, 100 bars and forever,
and weights them by their own one-step predictive record:

```
log wt_k <- alpha log w_k + L_k        w = softmax(log wt)
```

Left to itself, that posterior works out which market it is in. On stochastic-volatility data
`E[delta]` settles near 0.91 and the shortest memory leads; on constant-volatility data it
settles near 0.97. Nothing tells it which is which. Carrying the grid costs about 0.005 nats
against the best single choice made in hindsight, and beats the worst by 0.23.

### Two guarantees that are structural rather than clamped

| Guarantee | Consequence |
| --- | --- |
| `a >= a0 > 1` | `df > 2` always, so the predictive mean and variance always exist |
| `b >= b0 > 0` | the rate floor is unreachable, so a flat bar needs no special case |

The second is why a zero return is harmless here. It is not a degenerate case to guard
against; it is an ordinary and rather informative observation, and no logarithm is ever taken
of an observation. The grid evidence drops every term of the log marginal that does not depend
on the component, which is exact because such terms cancel in a softmax, and which removes the
one term that would have been infinite at a flat bar.

### What it emits

A distribution over the forward **return**, not over volatility. A volatility forecast cannot
be falsified against a number the market prints; a return predictive can, and the calibration
machinery already scores exactly that. The consequence is that `walk_forward` and `assess`
needed no changes at all to score this model.

Two things about that output look like defects and are not:

* The predictive is centred, so `P(return > 0)` is one half by construction and the Brier
  score is exactly 0.25. The directional half of a calibration report carries no information
  about this model. It forecasts spread, and spread is what should be scored.
* `uncertainty` does not fall to zero with more data. It plateaus, because under a discount
  you never become certain about a moving target. Measured on 600, 1500 and 4000 bars it sits
  at 0.175, 0.198, 0.194; the same filter with no forgetting falls 0.029, 0.018, 0.011.

Beside the return predictive sit the exact volatility posterior, a mixture of inverse-gammas,
and the exact predictive for realised variance over `h` bars, a mixture of `2b BetaPrime(h/2,
a)`. The mean of the second must equal the variance of the first, and does to a part in
`1e12`. Two families derived independently agreeing on one quantity is a real check on both.

`E[sigma]` and `sqrt(E[sigma^2])` are reported separately and deliberately. They differ by
Jensen, and a position sizer that squares an expected volatility to get a variance
systematically under-reserves, by more the less certain the filter is.

### A missing bar is not a quiet bar

A halted feed and a genuinely flat market both produce a return of zero, and the number alone
cannot tell them apart. Getting it wrong is not symmetric: 500 absorbed zeros drive the
estimate to 0.066 annual, while 500 skipped bars leave it at 0.382. Absorbing errs **narrow**,
which is the one direction a risk system must never err.

So the model never inspects the return to decide. If the feature is absent the bar is skipped,
which ages the state without informing it, and the posterior widens back toward the prior
rather than holding the last value. That is the exact closed-form answer, not an approximation
of one: with no evidence the statistics decay to zero and the posterior returns to where it
started. Deciding which bars are untrustworthy belongs to the data-quality layer, not here.

## Measuring the response in units of volatility

A regression assumes its noise scale is constant. On returns that is false in a way that
matters: the same coefficients describe a market whose bars are five times wider in a crisis,
and a model fitted across both is fitted to neither.

`response_scale` is the hook. The plain model returns one and every path reduces to the
ordinary regression. A `VolatilityScale` policy divides the response by a point-in-time
volatility before fitting and multiplies the predictive back afterwards, so one set of
coefficients describes both regimes.

```julia
BayesianReturnModel([:log_return_1]; horizon_bars = 1, policy = VolatilityScale(:volatility_20))
```

Measured out of sample on stochastic-volatility data, refitting every twenty bars:

| | interval error | mean log score |
| --- | --- | --- |
| plain | 10.56% | 1.3988 |
| volatility-scaled | 6.26% | 2.1446 |

The model is **parameterised** on the policy rather than carrying it as a field, so a scaled
model is a different type with its own version (`0.2.0` against `0.1.0`) and its own parameter
hash. Two models with identical regression state and different scaling do not make the same
predictions and must not claim the same identity.

The scale is read from a point-in-time **feature**, never from a volatility model handed in at
prediction time. A feature is built from bars that have closed, so a scaled model cannot reach
forward even by accident.

The floor is not a fudge. A volatility feature is exactly zero on a flat window, and zero is
not a scale: it would divide a real return by nothing when fitting and collapse the predictive
to a point when predicting. The floor is what a flat window is worth, and it is recorded in
the model's parameters so a reader can see which one was used.

## Calibration

A model that says 70% and is right half the time is worse than useless, because everything
downstream sizes positions off that number. Sharpe ratio and hit rate say nothing about it.

Two checks, because the posterior is used two ways:

**The whole distribution**, through the probability integral transform. `u = F(y)` is uniform
if and only if the model is calibrated, and the shape says what is wrong:

| PIT shape | Diagnosis |
| --- | --- |
| Mass at the edges | Intervals too narrow — overconfident |
| Hump in the middle | Intervals too wide |
| Shifted | Biased |

**The directional probability** `P(return > 0)`, with its own reliability curve and Brier
score, because that is the number the decision engine reads.

Sharpness is reported alongside. A model that always predicts the unconditional distribution
is perfectly calibrated and worth nothing, so calibration is a constraint to satisfy rather
than a score to maximise.

`is_overconfident` averages **signed** coverage errors, so read it alongside the mean absolute
error rather than instead of it. A constant-scale model fitted to changing volatility is too
wide in the middle and too narrow in the tails, and those errors cancel there while every
individual level is wrong.

## Walk-forward, and the embargo

Scoring a model on the data it was fitted to measures nothing. But the obvious fix leaks too:

> A training example at time `t` is not usable until its label has been realised, which for a
> horizon of `h` bars is `h` bars later.

Fitting on everything up to `t` and predicting at `t` uses labels that had not happened yet,
and the leak is invisible in the output — the backtest simply looks better than it is. The
training window therefore stops `h` bars short of every prediction, and `PredictionRecord`
refuses to construct if that was violated.

## Persistence

A model is stored as its **sufficient statistics**, not its posterior. Storing the mean and
covariance would lose the ability to keep updating, and a saved model that cannot absorb
tomorrow's observation is not much of a saved model.

JSON, not Julia serialisation: a model file should be readable, diffable and checkable into a
repository, and `Serialization` produces a binary blob tied to the exact package versions that
wrote it — precisely the wrong property for a record meant to explain a trade months later.

The bundle carries the parameter hash the model had when written, and loading verifies it. A
truncated or hand-edited file fails at load rather than silently trading on the wrong
coefficients.

Unlike the bar files, which are eight fixed columns and hand-written, this uses a JSON library.
The format here is nested and irregular, with string escaping and number parsing to get wrong;
hand-rolling that would be a liability rather than avoiding one.

## Using it

```julia
using BayesTrade, Dates

store  = load_store("data/nse")
engine = FeatureEngine(store, minimal_feature_set())
report = fit_return_model(engine, "RELIANCE"; horizon_bars = 5)

println(summarise(report))
save_model(report.model, "models/reliance.json")
```

```
RELIANCE: 1174 labelled rows, 3 features, 5-bar horizon

calibration over 919 predictions
  interval error   1.000%  (overconfident)
  direction ECE    9.271%
  Brier score      0.2650
  PIT KS           0.0859
  sharpness        3.4484%

  level   nominal   empirical
     50%     50.0%       49.2%
     90%     90.0%       87.8%
     95%     95.0%       94.2%

fitted momentum@0.1.0+fb9d7283 on 1174 rows
  residual scale 0.034942, uncertainty 0.0292

  column         standardised          std       z
  intercept        0.00301842   0.00101806    2.96
  log_return_1   -0.000206605   0.00101902   -0.20
  momentum_20_1    -0.0015329   0.00102292   -1.50
  volatility_20  -0.000514004   0.00102283   -0.50
```

That output is from the synthetic generator, whose returns are independent by construction.
Every feature `z` is small and the Brier score sits near 0.25, a coin. **That is the correct
answer**, and a model reporting an edge there would be fitting noise.

On `AR1Returns(phi = 0.35)` the same harness reports a `z` above 5 on `log_return_1` and a
Brier score below 0.25 — the edge that is genuinely there.
