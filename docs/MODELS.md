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
