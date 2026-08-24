"""
A three-state market regime filter with no expectation-maximisation anywhere.

The states are the ones the `Regime` enum already names: bull, bear, sideways. The chain is
hidden, so the honest thing to report is a posterior over which state today is, and a
predictive that mixes over that posterior rather than committing to a label.

The usual way to fit a hidden Markov model is Baum-Welch, which is expectation-maximisation:
iterative, of variable duration, and therefore banned from anything that runs in the trading
loop. Two choices avoid it entirely.

**The transition matrix has one parameter.**

    A = lambda I + (1 - lambda) 1 pi'

Sticky-restart: hold with probability `lambda`, otherwise redraw from the long-run
distribution. Then `pi` is the stationary distribution by construction rather than by an
eigenproblem, `A^h` has a closed form, and the whole matrix is identified by a single scalar
that squared returns reveal clearly.

**State identity is prior, not learned.** Each state's drift and variance are fixed multiples
of shape vectors set in advance:

    mu_k = c + m g_k        sigma_k^2 = s^2 (1 + kappa u_k)

with `g` negative for bear and `u` largest for bear. Four scalars are estimated from the data
and the sign pattern never moves. This is what makes the model immune to label switching: a
learned mixture can swap which component is called "bear" between refits, which would make
every stored prediction and every parameter hash meaningless without raising anything.

The cost is stated rather than hidden. The direction axis is identified mostly by the fact
that bear markets are violent, which is prior structure rather than something the data taught
the model, and the drift estimate is roughly all noise at any window a trader has. A market
whose drawdowns are quiet is one this model reads poorly.
"""

const REGIME_STATES = (BULL, BEAR, SIDEWAYS)
const N_REGIMES = 3
const REGIME_BLOCK = 21
const REGIME_BASELINE = 20
const REGIME_VARIANCE_FLOOR = 0.15

"""
    regime_shape(values, stationary)

Centre and scale a shape vector so it has mean zero and unit variance under `stationary`.

Both properties are load-bearing: mean zero makes the state drifts average to the sample mean
exactly, and unit variance makes the estimated spread `m` mean what it says.
"""
function regime_shape(values::NTuple{N_REGIMES, Float64}, stationary::Vector{Float64})
    centre = 0.0
    for index in 1:N_REGIMES
        centre += stationary[index] * values[index]
    end
    spread = 0.0
    for index in 1:N_REGIMES
        spread += stationary[index] * (values[index] - centre)^2
    end
    spread > 0 || throw(ArgumentError("a shape vector must vary across states"))
    scale = sqrt(spread)
    return Float64[(values[index] - centre) / scale for index in 1:N_REGIMES]
end

"""
    RegimePrior

What is believed before any data: where the states sit relative to each other, and how firmly.
"""
struct RegimePrior
    stationary::Vector{Float64}
    drift_shape::Vector{Float64}
    variance_shape::Vector{Float64}
    persistence::Float64
    drift_scale::Float64
    dispersion::Float64
    shape::Float64
    strength::Float64

    function RegimePrior(;
            stationary = (0.35, 0.25, 0.4),
            drift_values = (1.0, -1.0, 0.0),
            variance_values = (-0.5, 1.5, 0.0),
            persistence::Real = 1 - 1 / 25,
            expected_duration::Union{Real, Nothing} = nothing,
            drift_scale::Real = 0.25 / BARS_PER_YEAR,
            dispersion::Real = 0.55,
            shape::Real = 2.0,
            strength::Real = 250.0,
        )
        weights = convert(Vector{Float64}, collect(stationary))
        length(weights) == N_REGIMES ||
            throw(ArgumentError(string("need ", N_REGIMES, " weights, got ", length(weights))))
        for weight in weights
            weight > 0 || throw(ArgumentError(string("stationary weights must be positive")))
        end
        total = sum(weights)
        isapprox(total, 1.0; atol = 1.0e-9) ||
            throw(ArgumentError(string("stationary weights sum to ", total, ", not 1")))

        held = expected_duration === nothing ? Float64(persistence) :
            1 - 1 / Float64(expected_duration)
        0 <= held < 1 ||
            throw(ArgumentError(string("persistence must lie in [0, 1), got ", held)))
        drift_scale >= 0 ||
            throw(ArgumentError(string("drift_scale must be non-negative, got ", drift_scale)))
        dispersion >= 0 ||
            throw(ArgumentError(string("dispersion must be non-negative, got ", dispersion)))
        shape > 1 || throw(ArgumentError(string("shape must exceed 1, got ", shape)))
        strength >= 0 ||
            throw(ArgumentError(string("strength must be non-negative, got ", strength)))

        drifts = regime_shape(
            (Float64(drift_values[1]), Float64(drift_values[2]), Float64(drift_values[3])),
            weights,
        )
        variances = regime_shape(
            (
                Float64(variance_values[1]), Float64(variance_values[2]),
                Float64(variance_values[3]),
            ),
            weights,
        )
        drifts[1] > drifts[2] || throw(
            ArgumentError("the first state must have the higher drift, or the labels lie"),
        )
        return new(
            weights, drifts, variances, held, Float64(drift_scale), Float64(dispersion),
            Float64(shape), Float64(strength),
        )
    end
end

"""
    regime_transition(persistence, stationary)

`A = lambda I + (1 - lambda) 1 pi'`, the sticky-restart chain.

Row-stochastic by construction, with `pi` as its stationary distribution and `lambda` as its
second eigenvalue, so nothing here has to be solved for or checked afterwards.
"""
function regime_transition(persistence::Float64, stationary::Vector{Float64})
    matrix = Matrix{Float64}(undef, N_REGIMES, N_REGIMES)
    for row in 1:N_REGIMES, column in 1:N_REGIMES
        held = row == column ? persistence : 0.0
        matrix[row, column] = held + (1 - persistence) * stationary[column]
    end
    return matrix
end

"""
    RegimeParameters

The four estimated scalars, and the per-state quantities they imply.
"""
struct RegimeParameters
    prior::RegimePrior
    centre::Float64
    drift_spread::Float64
    persistence::Float64
    mean_variance::Float64
    dispersion::Float64
    n_rows::Int
    drifts::Vector{Float64}
    variances::Vector{Float64}
    concentrations::Vector{Float64}
    shapes::Vector{Float64}

    function RegimeParameters(
            prior::RegimePrior; centre::Real, drift_spread::Real, persistence::Real,
            mean_variance::Real, dispersion::Real, n_rows::Integer,
        )
        spread = Float64(drift_spread)
        held = Float64(persistence)
        variance = Float64(mean_variance)
        spread >= 0 || throw(ArgumentError("drift_spread must be non-negative"))
        0 <= held < 1 || throw(ArgumentError("persistence must lie in [0, 1)"))
        variance > 0 || throw(ArgumentError("mean_variance must be positive"))
        n_rows >= 0 || throw(ArgumentError("n_rows must be non-negative"))

        # Capped against the most negative shape entry, the one that can drive a variance to
        # zero, so the quietest state keeps a floor share with no root-finding.
        widest = -minimum(prior.variance_shape)
        allowed = (1 - REGIME_VARIANCE_FLOOR) / max(1.0e-12, widest)
        spreadable = min(Float64(dispersion), allowed)

        drifts = Float64[
            Float64(centre) + spread * prior.drift_shape[index] for index in 1:N_REGIMES
        ]
        variances = Float64[
            variance * (1 + spreadable * prior.variance_shape[index]) for
                index in 1:N_REGIMES
        ]
        rows = Float64(n_rows)
        concentrations = Float64[prior.stationary[index] * rows + 1 for index in 1:N_REGIMES]
        shapes = Float64[
            prior.shape + prior.stationary[index] * rows / 2 for index in 1:N_REGIMES
        ]
        return new(
            prior, Float64(centre), spread, held, variance, spreadable, Int(n_rows),
            drifts, variances, concentrations, shapes,
        )
    end
end

transition_matrix(parameters::RegimeParameters) =
    regime_transition(parameters.persistence, parameters.prior.stationary)

"""
    emission(parameters, state; horizon_bars = 1)

One state's contribution to the predictive, as a Student-t.

The scale carries two terms. `h` is within-state noise accumulating over the horizon; `h^2/k`
is a mean the model does not know exactly, held over the whole horizon, so it grows faster.
Student-t rather than normal because the parameters are uncertain, and because a t keeps the
filter's normaliser away from zero for any finite return, which makes "this bar was impossible
under every regime" structurally unreachable.
"""
function emission(parameters::RegimeParameters, state::Integer; horizon_bars::Integer = 1)
    horizon = Float64(horizon_bars)
    shape = parameters.shapes[state]
    concentration = parameters.concentrations[state]
    scale = sqrt(
        parameters.variances[state] * ((shape - 1) / shape) *
            (horizon + horizon^2 / concentration),
    )
    return student_t(horizon * parameters.drifts[state], scale, 2 * shape)
end

"""
    autocovariance(values, lag)

Sample autocovariance at one lag, about the sample mean.
"""
function autocovariance(values::Vector{Float64}, lag::Integer)
    n = length(values)
    n > lag || return 0.0
    centre = mean(values)
    total = 0.0
    for index in 1:(n - lag)
        total += (values[index] - centre) * (values[index + lag] - centre)
    end
    return total / (n - lag)
end

"""
    block_variance(values, block)

Variance of the underlying mean, debiased for the noise a block mean still carries.

A block mean of `L` bars is the state's drift plus noise of variance `sigma^2 / L`, so the
variance of the block means overstates the drift's own variance by exactly that, and it is
subtracted rather than hoped away.
"""
function block_variance(values::Vector{Float64}, block::Integer)
    n_blocks = length(values) ÷ block
    n_blocks >= 2 || return 0.0
    means = Vector{Float64}(undef, n_blocks)
    for index in 1:n_blocks
        total = 0.0
        for offset in 1:block
            total += values[(index - 1) * block + offset]
        end
        means[index] = total / block
    end
    inflated = var(means) - var(values) / block
    return inflated / (1 - 1 / block)
end

"""
    estimate_regime_parameters(returns; prior)

Estimate the four scalars in five passes over the returns and about thirty operations.

No optimiser and no iteration to convergence. Each scalar has one estimator chosen because it
is identified at a window a trader actually has:

* the centre is the sample mean
* the drift spread comes from block means, debiased
* the persistence comes from **squared** returns, where the regime signal actually lives; the
  textbook estimator from return autocovariances is a ratio of two noise terms and is useless
  at any affordable window
* the variance dispersion inverts the one-lag autocovariance of squared returns, using the
  shrunk persistence rather than the noisier two-lag term

Everything is shrunk toward the prior by `n / (n + strength)`, which is what keeps a short
window from producing confident nonsense.
"""
function estimate_regime_parameters(
        returns::AbstractVector{<:Real}; prior::RegimePrior = RegimePrior(),
    )
    values = convert(Vector{Float64}, returns)
    all(isfinite, values) || throw(ArgumentError("returns must be finite"))
    n = length(values)
    n >= 2 || throw(ArgumentError(string("need at least 2 returns, got ", n)))

    weight = n / (n + prior.strength)
    centre = mean(values)
    total_variance = var(values)

    drift_variance = weight * max(0.0, block_variance(values, REGIME_BLOCK)) +
        (1 - weight) * prior.drift_scale^2
    drift_spread = sqrt(max(drift_variance, 0.0))

    squares = Float64[(value - centre)^2 for value in values]
    first_lag = autocovariance(squares, 1)
    far_lag = autocovariance(squares, REGIME_BASELINE)
    # Read across a long baseline rather than between neighbouring lags. Consecutive lags
    # differ by a few per cent where the sampling noise is fifteen, so the short-baseline
    # ratio is mostly noise: measured over twelve seeds at 10000 bars it has a standard
    # deviation of 0.031 and reaches 0.992, against 0.018 and 0.953 here.
    measured = (first_lag > 0 && far_lag > 0) ?
        (far_lag / first_lag)^(1 / (REGIME_BASELINE - 1)) : prior.persistence
    persistence = clamp(
        weight * measured + (1 - weight) * prior.persistence, 0.5, 0.99,
    )

    mean_variance = max(total_variance - drift_variance, 1.0e-12)
    variance_of_variance = weight * max(0.0, first_lag / persistence) +
        (1 - weight) * (prior.dispersion * mean_variance)^2
    dispersion = sqrt(max(variance_of_variance, 0.0)) / mean_variance

    return RegimeParameters(
        prior; centre = centre, drift_spread = drift_spread, persistence = persistence,
        mean_variance = mean_variance, dispersion = dispersion, n_rows = n,
    )
end

"""
    RegimeFilter

The forward recursion: a posterior over today's state, updated one bar at a time.
"""
mutable struct RegimeFilter
    parameters::RegimeParameters
    transitions::Matrix{Float64}
    belief::Vector{Float64}
    log_likelihood::Float64
    n_seen::Int
    n_skipped::Int

    function RegimeFilter(parameters::RegimeParameters)
        return new(
            parameters, transition_matrix(parameters),
            copy(parameters.prior.stationary), 0.0, 0, 0,
        )
    end
end

n_states(::RegimeFilter) = N_REGIMES
n_absorbed(filter::RegimeFilter) = filter.n_seen
n_skipped(filter::RegimeFilter) = filter.n_skipped

"""
    regime_probabilities(filter)

`P(state today | everything seen)`, in the order of [`REGIME_STATES`](@ref).
"""
regime_probabilities(filter::RegimeFilter) = copy(filter.belief)

"""
    regime_belief(filter)

The state posterior as a [`LabelledCategorical`](@ref), carrying its own labels.
"""
regime_belief(filter::RegimeFilter) =
    LabelledCategorical(collect(REGIME_STATES), copy(filter.belief))

"""
    propagate(belief, transitions)

One step of the chain, before any new evidence.
"""
function propagate(belief::Vector{Float64}, transitions::Matrix{Float64})
    moved = Vector{Float64}(undef, N_REGIMES)
    for column in 1:N_REGIMES
        total = 0.0
        for row in 1:N_REGIMES
            total += belief[row] * transitions[row, column]
        end
        moved[column] = total
    end
    return moved
end

"""
    observe_return!(filter, value)

Absorb one bar: propagate, then correct by the emission likelihood.

Done in logs with an explicit maximum shift. Renormalising every step is not an optimisation:
the unnormalised recursion underflows within a few hundred bars and then reports the prior
forever, silently.
"""
function observe_return!(filter::RegimeFilter, value::Real)
    observation = Float64(value)
    isfinite(observation) ||
        throw(ArgumentError(string("a return must be finite, got ", value)))

    moved = propagate(filter.belief, filter.transitions)
    densities = Vector{Float64}(undef, N_REGIMES)
    largest = -Inf
    for index in 1:N_REGIMES
        density = logpdf(emission(filter.parameters, index), observation)
        densities[index] = density
        density > largest && (largest = density)
    end

    total = 0.0
    for index in 1:N_REGIMES
        moved[index] *= exp(densities[index] - largest)
        total += moved[index]
    end
    total > 0 || throw(
        ArgumentError(string("no regime can account for a return of ", observation)),
    )

    for index in 1:N_REGIMES
        filter.belief[index] = moved[index] / total
    end
    filter.log_likelihood += log(total) + largest
    filter.n_seen += 1
    return filter
end

"""
    skip_observation!(filter)

Age the chain past a bar that cannot be read.

Marginalising over the unseen bar, which is the exact answer rather than an approximation of
one. Substituting a zero return would be evidence for the quiet state, and inventing evidence
from a gap in the data is how a model becomes confidently wrong.
"""
function skip_observation!(filter::RegimeFilter)
    filter.belief = propagate(filter.belief, filter.transitions)
    filter.n_skipped += 1
    return filter
end

"""
    fit_filter!(filter, returns)

Run the recursion over a series, starting again from the stationary distribution.
"""
function fit_filter!(filter::RegimeFilter, returns::AbstractVector{<:Real})
    values = convert(Vector{Float64}, returns)
    all(isfinite, values) || throw(ArgumentError("returns must be finite"))
    reset!(filter)
    for value in values
        observe_return!(filter, value)
    end
    return filter
end

"""
    reset!(filter)

Forget the data and return to the stationary distribution.
"""
function reset!(filter::RegimeFilter)
    for index in 1:N_REGIMES
        filter.belief[index] = filter.parameters.prior.stationary[index]
    end
    filter.log_likelihood = 0.0
    filter.n_seen = 0
    filter.n_skipped = 0
    return filter
end

"""
    horizon_weights(filter; horizon_bars = 1)

`P(state in h bars | everything seen)`, exactly and in `O(K)`.

`A^h = lambda^h I + (1 - lambda^h) 1 pi'`, so the belief decays geometrically toward the
long-run distribution and no matrix power is ever formed.
"""
function horizon_weights(filter::RegimeFilter; horizon_bars::Integer = 1)
    horizon_bars >= 1 ||
        throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
    held = filter.parameters.persistence^horizon_bars
    stationary = filter.parameters.prior.stationary
    return Float64[
        held * filter.belief[index] + (1 - held) * stationary[index] for
            index in 1:N_REGIMES
    ]
end

"""
    predict_return(filter; horizon_bars = 1)

The predictive over the forward log return, mixing the emissions by where the chain will be.

A distribution over a number the market prints, not over a label. That is what the calibration
machinery scores, and a regime model that could only report state probabilities would be
unfalsifiable.
"""
function predict_return(filter::RegimeFilter; horizon_bars::Integer = 1)
    weights = horizon_weights(filter; horizon_bars = horizon_bars)
    components = [
        emission(filter.parameters, index; horizon_bars = horizon_bars) for
            index in 1:N_REGIMES
    ]
    return MixtureModel(components, weights)
end

"""
    variance_decomposition(filter; horizon_bars = 1)

Predictive variance split three ways, as an identity rather than a definition.

`aleatoric` is within-state noise plus the chain jumping, and no amount of data removes either.
`state` is not knowing which regime today is, and evidence does remove that. `parameter` is not
knowing the state means. The three sum to the predictive variance exactly.

Lumping the whole component spread into the epistemic bucket, which is the obvious shortcut,
overstates what evidence can fix: part of that spread is the transition itself firing.
"""
function variance_decomposition(filter::RegimeFilter; horizon_bars::Integer = 1)
    horizon_bars >= 1 ||
        throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
    horizon = Float64(horizon_bars)
    parameters = filter.parameters
    stationary = parameters.prior.stationary
    held = parameters.persistence^horizon_bars

    means = Float64[horizon * parameters.drifts[index] for index in 1:N_REGIMES]
    noise = Float64[horizon * parameters.variances[index] for index in 1:N_REGIMES]

    conditional = Vector{Float64}(undef, N_REGIMES)
    for row in 1:N_REGIMES
        total = 0.0
        for column in 1:N_REGIMES
            step = (row == column ? held : 0.0) + (1 - held) * stationary[column]
            total += step * means[column]
        end
        conditional[row] = total
    end

    centre = 0.0
    for row in 1:N_REGIMES
        centre += filter.belief[row] * conditional[row]
    end

    aleatoric = 0.0
    for row in 1:N_REGIMES
        inner = 0.0
        for column in 1:N_REGIMES
            step = (row == column ? held : 0.0) + (1 - held) * stationary[column]
            inner += step * (noise[column] + (means[column] - conditional[row])^2)
        end
        aleatoric += filter.belief[row] * inner
    end

    state = 0.0
    for row in 1:N_REGIMES
        state += filter.belief[row] * (conditional[row] - centre)^2
    end

    weights = horizon_weights(filter; horizon_bars = horizon_bars)
    parameter = 0.0
    for index in 1:N_REGIMES
        parameter += weights[index] * parameters.variances[index] * horizon^2 /
            parameters.concentrations[index]
    end

    return (aleatoric = aleatoric, state = state, parameter = parameter)
end

"""
    predict(filter; horizon_bars = 1)

The predictive and its reducible share, as a `(distribution, epistemic_variance)` pair.
"""
function predict(filter::RegimeFilter; horizon_bars::Integer = 1)
    split = variance_decomposition(filter; horizon_bars = horizon_bars)
    return predict_return(filter; horizon_bars = horizon_bars),
        split.state + split.parameter
end

"""
    regime_confidence(filter)

How sure the filter is, as one minus the normalised entropy of its belief.

One when it knows the state, zero when the data has not distinguished them.
"""
regime_confidence(filter::RegimeFilter) = 1 - normalised_entropy(regime_belief(filter))

"""
    most_likely_regime(filter)

The state carrying the most mass. A summary for a log, never an input to a decision.
"""
most_likely_regime(filter::RegimeFilter) = REGIME_STATES[argmax(filter.belief)]

state(filter::RegimeFilter) = Dict{String, Any}(
    "belief" => copy(filter.belief),
    "log_likelihood" => filter.log_likelihood,
    "n_seen" => filter.n_seen,
    "n_skipped" => filter.n_skipped,
)

"""
    load_state!(filter, state)

Restore a belief produced by [`state`](@ref), checked before anything is assigned.
"""
function load_state!(filter::RegimeFilter, saved::AbstractDict)
    belief = copy(convert(Vector{Float64}, saved["belief"]))
    log_likelihood = Float64(saved["log_likelihood"])
    n_seen = Int(saved["n_seen"])
    n_skipped = Int(saved["n_skipped"])

    length(belief) == N_REGIMES || throw(
        ArgumentError(
            string("state describes ", length(belief), " states, expected ", N_REGIMES),
        ),
    )
    all(isfinite, belief) || throw(ArgumentError("state belief must be finite"))
    for value in belief
        value >= 0 || throw(ArgumentError("state belief must be non-negative"))
    end
    total = sum(belief)
    isapprox(total, 1.0; atol = 1.0e-9) ||
        throw(ArgumentError(string("state belief sums to ", total, ", not 1")))
    isfinite(log_likelihood) || throw(ArgumentError("state log_likelihood must be finite"))
    (n_seen >= 0 && n_skipped >= 0) ||
        throw(ArgumentError("state counters must be non-negative"))

    filter.belief = belief
    filter.log_likelihood = log_likelihood
    filter.n_seen = n_seen
    filter.n_skipped = n_skipped
    return filter
end

parameters(parameters::RegimeParameters) = Dict{String, Any}(
    "centre" => parameters.centre,
    "drift_spread" => parameters.drift_spread,
    "persistence" => parameters.persistence,
    "mean_variance" => parameters.mean_variance,
    "dispersion" => parameters.dispersion,
    "n_rows" => parameters.n_rows,
    "drifts" => copy(parameters.drifts),
    "variances" => copy(parameters.variances),
    "stationary" => copy(parameters.prior.stationary),
)

parameters(filter::RegimeFilter) = merge(
    parameters(filter.parameters),
    Dict{String, Any}(
        "belief" => copy(filter.belief),
        "confidence" => regime_confidence(filter),
    ),
)

Base.show(io::IO, filter::RegimeFilter) = @printf(
    io, "<RegimeFilter %s p=%.3f n=%d lambda=%.3f>",
    slug(most_likely_regime(filter)), maximum(filter.belief), filter.n_seen,
    filter.parameters.persistence
)
