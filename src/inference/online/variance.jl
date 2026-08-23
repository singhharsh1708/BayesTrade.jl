"""
A conjugate variance filter that does not know how fast to forget, and says so.

Volatility moves, so a variance estimate has to decay old evidence. How fast is the whole
question, and it is not answerable in advance: the right memory length in a quiet market is
not the right one in a violent one, and nothing announces which market today is. The usual
answer is to pick a decay rate by hand and live with it. This keeps a posterior over it.

The mathematics. With the return centre `mu` known, `z = (r - mu)^2` and

    r | sigma^2 ~ Normal(mu, sigma^2)   =>   p(r | sigma^2) ~ (sigma^2)^(-1/2) exp(-z / 2 sigma^2)

which is an inverse-gamma kernel with shape increment `1/2` and rate increment `z/2`. So a
single component with discount `delta` keeps two discounted statistics and derives the rest:

    S <- delta S + w              Q <- delta Q + w z
    a  = a0 + S / 2               b  = b0 + Q / 2

The prior sits outside the discounted statistics, exactly as it does for the linear model, so
forgetting discounts the data and not the belief that preceded it. Two structural guarantees
follow, and both are load-bearing:

    a >= a0 > 1   =>   df = 2a > 2   =>   the predictive always has a finite mean and variance
    b >= b0 > 0   =>   the rate floor is unreachable rather than merely defensive

The second is why a flat bar is harmless here. A zero return is not a degenerate case to be
guarded against; it is an ordinary and rather informative observation, and no logarithm is
ever taken of it.

Several such components run side by side over a grid of discounts, weighted by their own
one-step predictive record:

    log wt_k <- alpha log w_k + L_k        w = softmax(log wt)

`L_k` is the log evidence of the bar just seen under component `k`. Every term of the marginal
that does not depend on `k` cancels in the softmax, so they are dropped, which is exact rather
than an approximation and removes the only term that could have been infinite at `z = 0`.

The grid is spaced evenly in memory length `1 / (1 - delta)` rather than in `delta`, because
memory length is the quantity anyone reasons in. The last component never forgets, which makes
the mixture's own answer to "is this market changing at all" readable off its weights.

What the filter emits is a distribution over the next *return*, not over volatility. That is
deliberate: it is what the calibration machinery scores, and a model whose output cannot be
falsified against a realised number is not a model. The volatility posterior is available
beside it for anyone who wants to look.
"""

"""
    VARIANCE_MIN_RATE

Floor on a component's rate.

Defensive only. `b = b0 + Q/2` with `b0 > 0` and `Q >= 0` cannot reach it, and a test asserts
that it does not, even after hundreds of exactly-zero returns.
"""
const VARIANCE_MIN_RATE = 1.0e-18

"""
    MIN_LOG_WEIGHT

Floor on a component's log weight, at `log(1e-20)`.

Weights are kept in log space and floored so a component that has been wrong for a thousand
bars can still come back. In linear space it would underflow to zero, and `alpha * log(0)` is
`-Inf`, which kills the component permanently: the model silently stops being adaptive and
carries on producing plausible numbers with no error and no symptom.
"""
const MIN_LOG_WEIGHT = -46.0

"""
    DEFAULT_DISCOUNTS

Memory lengths of 10, 20, 40, 100 bars and forever.

A tuple rather than a vector, so the default cannot be mutated by whoever touches it first.
"""
const DEFAULT_DISCOUNTS = (0.9, 0.95, 0.975, 0.99, 1.0)

"""
    InverseGammaPrior

Conjugate prior for a variance.

`shape > 1` is required rather than encouraged: at or below one the prior mean does not exist,
and since the shape only ever grows, requiring it here is what makes a finite predictive mean
a property of the type instead of something to check at every call site.
"""
struct InverseGammaPrior
    shape::Float64
    rate::Float64

    function InverseGammaPrior(shape::Real, rate::Real)
        shape > 1 ||
            throw(ArgumentError(string("prior shape must exceed 1, got ", shape)))
        # Above the rate floor, not merely above zero. Every posterior rate is at least
        # this one, so requiring it here is what makes the floor genuinely unreachable
        # instead of unreachable-for-sensible-priors.
        rate > VARIANCE_MIN_RATE || throw(
            ArgumentError(
                string("prior rate must exceed ", VARIANCE_MIN_RATE, ", got ", rate),
            ),
        )
        return new(Float64(shape), Float64(rate))
    end
end

"""
    expected_noise_variance(prior)

`E[sigma^2]` under the prior.
"""
expected_noise_variance(prior::InverseGammaPrior) = prior.rate / (prior.shape - 1)

"""
    variance_prior(; volatility_scale, shape = 2.0)

A prior that expresses where volatility is expected to sit, per bar.

`shape` sets how firmly. Two is deliberately vague: the prior then carries the weight of about
two observations, which a hundred bars of data overwhelms, while still keeping the rate away
from zero when the market goes quiet.
"""
function variance_prior(; volatility_scale::Real, shape::Real = 2.0)
    volatility_scale > 0 || throw(
        ArgumentError(string("volatility_scale must be positive, got ", volatility_scale)),
    )
    return InverseGammaPrior(shape, volatility_scale^2 * (shape - 1))
end

"""
    DiscountedVarianceFilter

A grid of conjugate variance filters and a posterior over which of them to believe.

Each component holds a discounted equivalent-observation count and a discounted weighted sum
of squares. Shape and rate are derived on read rather than stored, so there is one
representation of the state and no way for a cached copy to drift from it.
"""
mutable struct DiscountedVarianceFilter
    prior::InverseGammaPrior
    discounts::Vector{Float64}
    weight_forgetting::Float64
    centre::Float64
    weights::Vector{Float64}
    squares::Vector{Float64}
    log_weights::Vector{Float64}
    evidence::Vector{Float64}
    n_seen::Int
    n_skipped::Int

    function DiscountedVarianceFilter(
            prior::InverseGammaPrior;
            discounts = DEFAULT_DISCOUNTS,
            weight_forgetting::Real = 0.98,
            centre::Real = 0.0,
        )
        grid = convert(Vector{Float64}, collect(discounts))
        isempty(grid) && throw(ArgumentError("need at least one discount"))
        for value in grid
            0 < value <= 1 || throw(
                ArgumentError(string("discounts must lie in (0, 1], got ", value)),
            )
        end
        for index in 2:length(grid)
            grid[index] > grid[index - 1] || throw(
                ArgumentError(
                    string("discounts must be strictly increasing, got ", grid),
                ),
            )
        end
        0 < weight_forgetting <= 1 || throw(
            ArgumentError(
                string("weight_forgetting must lie in (0, 1], got ", weight_forgetting),
            ),
        )
        isfinite(centre) ||
            throw(ArgumentError(string("centre must be finite, got ", centre)))

        size = length(grid)
        return new(
            prior, grid, Float64(weight_forgetting), Float64(centre),
            zeros(Float64, size), zeros(Float64, size),
            fill(-log(size), size), zeros(Float64, size), 0, 0,
        )
    end
end

n_components(filter::DiscountedVarianceFilter) = length(filter.discounts)

"""
    n_absorbed(filter)

Observations actually absorbed, regardless of how much they still count.
"""
n_absorbed(filter::DiscountedVarianceFilter) = filter.n_seen

"""
    n_skipped(filter)

Bars deliberately passed over, which decay the state without informing it.
"""
n_skipped(filter::DiscountedVarianceFilter) = filter.n_skipped

"""
    posterior_shape(filter, component)

Shape after absorbing everything seen so far.
"""
posterior_shape(filter::DiscountedVarianceFilter, component::Integer) =
    filter.prior.shape + filter.weights[component] / 2

"""
    posterior_rate(filter, component)

Rate after absorbing everything seen so far.
"""
posterior_rate(filter::DiscountedVarianceFilter, component::Integer) =
    max(filter.prior.rate + filter.squares[component] / 2, VARIANCE_MIN_RATE)

"""
    evolved_shape(filter, component)

Shape one bar ahead, after ageing but before any new evidence.

The filtered state answers "how volatile is it now"; this answers "how volatile will the next
bar be", and they are not the same question. Ageing pulls the answer very slightly toward the
prior, which is the correct direction for a belief that is about to be a bar out of date.
"""
evolved_shape(filter::DiscountedVarianceFilter, component::Integer) =
    filter.prior.shape + filter.discounts[component] * filter.weights[component] / 2

"""
    evolved_rate(filter, component)

Rate one bar ahead, after ageing but before any new evidence.
"""
evolved_rate(filter::DiscountedVarianceFilter, component::Integer) = max(
    filter.prior.rate + filter.discounts[component] * filter.squares[component] / 2,
    VARIANCE_MIN_RATE,
)

"""
    discount_grid(filter)

The discounts being weighed, shortest memory first.
"""
discount_grid(filter::DiscountedVarianceFilter) = copy(filter.discounts)

"""
    discount_weights(filter)

Posterior probability of each discount, summing to one.
"""
discount_weights(filter::DiscountedVarianceFilter) = exp.(filter.log_weights)

"""
    expected_discount(filter)

`E[delta]`, the posterior mean discount.

The robust summary of what the filter currently believes about how fast this market forgets.
The mode is noisier: neighbouring components make near-identical predictions, so which one
leads flips on very little.
"""
function expected_discount(filter::DiscountedVarianceFilter)
    total = 0.0
    for index in 1:n_components(filter)
        total += exp(filter.log_weights[index]) * filter.discounts[index]
    end
    return total
end

"""
    discount_entropy(filter)

Entropy of the discount posterior in nats, between zero and `log(n_components)`.

Near zero means the filter is sure which memory length this market has. Near the maximum means
the data has not distinguished them, which is the honest state early on.
"""
function discount_entropy(filter::DiscountedVarianceFilter)
    total = 0.0
    for value in filter.log_weights
        weight = exp(value)
        # Skipping only exact zeros, where the limit of `w log w` is zero. A `NaN` weight
        # must carry through rather than be stepped over: this is the diagnostic that
        # would reveal a corrupt posterior, and reporting zero here would report perfect
        # certainty at exactly the moment there is none.
        iszero(weight) || (total -= weight * value)
    end
    return total
end

"""
    effective_sample_size(filter)

Discounted observation count the posterior actually reflects, averaged over the grid.

With a discount below one this converges to `1 / (1 - delta)` rather than growing.
"""
function effective_sample_size(filter::DiscountedVarianceFilter)
    total = 0.0
    for index in 1:n_components(filter)
        total += exp(filter.log_weights[index]) * filter.weights[index]
    end
    return total
end

"""
    steady_state_shape(prior_shape, discount)

Shape a component converges to under unit-weight observations, `Inf` when it never forgets.
"""
steady_state_shape(prior_shape::Real, discount::Real) =
    discount >= 1 ? Inf : prior_shape + 0.5 / (1 - discount)

"""
    noise_variance(filter)

`E[sigma^2]`, the posterior mean variance of one bar.
"""
function noise_variance(filter::DiscountedVarianceFilter)
    total = 0.0
    for index in 1:n_components(filter)
        shape = posterior_shape(filter, index)
        total += exp(filter.log_weights[index]) * posterior_rate(filter, index) /
            (shape - 1)
    end
    return total
end

"""
    residual_scale(filter)

`sqrt(E[sigma^2])`.

Not `E[sigma]`, and the difference is not cosmetic. See [`expected_volatility`](@ref).
"""
residual_scale(filter::DiscountedVarianceFilter) = sqrt(noise_variance(filter))

"""
    gamma_half_ratio(shape)

`gamma(shape - 1/2) / gamma(shape)`, which is `E[sigma]` divided by the root of the rate.

Switched to the asymptotic series once the shape is large, because the direct form takes the
difference of two log-gammas whose magnitudes dwarf the answer. At a shape of `1e13` the
direct form is already three per cent out, and at `1e17` it returns exactly `1` for a quantity
whose true value is near `3e-9`. The series is good to better than `1e-24` wherever it is
used, and the crossover sits where both forms are still accurate to a part in `1e7`.

A shape that large needs either twenty billion bars or a prior that asks for one, and the
prior is whatever the caller passes.
"""
function gamma_half_ratio(shape::Float64)
    shape >= 1.0e8 || return exp(loggamma(shape - 0.5) - loggamma(shape))
    return (1 + 3 / (8 * shape) + 25 / (128 * shape^2)) / sqrt(shape)
end

"""
    expected_volatility(filter)

`E[sigma]`, the posterior mean volatility of one bar.

Strictly below [`residual_scale`](@ref) by Jensen's inequality, and reported separately for
that reason: a position sizer that squares an expected volatility to get a variance
systematically under-reserves, by more the less certain the filter is.
"""
function expected_volatility(filter::DiscountedVarianceFilter)
    total = 0.0
    for index in 1:n_components(filter)
        total += exp(filter.log_weights[index]) *
            sqrt(posterior_rate(filter, index)) *
            gamma_half_ratio(posterior_shape(filter, index))
    end
    return total
end

"""
    expected_log_volatility(filter)

`E[log sigma]`, the scale on which volatility is roughly symmetric.
"""
function expected_log_volatility(filter::DiscountedVarianceFilter)
    total = 0.0
    for index in 1:n_components(filter)
        total += exp(filter.log_weights[index]) * component_log_volatility(filter, index)
    end
    return total
end

component_log_volatility(filter::DiscountedVarianceFilter, component::Integer) =
    0.5 * (log(posterior_rate(filter, component)) - digamma(posterior_shape(filter, component)))

"""
    discount_disagreement(filter)

How much the grid disagrees about the current volatility, on the log scale.

Pure between-component variance: it ignores each component's own uncertainty and measures only
the spread of their answers. It spikes at a regime break, when the short-memory components have
already moved and the long-memory ones have not, which makes it the closest thing this filter
has to an early warning.
"""
function discount_disagreement(filter::DiscountedVarianceFilter)
    centre = expected_log_volatility(filter)
    total = 0.0
    for index in 1:n_components(filter)
        deviation = component_log_volatility(filter, index) - centre
        total += exp(filter.log_weights[index]) * deviation^2
    end
    return total
end

"""
    volatility_uncertainty(filter)

Posterior standard deviation of `log sigma`, within components and between them.

The scalar the fusion layer reads. On the log scale so it is a relative uncertainty, which is
comparable across a quiet market and a violent one.
"""
function volatility_uncertainty(filter::DiscountedVarianceFilter)
    centre = expected_log_volatility(filter)
    total = 0.0
    for index in 1:n_components(filter)
        within = 0.25 * trigamma(posterior_shape(filter, index))
        between = (component_log_volatility(filter, index) - centre)^2
        total += exp(filter.log_weights[index]) * (within + between)
    end
    return sqrt(total)
end

"""
    plugin_variance(filter; horizon_bars = 1)

The share of predictive variance that would remain if the variance were known.

Irreducible: it is the market, not the model's ignorance of it.
"""
function plugin_variance(filter::DiscountedVarianceFilter; horizon_bars::Integer = 1)
    horizon_bars >= 1 ||
        throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
    total = 0.0
    for index in 1:n_components(filter)
        total += exp(filter.log_weights[index]) * horizon_bars *
            evolved_rate(filter, index) / evolved_shape(filter, index)
    end
    return total
end

"""
    variance_inflation(filter; horizon_bars = 1)

The share of predictive variance that comes from not knowing the variance.

Per component this is exactly `1 / a` of the total, which is `2 / df`: the same identity the
linear model's leverage term satisfies. It does not fall to zero with more data. Under a
discount it plateaus, which is correct and worth saying plainly, because you never do become
certain about a moving target.
"""
function variance_inflation(filter::DiscountedVarianceFilter; horizon_bars::Integer = 1)
    horizon_bars >= 1 ||
        throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
    total = 0.0
    for index in 1:n_components(filter)
        shape = evolved_shape(filter, index)
        total += exp(filter.log_weights[index]) * horizon_bars *
            evolved_rate(filter, index) / (shape * (shape - 1))
    end
    return total
end

"""
    log_variance_evidence(shape, rate, estimate, weight)

Log evidence of one variance estimate under an inverse-gamma belief, up to a constant.

Every term of the marginal that does not depend on the component is dropped, since the weights
are a softmax over components and such terms cancel identically. That is exact, not an
approximation, and it is what makes this safe: the dropped `(w/2 - 1) log(z)` term is the only
one that could have been infinite, and it goes to `-Inf` at a flat bar. Keeping it would send
every component to the same infinity and the softmax to `NaN`, destroying the grid posterior
permanently on the first perfectly flat bar.
"""
log_variance_evidence(shape::Float64, rate::Float64, estimate::Float64, weight::Float64) =
    shape * log(rate) + loggamma(shape + weight / 2) - loggamma(shape) -
    (shape + weight / 2) * log(rate + weight * estimate / 2)

"""
    observe_variance!(filter, estimate; weight = 1.0)

Absorb one variance estimate carrying `weight` equivalent observations.

The kernel every other absorbing path goes through. `weight` exists so an estimator that
carries more information than a single squared return, such as one built from the bar's range,
can enter at its true information content rather than being counted as one observation.

The grid is reweighted on the one-step-ahead state, before the state absorbs the bar, so each
component is scored on a genuine forecast rather than on how well it fits data it has already
seen.
"""
function observe_variance!(
        filter::DiscountedVarianceFilter, estimate::Real; weight::Real = 1.0,
    )
    value = Float64(estimate)
    mass = Float64(weight)
    isfinite(value) && value >= 0 || throw(
        ArgumentError(
            string("variance estimate must be finite and non-negative, got ", estimate),
        ),
    )
    isfinite(mass) && mass > 0 ||
        throw(ArgumentError(string("weight must be finite and positive, got ", weight)))
    # Both arguments can be finite while the observation is still unabsorbable: the
    # product can overflow, and so can the accumulated sum of squares after several large
    # bars, neither of which any single-argument check sees. An infinite rate sends every
    # component's evidence to `-Inf`, the softmax to `NaN`, and the grid posterior is then
    # destroyed for good, silently, with the error surfacing calls later somewhere inside
    # the mixture constructor.
    #
    # So the evidence is computed first and checked before anything is written. Nothing
    # here mutates until the whole step is known to be representable, which makes a
    # refused observation a bar that did not happen rather than a bar that half happened.
    size = n_components(filter)
    for index in 1:size
        filter.evidence[index] = log_variance_evidence(
            evolved_shape(filter, index), evolved_rate(filter, index), value, mass,
        )
    end

    for index in 1:size
        isfinite(filter.evidence[index]) && isfinite(
            filter.discounts[index] * filter.squares[index] + mass * value,
        ) || throw(
            ArgumentError(
                string(
                    "estimate ", estimate, " at weight ", weight,
                    " is too large to absorb without overflowing the statistics",
                ),
            ),
        )
    end

    for index in 1:size
        discount = filter.discounts[index]
        filter.weights[index] = discount * filter.weights[index] + mass
        filter.squares[index] = discount * filter.squares[index] + mass * value
    end

    reweight!(filter)
    filter.n_seen += 1
    return filter
end

"""
    skip_observation!(filter)

Age the state by one bar without learning anything from it.

The right response to a bar that cannot be trusted, and it is already the exact answer rather
than an approximation of one: with no evidence the statistics decay toward zero, so the
posterior returns to the prior. A skipped bar therefore widens the belief toward where it
started. It never drags the level toward the last observation, which is what a naive carry
forward would do and which is exactly wrong when the reason for skipping is that the feed
stopped.
"""
function skip_observation!(filter::DiscountedVarianceFilter)
    for index in 1:n_components(filter)
        discount = filter.discounts[index]
        filter.weights[index] *= discount
        filter.squares[index] *= discount
    end
    for index in 1:n_components(filter)
        filter.evidence[index] = 0.0
    end
    reweight!(filter)
    filter.n_skipped += 1
    return filter
end

"""
    reweight!(filter)

Fold the current evidence into the grid posterior, in log space.

`alpha` below one lets the answer to "which memory length" itself move over time, which it must
if the market's own memory length changes. The floor keeps a beaten component revivable.
"""
function reweight!(filter::DiscountedVarianceFilter)
    size = n_components(filter)
    largest = -Inf
    for index in 1:size
        value = filter.weight_forgetting * filter.log_weights[index] +
            filter.evidence[index]
        filter.log_weights[index] = value
        value > largest && (largest = value)
    end

    total = 0.0
    for index in 1:size
        total += exp(filter.log_weights[index] - largest)
    end
    offset = largest + log(total)

    for index in 1:size
        filter.log_weights[index] = max(filter.log_weights[index] - offset, MIN_LOG_WEIGHT)
    end

    total = 0.0
    for index in 1:size
        total += exp(filter.log_weights[index])
    end
    correction = log(total)
    for index in 1:size
        filter.log_weights[index] -= correction
    end
    return filter
end

"""
    update!(filter, observation)

Absorb one return.
"""
update!(filter::DiscountedVarianceFilter, observation::Real) =
    observe_variance!(filter, abs2(Float64(observation) - filter.centre); weight = 1.0)

"""
    fit!(filter, returns)

Absorb a series of returns, starting again from the prior.

One forward pass, in order. Unlike the linear model there is no batch shortcut worth having:
the grid posterior is prequential, so each bar has to be scored against the state that existed
before it, and that is a loop by definition.
"""
function fit!(filter::DiscountedVarianceFilter, returns::AbstractVector{<:Real})
    values = convert(Vector{Float64}, returns)
    all(isfinite, values) || throw(ArgumentError("returns must be finite"))
    reset!(filter)
    for value in values
        update!(filter, value)
    end
    return filter
end

"""
    fit!(filter, estimates, weights)

Absorb a series of variance estimates with their information contents.
"""
function fit!(
        filter::DiscountedVarianceFilter,
        estimates::AbstractVector{<:Real}, weights::AbstractVector{<:Real},
    )
    values = convert(Vector{Float64}, estimates)
    masses = convert(Vector{Float64}, weights)
    length(values) == length(masses) || throw(
        ArgumentError(
            string(
                "got ", length(values), " estimates and ", length(masses), " weights",
            ),
        ),
    )
    # Everything is checked before anything is absorbed. Validating inside the loop would
    # leave a rejected batch having already overwritten the filter with a prefix of itself,
    # which is neither the old state nor the prior and looks like neither.
    for index in eachindex(values)
        isfinite(values[index]) && values[index] >= 0 || throw(
            ArgumentError(
                string(
                    "estimate ", index, " must be finite and non-negative, got ",
                    values[index],
                ),
            ),
        )
        isfinite(masses[index]) && masses[index] > 0 || throw(
            ArgumentError(
                string(
                    "weight ", index, " must be finite and positive, got ", masses[index],
                ),
            ),
        )
        isfinite(masses[index] * values[index]) || throw(
            ArgumentError(
                string(
                    "estimate ", index, " at its weight is too large to absorb without ",
                    "overflowing the statistics",
                ),
            ),
        )
    end
    reset!(filter)
    for index in eachindex(values)
        observe_variance!(filter, values[index]; weight = masses[index])
    end
    return filter
end

"""
    reset!(filter)

Forget the data and return to the prior, keeping the prior and the grid.
"""
function reset!(filter::DiscountedVarianceFilter)
    size = n_components(filter)
    fill!(filter.weights, 0.0)
    fill!(filter.squares, 0.0)
    fill!(filter.log_weights, -log(size))
    fill!(filter.evidence, 0.0)
    filter.n_seen = 0
    filter.n_skipped = 0
    return filter
end

"""
    return_predictive(filter; horizon_bars = 1)

The predictive distribution of the return over the next `horizon_bars`.

A mixture of Student-t, one per discount. The location scales with the horizon and the scale
with its square root, while the degrees of freedom do not move at all: a longer horizon adds
noise, not evidence about how much noise there is.

The square-root rule assumes the variance holds over the horizon, which is exactly what this
filter says is false, so beyond one bar this is an approximation. It is measured rather than
argued about: a test scores both the one-bar and the ten-bar predictive against realised sums
on a stochastic-volatility path and requires both to be calibrated. Ten bars is not reliably
the worse of the two, because summing ten bars averages over the fluctuations the rule
ignores, so no ordering is claimed here.
"""
function return_predictive(filter::DiscountedVarianceFilter; horizon_bars::Integer = 1)
    horizon_bars >= 1 ||
        throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
    horizon = Float64(horizon_bars)
    components = [
        student_t(
                horizon * filter.centre,
                sqrt(horizon * evolved_rate(filter, index) / evolved_shape(filter, index)),
                2 * evolved_shape(filter, index),
            ) for index in 1:n_components(filter)
    ]
    return MixtureModel(components, discount_weights(filter))
end

"""
    variance_posterior(filter)

Posterior over the current variance of one bar, as a mixture of inverse-gammas.

The filtered state, not the evolved one: this is a statement about now, whereas
[`return_predictive`](@ref) is a statement about next.
"""
function variance_posterior(filter::DiscountedVarianceFilter)
    components = [
        InverseGamma(posterior_shape(filter, index), posterior_rate(filter, index))
            for index in 1:n_components(filter)
    ]
    return MixtureModel(components, discount_weights(filter))
end

"""
    predict_realised_variance(filter; horizon_bars = 1)

Predictive for the realised variance actually accumulated over the next `horizon_bars`, that
is for `sum (r - mu)^2`.

Exact rather than approximate, and worth the derivation. Given `sigma^2` the sum is
`sigma^2 chi^2_h`; writing `sigma^2 = b / G` with `G ~ Gamma(a, 1)` makes the ratio a beta
prime, so the marginal is `2 b BetaPrime(h/2, a)` with no moment matching anywhere.

Its mean must equal the variance of [`return_predictive`](@ref) at the same horizon, and it
does to a part in `1e-12`. Two families derived independently agreeing on the same quantity is
a real check on both.
"""
function predict_realised_variance(
        filter::DiscountedVarianceFilter; horizon_bars::Integer = 1,
    )
    horizon_bars >= 1 ||
        throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
    half = Float64(horizon_bars) / 2
    components = [
        2 * evolved_rate(filter, index) * BetaPrime(half, evolved_shape(filter, index))
            for index in 1:n_components(filter)
    ]
    return MixtureModel(components, discount_weights(filter))
end

"""
    volatility_interval(filter, level = $DEFAULT_LEVEL)

Credible interval for the current volatility of one bar.

The square root of the variance interval, which is exact rather than convenient: the square
root is monotone, so it carries quantiles across unchanged.
"""
function volatility_interval(
        filter::DiscountedVarianceFilter, level::Real = DEFAULT_LEVEL,
    )
    # The tail is a declared `Float64` rather than whatever arithmetic on an abstract
    # `Real` produces. `Real` is open, so the result of converting one is not knowable,
    # and both `Statistics` and `Distributions` own a `quantile`: with an unknown
    # probability the iterator method of the first genuinely applies to a concrete
    # mixture, and it would fail rather than return a quantile.
    posterior = variance_posterior(filter)
    tail::Float64 = (1 - level) / 2
    return CredibleInterval(
        sqrt(quantile(posterior, tail)), sqrt(quantile(posterior, 1 - tail)), level,
    )
end

"""
    predict(filter; horizon_bars = 1)

The return predictive and its epistemic share, as a `(distribution, epistemic_variance)` pair.

The split is an identity rather than a definition: the plug-in part and the inflation add to
the predictive variance exactly, because every component shares the same location and the
between-component term of the law of total variance vanishes.
"""
function predict(filter::DiscountedVarianceFilter; horizon_bars::Integer = 1)
    distribution = return_predictive(filter; horizon_bars = horizon_bars)
    return distribution, variance_inflation(filter; horizon_bars = horizon_bars)
end

"""
    state(filter)

The sufficient statistics, enough to reconstruct the posterior exactly.

The prior and the grid are configuration, not state, and are excluded for the same reason the
linear model excludes its own: a file that carries both invites the two to disagree.
"""
state(filter::DiscountedVarianceFilter) = Dict{String, Any}(
    "weights" => copy(filter.weights),
    "squares" => copy(filter.squares),
    "log_weights" => copy(filter.log_weights),
    "n_seen" => filter.n_seen,
    "n_skipped" => filter.n_skipped,
)

"""
    load_state!(filter, state)

Restore statistics produced by [`state`](@ref).

Everything is checked into locals before anything is assigned, so a rejected state leaves the
filter exactly as it was rather than half-restored.
"""
function load_state!(filter::DiscountedVarianceFilter, saved::AbstractDict)
    # Copied, not just converted. `convert` returns its argument unchanged when it is
    # already a `Vector{Float64}`, so assigning the result would leave the filter sharing
    # memory with the caller's dictionary, and a later write to that dictionary would
    # silently rewrite the posterior.
    size = n_components(filter)
    weights = copy(convert(Vector{Float64}, saved["weights"]))
    squares = copy(convert(Vector{Float64}, saved["squares"]))
    log_weights = copy(convert(Vector{Float64}, saved["log_weights"]))
    n_seen = Int(saved["n_seen"])
    n_skipped = Int(saved["n_skipped"])

    for (name, vector) in (
            ("weights", weights), ("squares", squares), ("log_weights", log_weights),
        )
        length(vector) == size || throw(
            ArgumentError(
                string(
                    "state describes ", length(vector), " ", name, " for a ", size,
                    "-component filter",
                ),
            ),
        )
        all(isfinite, vector) ||
            throw(ArgumentError(string("state ", name, " must be finite")))
    end
    for value in weights
        value >= 0 || throw(ArgumentError(string("state weights must be non-negative")))
    end
    for value in squares
        value >= 0 || throw(ArgumentError(string("state squares must be non-negative")))
    end

    total = 0.0
    for value in log_weights
        total += exp(value)
    end
    isapprox(total, 1.0; atol = 1.0e-9) || throw(
        ArgumentError(string("state log_weights exponentiate to ", total, ", not 1")),
    )
    (n_seen >= 0 && n_skipped >= 0) ||
        throw(ArgumentError("state counters must be non-negative"))

    filter.weights = weights
    filter.squares = squares
    filter.log_weights = log_weights
    filter.n_seen = n_seen
    filter.n_skipped = n_skipped
    return filter
end

"""
    parameters(filter)

Fitted parameters, in a form that hashes stably and reads in a log.
"""
function parameters(filter::DiscountedVarianceFilter)
    size = n_components(filter)
    return Dict{String, Any}(
        "discounts" => copy(filter.discounts),
        "discount_weights" => discount_weights(filter),
        "weight_forgetting" => filter.weight_forgetting,
        "centre" => filter.centre,
        "prior_shape" => filter.prior.shape,
        "prior_rate" => filter.prior.rate,
        "shapes" => Float64[posterior_shape(filter, index) for index in 1:size],
        "rates" => Float64[posterior_rate(filter, index) for index in 1:size],
        "expected_discount" => expected_discount(filter),
        "expected_volatility" => expected_volatility(filter),
        "effective_sample_size" => effective_sample_size(filter),
    )
end

Base.show(io::IO, filter::DiscountedVarianceFilter) = @printf(
    io, "<DiscountedVarianceFilter k=%d n=%d volatility=%.5g E[delta]=%.4f>",
    n_components(filter), filter.n_seen, expected_volatility(filter),
    expected_discount(filter)
)
