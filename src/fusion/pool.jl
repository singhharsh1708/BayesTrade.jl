"""
Combining several models' predictives into one.

Two rules are on offer and they are not equivalent. A **logarithmic** pool multiplies the
densities, which is what Bayes' rule gives for *independent* evidence and which produces a
predictive sharper than any of its inputs. A **linear** pool averages them, and is at least as
wide as the weighted average of its inputs, exceeding it by exactly the amount they disagree.

This uses the linear pool, deliberately.

The three models are not independent. They read the same bars, and often the same feature: the
return model, the volatility model and the regime model all ultimately look at the same series
of closes. Multiplying their densities would count that shared evidence three times and produce
a confident answer built from one observation wearing three hats. In a system whose entire
purpose is to size positions by uncertainty, a spuriously narrow predictive is the most
expensive possible error.

The linear pool is the conservative reading: it says the truth is one of these models' stories
and we are unsure which. That extra width is model uncertainty, and it is real.
"""

"""
    OpinionPool(components, weights)

A weighted mixture of predictives, over a tuple so the component types stay concrete.

`MixtureModel` would do this, but only for components that all share one type. These do not:
the return model emits a Student-t, the volatility and regime models emit mixtures of them.
"""
struct OpinionPool{C <: Tuple} <: ContinuousUnivariateDistribution
    components::C
    weights::Vector{Float64}

    function OpinionPool(components::C, weights::AbstractVector{<:Real}) where {C <: Tuple}
        isempty(components) && throw(ArgumentError("a pool needs at least one component"))
        masses = convert(Vector{Float64}, collect(weights))
        length(masses) == length(components) || throw(
            ArgumentError(
                string(
                    length(components), " components against ", length(masses), " weights",
                ),
            ),
        )
        for mass in masses
            (isfinite(mass) && mass >= 0) ||
                throw(ArgumentError(string("weights must be finite and non-negative")))
        end
        total = sum(masses)
        total > 0 || throw(ArgumentError("weights must not all be zero"))
        isapprox(total, 1.0; atol = 1.0e-9) ||
            throw(ArgumentError(string("weights sum to ", total, ", not 1")))
        return new{C}(components, masses)
    end
end

n_components(pool::OpinionPool) = length(pool.components)

Base.minimum(::OpinionPool) = -Inf
Base.maximum(::OpinionPool) = Inf
Distributions.insupport(::OpinionPool, x::Real) = isfinite(x)

function Distributions.mean(pool::OpinionPool)
    total = 0.0
    for (index, component) in enumerate(pool.components)
        # Zero weights are skipped rather than multiplied. A component with an infinite
        # mean, which a Student-t below one degree of freedom has, would otherwise turn a
        # weight of zero into a NaN and poison a pool that does not use it at all.
        iszero(pool.weights[index]) && continue
        total += pool.weights[index] * mean(component)
    end
    return total
end

"""
    var(pool)

Total variance by the law of total variance: the average spread of the components plus the
spread of their centres.

The second term is what the pool adds over any single model, and it is exactly the models
disagreeing with each other.
"""
function Distributions.var(pool::OpinionPool)
    centre = mean(pool)
    total = 0.0
    for (index, component) in enumerate(pool.components)
        iszero(pool.weights[index]) && continue
        total += pool.weights[index] * (var(component) + (mean(component) - centre)^2)
    end
    return total
end

"""
    disagreement(pool)

The share of the pool's variance that comes from the models disagreeing rather than from any
one of them being unsure.
"""
function disagreement(pool::OpinionPool)
    centre = mean(pool)
    between = 0.0
    for (index, component) in enumerate(pool.components)
        iszero(pool.weights[index]) && continue
        between += pool.weights[index] * (mean(component) - centre)^2
    end
    return between
end

function Distributions.pdf(pool::OpinionPool, x::Real)
    total = 0.0
    for (index, component) in enumerate(pool.components)
        iszero(pool.weights[index]) && continue
        total += pool.weights[index] * pdf(component, Float64(x))
    end
    return total
end

function Distributions.logpdf(pool::OpinionPool, x::Real)
    value = Float64(x)
    largest = -Inf
    terms = Vector{Float64}(undef, n_components(pool))
    for (index, component) in enumerate(pool.components)
        weight = pool.weights[index]
        term = weight > 0 ? log(weight) + logpdf(component, value) : -Inf
        terms[index] = term
        term > largest && (largest = term)
    end
    isfinite(largest) || return -Inf
    total = 0.0
    for term in terms
        total += exp(term - largest)
    end
    return largest + log(total)
end

function Distributions.cdf(pool::OpinionPool, x::Real)
    total = 0.0
    for (index, component) in enumerate(pool.components)
        iszero(pool.weights[index]) && continue
        total += pool.weights[index] * cdf(component, Float64(x))
    end
    return total
end

Distributions.ccdf(pool::OpinionPool, x::Real) = 1 - cdf(pool, x)

"""
    quantile(pool, p)

Inverted by bisection, bracketed by the components' own quantiles.

A weighted average of monotone functions is monotone, and the pool's quantile lies between the
smallest and largest component quantile at the same level, so the bracket is exact rather than
a guess that has to be widened.
"""
function Distributions.quantile(pool::OpinionPool, p::Real)
    level = Float64(p)
    0 < level < 1 ||
        throw(ArgumentError(string("quantile level must lie in (0, 1), got ", p)))

    lower = Inf
    upper = -Inf
    for component in pool.components
        edge = quantile(component, level)
        lower = min(lower, edge)
        upper = max(upper, edge)
    end
    lower == upper && return lower

    for _ in 1:200
        middle = (lower + upper) / 2
        (middle == lower || middle == upper) && break
        cdf(pool, middle) < level ? (lower = middle) : (upper = middle)
    end
    return (lower + upper) / 2
end

function Base.rand(rng::AbstractRNG, pool::OpinionPool)
    threshold = rand(rng)
    cumulative = 0.0
    for (index, component) in enumerate(pool.components)
        cumulative += pool.weights[index]
        cumulative >= threshold && return rand(rng, component)
    end
    return rand(rng, last(pool.components))
end
