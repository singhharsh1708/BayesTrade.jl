"""
Predictive distributions and the structured result every model returns.

The distributions themselves come from Distributions.jl rather than being reimplemented
here. `Normal`, `TDist`, `LogNormal` and `MixtureModel` already have correct `cdf`,
`quantile`, `logpdf` and moments, including the bisection inverse a mixture needs, and
reimplementing them would be a large surface of arithmetic nobody else has checked.

What this file adds is the part Distributions.jl has no opinion about: which model produced
a prediction, when it was valid, and how much of its spread is reducible.
"""

const DEFAULT_LEVEL = 0.95

"""
    student_t(location, scale, df)

Student-t with a location and scale, the posterior predictive of a conjugate linear model
under an unknown noise variance.

The heavier tails are not a technicality. They are what an unknown variance implies, and a
normal in their place would understate the probability of a large loss, which is the one
quantity this system must not understate.
"""
function student_t(location::Real, scale::Real, df::Real)
    scale > 0 || throw(ArgumentError("scale must be positive, got $scale"))
    df > 0 || throw(ArgumentError("df must be positive, got $df"))
    return location + scale * TDist(df)
end

"""
    ModelVersion

Identity of the exact model that produced a prediction.

Stored with every prediction so any historical trade can be traced back to the parameters
that caused it. `version` is manual and changes when the mathematics change; `params_hash`
captures a change in fitted values, which is a different thing.
"""
Base.@kwdef struct ModelVersion
    name::ModelName
    version::VersionNumber
    fitted_at::Union{DateTime, Nothing} = nothing
    params_hash::Union{String, Nothing} = nothing
    train_start::Union{DateTime, Nothing} = nothing
    train_end::Union{DateTime, Nothing} = nothing

    function ModelVersion(name, version, fitted_at, params_hash, train_start, train_end)
        if train_start !== nothing && train_end !== nothing && train_start > train_end
            throw(ArgumentError("train_start $train_start is after train_end $train_end"))
        end
        return new(name, version, fitted_at, params_hash, train_start, train_end)
    end
end

"""
    identifier(version)

Human-readable identity, as it appears in a log or a report.
"""
function identifier(version::ModelVersion)
    suffix = version.params_hash === nothing ? "" : "+" * first(version.params_hash, 8)
    return string(slug(version.name), '@', version.version, suffix)
end

"""
    CredibleInterval

A central credible interval from a posterior predictive distribution.

A Bayesian credible interval, not a frequentist confidence interval: a statement about
where the quantity lies given the data and the model, which is the statement a position
sizing rule actually needs.

A struct rather than a tuple so `value in interval` reads as itself and so the level cannot
be silently dropped when the bounds are passed around.
"""
struct CredibleInterval
    lower::Float64
    upper::Float64
    level::Float64

    function CredibleInterval(lower::Real, upper::Real, level::Real)
        lower <= upper ||
            throw(ArgumentError("lower ($lower) exceeds upper ($upper)"))
        0 < level < 1 || throw(ArgumentError("level must lie in (0, 1), got $level"))
        return new(Float64(lower), Float64(upper), Float64(level))
    end
end

"""
    credible_interval(distribution, level = $DEFAULT_LEVEL)

Central credible interval of a distribution.
"""
function credible_interval(distribution::UnivariateDistribution, level::Real = DEFAULT_LEVEL)
    0 < level < 1 || throw(ArgumentError("level must lie in (0, 1), got $level"))
    tail = (1 - level) / 2
    return CredibleInterval(
        quantile(distribution, tail), quantile(distribution, 1 - tail), level,
    )
end

"""
    width(interval)

Width of a credible interval, the usual one-number summary of how much a model admits it
does not know.
"""
width(interval::CredibleInterval) = interval.upper - interval.lower

"""
    in(value, interval)

Whether a credible interval covers `value`.
"""
Base.in(value::Real, interval::CredibleInterval) =
    interval.lower <= value <= interval.upper

Base.show(io::IO, interval::CredibleInterval) = @printf(
    io, "%.1f%% credible interval [%.6g, %.6g]",
    100 * interval.level, interval.lower, interval.upper
)

"""
    probability_above(distribution, threshold)

`P(X > threshold)`.
"""
probability_above(distribution::UnivariateDistribution, threshold::Real) =
    ccdf(distribution, threshold)

"""
    probability_below(distribution, threshold)

`P(X < threshold)`.
"""
probability_below(distribution::UnivariateDistribution, threshold::Real) =
    cdf(distribution, threshold)

"""
    probability_positive(distribution)

`P(X > 0)`, the number the decision engine reads.
"""
probability_positive(distribution::UnivariateDistribution) = probability_above(distribution, 0.0)

"""
    probability_loss_exceeds(distribution, magnitude)

`P(X < -magnitude)` for a non-negative loss magnitude.

Taking a magnitude rather than a signed threshold makes the sign convention impossible to
get wrong at the call site, where getting it wrong means sizing into the tail you were
trying to avoid.
"""
function probability_loss_exceeds(distribution::UnivariateDistribution, magnitude::Real)
    magnitude >= 0 ||
        throw(ArgumentError("loss magnitude must be non-negative, got $magnitude"))
    return probability_below(distribution, -magnitude)
end

"""
    LabelledCategorical{L}

A distribution over named outcomes, such as market regimes or sentiment classes.

Distributions.jl's `Categorical` is indexed by position, which is fine for sampling and
useless in a report. Carrying the labels means an explanation can say which state was
likely rather than which index.
"""
struct LabelledCategorical{L}
    labels::Vector{L}
    probabilities::Vector{Float64}

    function LabelledCategorical(labels::Vector{L}, probabilities::Vector{<:Real}) where {L}
        isempty(labels) && throw(ArgumentError("need at least one category"))
        length(labels) == length(probabilities) || throw(
            ArgumentError(
                "$(length(labels)) labels against $(length(probabilities)) probabilities",
            ),
        )
        length(unique(labels)) == length(labels) ||
            throw(ArgumentError("category labels must be unique, got $labels"))
        any(<(0), probabilities) &&
            throw(ArgumentError("probabilities must be non-negative"))
        total = sum(probabilities)
        isapprox(total, 1; atol = 1.0e-6) ||
            throw(ArgumentError("probabilities must sum to 1, got $total"))
        return new{L}(labels, Float64.(probabilities))
    end
end

"""
    LabelledCategorical(pairs...)

Build from `label => probability` pairs.
"""
LabelledCategorical(pairs::Pair...) =
    LabelledCategorical(collect(first.(pairs)), collect(Float64.(last.(pairs))))

"""
    probability_of(categorical, label)

Probability assigned to `label`, or zero if it is not a category.
"""
function probability_of(categorical::LabelledCategorical{L}, label) where {L}
    index = findfirst(==(label), categorical.labels)
    return index === nothing ? 0.0 : categorical.probabilities[index]
end

"""
    most_likely(categorical)

The single most probable label. For display; never for a decision on its own.
"""
most_likely(categorical::LabelledCategorical) =
    categorical.labels[argmax(categorical.probabilities)]

"""
    Distributions.entropy(categorical)

Shannon entropy in nats: the distribution's own admission of uncertainty.
"""
Distributions.entropy(categorical::LabelledCategorical) =
    -sum(p * log(p) for p in categorical.probabilities if p > 0; init = 0.0)

"""
    normalised_entropy(categorical)

Entropy scaled to `[0, 1]`.

Normalised rather than raw so it stays comparable across models with different numbers of
states, which a raw entropy would not be.
"""
function normalised_entropy(categorical::LabelledCategorical)
    n = length(categorical.labels)
    n < 2 && return 0.0
    return entropy(categorical) / log(n)
end

"""
    uniform_categorical(labels)

The maximum-entropy distribution over `labels`: the honest answer before any data.
"""
function uniform_categorical(labels::Vector{L}) where {L}
    isempty(labels) && throw(ArgumentError("need at least one label"))
    return LabelledCategorical(labels, fill(1 / length(labels), length(labels)))
end

"""
    normalise(weights)

Scale non-negative weights to sum to one.
"""
function normalise(weights::AbstractVector{<:Real})
    total = sum(weights)
    total > 0 || throw(ArgumentError("weights must sum to a positive value, got $total"))
    return collect(Float64.(weights ./ total))
end

"""
    ProbabilisticResult{D}

What every model returns: a distribution, and the provenance to reproduce it.

Parametric on the distribution type, so a Gaussian result and a mixture result are
different types with no runtime dispatch cost and no tagged union to keep in step.

`epistemic_variance` is carried alongside the distribution rather than inside it. The split
between reducible and irreducible uncertainty is a statement about *why* the distribution
is that wide, not a property of the distribution itself, and Distributions.jl types stay
plain as a result.
"""
struct ProbabilisticResult{D}
    model::ModelVersion
    symbol::String
    as_of::DateTime
    horizon_bars::Int
    distribution::D
    n_observations::Int
    epistemic_variance::Float64
    diagnostics::Dict{Symbol, Float64}

    function ProbabilisticResult(;
            model::ModelVersion,
            symbol::AbstractString,
            as_of::DateTime,
            horizon_bars::Integer,
            distribution::D,
            n_observations::Integer = 0,
            epistemic_variance::Real = 0.0,
            diagnostics::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
        ) where {D}
        isempty(symbol) && throw(ArgumentError("result needs a symbol"))
        horizon_bars > 0 ||
            throw(ArgumentError("horizon_bars must be positive, got $horizon_bars"))
        n_observations >= 0 || throw(ArgumentError("n_observations cannot be negative"))
        epistemic_variance >= 0 ||
            throw(ArgumentError("epistemic_variance cannot be negative"))
        return new{D}(
            model, String(symbol), as_of, Int(horizon_bars), distribution,
            Int(n_observations), Float64(epistemic_variance), diagnostics,
        )
    end
end

Distributions.mean(result::ProbabilisticResult) = mean(result.distribution)
Distributions.var(result::ProbabilisticResult) = var(result.distribution)
Distributions.std(result::ProbabilisticResult) = std(result.distribution)

probability_positive(result::ProbabilisticResult) = probability_positive(result.distribution)
probability_above(result::ProbabilisticResult, threshold::Real) =
    probability_above(result.distribution, threshold)
probability_loss_exceeds(result::ProbabilisticResult, magnitude::Real) =
    probability_loss_exceeds(result.distribution, magnitude)
credible_interval(result::ProbabilisticResult, level::Real = DEFAULT_LEVEL) =
    credible_interval(result.distribution, level)

"""
    epistemic_share(result)

Fraction of the predictive variance that more data could remove.

Near one means the model is mostly unsure of itself and evidence will help. Near zero means
the spread is market noise and no amount of data will narrow it. Those call for different
responses from a position sizer, which is the reason the split is tracked at all.
"""
function epistemic_share(result::ProbabilisticResult)
    total = var(result.distribution)
    (!isfinite(total) || total <= 0) && return 0.0
    return clamp(result.epistemic_variance / total, 0.0, 1.0)
end
