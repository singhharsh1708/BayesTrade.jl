"""
Turning several models' answers into one.

The fusion layer combines posteriors and their reliabilities. It applies no limits and makes no
decisions: what comes out is still a distribution over the forward return, and what to do about
it belongs to the decision engine and, above that, to the risk engine.

A model that cannot answer this bar is simply absent, and the weights renormalise over those
that can. That is the honest treatment. Substituting a prior-only guess would let a model that
knows nothing about this bar drag the pool toward its prior while still looking like evidence.
"""

"""
    FusedPrediction

One predictive assembled from several, with the record of who contributed and how much.

Not a `ProbabilisticResult`, deliberately. That type carries one `ModelVersion`, and the whole
point here is that no single model is responsible for this number. Every contributing version
is kept so a trade months old can be traced back to the exact models and weights behind it.
"""
struct FusedPrediction{D <: UnivariateDistribution}
    symbol::String
    as_of::DateTime
    horizon_bars::Int
    distribution::D
    epistemic_variance::Float64
    weights::LabelledCategorical{ModelName}
    sources::Vector{ModelVersion}
    diagnostics::Dict{Symbol, Float64}

    function FusedPrediction(;
            symbol::AbstractString,
            as_of::DateTime,
            horizon_bars::Integer,
            distribution::D,
            epistemic_variance::Real,
            weights::LabelledCategorical{ModelName},
            sources::Vector{ModelVersion},
            diagnostics::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
        ) where {D <: UnivariateDistribution}
        isempty(symbol) && throw(ArgumentError("a prediction needs a symbol"))
        horizon_bars >= 1 ||
            throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
        isempty(sources) && throw(ArgumentError("a fused prediction needs a source"))
        length(sources) == length(weights.labels) || throw(
            ArgumentError(
                string(
                    length(sources), " sources against ", length(weights.labels),
                    " weights",
                ),
            ),
        )
        (isfinite(epistemic_variance) && epistemic_variance >= 0) ||
            throw(ArgumentError("epistemic_variance must be finite and non-negative"))
        return new{D}(
            String(symbol), as_of, Int(horizon_bars), distribution,
            Float64(epistemic_variance), weights, sources, diagnostics,
        )
    end
end

n_models(prediction::FusedPrediction) = length(prediction.sources)
Distributions.mean(prediction::FusedPrediction) = mean(prediction.distribution)
Distributions.var(prediction::FusedPrediction) = var(prediction.distribution)
Distributions.std(prediction::FusedPrediction) = std(prediction.distribution)
probability_positive(prediction::FusedPrediction) =
    probability_positive(prediction.distribution)
credible_interval(prediction::FusedPrediction, level::Real = DEFAULT_LEVEL) =
    credible_interval(prediction.distribution, level)

"""
    epistemic_share(prediction)

How much of the spread is reducible: the models' own epistemic uncertainty plus their
disagreement with each other.
"""
epistemic_share(prediction::FusedPrediction) =
    prediction.epistemic_variance / var(prediction.distribution)

"""
    fuse(reliability, results)

Pool the given results, weighting each model by what it has earned.

Every result must describe the same symbol, moment and horizon. Fusing predictions about
different things is not a combination, it is a mistake, and it is caught here rather than
producing a number nobody can interpret.
"""
function fuse(reliability::ModelReliability, results::Tuple)
    isempty(results) && throw(ArgumentError("nothing to fuse"))
    first_result = first(results)
    symbol = first_result.symbol
    as_of = first_result.as_of
    horizon = first_result.horizon_bars

    for result in results
        result.symbol == symbol || throw(
            ArgumentError(
                string("fusing ", symbol, " with ", result.symbol),
            ),
        )
        result.as_of == as_of ||
            throw(ArgumentError(string("fusing ", as_of, " with ", result.as_of)))
        result.horizon_bars == horizon || throw(
            ArgumentError(
                string(
                    "fusing a ", horizon, "-bar horizon with a ", result.horizon_bars,
                    "-bar one",
                ),
            ),
        )
    end

    names = ModelName[result.model.name for result in results]
    length(unique(names)) == length(names) ||
        throw(ArgumentError(string("two results from the same model: ", names)))

    masses = Vector{Float64}(undef, length(names))
    available = reliabilities(reliability)
    for (index, name) in enumerate(names)
        position = findfirst(==(name), reliability.names)
        position === nothing &&
            throw(ArgumentError(string("no reliability recorded for ", slug(name))))
        masses[index] = available[position]
    end
    total = sum(masses)
    # Renormalised over the models that answered, so an absent model costs the others nothing.
    # If every present model has been floored to nothing, fall back to equal shares rather
    # than dividing by zero: the pool then says it has no idea, which is true.
    if total <= 0
        fill!(masses, 1 / length(masses))
    else
        masses ./= total
    end

    distributions = map(result -> result.distribution, results)
    pool = OpinionPool(distributions, masses)

    within = 0.0
    for (index, result) in enumerate(results)
        within += masses[index] * result.epistemic_variance
    end
    between = disagreement(pool)

    return FusedPrediction(
        symbol = symbol,
        as_of = as_of,
        horizon_bars = horizon,
        distribution = pool,
        # The models disagreeing is reducible in principle: it is uncertainty about which
        # model is right, and evidence about that is exactly what the reliabilities are.
        epistemic_variance = within + between,
        weights = LabelledCategorical(names, masses),
        sources = ModelVersion[result.model for result in results],
        diagnostics = Dict{Symbol, Float64}(
            :n_models => Float64(length(results)),
            :disagreement => between,
            :within_model_epistemic => within,
            :weight_entropy => normalised_entropy(LabelledCategorical(names, masses)),
        ),
    )
end

"""
    score_fusion!(reliability, results, outcome)

Score each model on the outcome it forecast, then move the weights.

Called after the bar is realised and never before, so a weight is always earned on a forecast.
"""
function score_fusion!(
        reliability::ModelReliability, results::Tuple, outcome::Real,
    )
    realised = Float64(outcome)
    names = ModelName[result.model.name for result in results]
    length(unique(names)) == length(names) ||
        throw(ArgumentError(string("two results from the same model: ", names)))
    densities = fill(-Inf, n_models(reliability))
    for result in results
        position = findfirst(==(result.model.name), reliability.names)
        position === nothing &&
            throw(ArgumentError(string("no reliability recorded for ", slug(result.model.name))))
        densities[position] = logpdf(result.distribution, realised)
    end
    return score!(reliability, densities)
end

Base.show(io::IO, prediction::FusedPrediction) = @printf(
    io, "<FusedPrediction %s %s h=%d models=%d mean=%.5g sd=%.5g>",
    prediction.symbol, prediction.as_of, prediction.horizon_bars,
    n_models(prediction), mean(prediction), std(prediction)
)
