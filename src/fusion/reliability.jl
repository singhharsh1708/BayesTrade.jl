"""
How much to believe each model, learned from its own record.

The weights are prequential: every model is scored on a bar before that bar is used to update
anything, so a weight is always earned on a forecast rather than on a fit. The recursion is the
one the variance filter already uses for its discount grid:

    log w_k <- alpha log w_k + logpdf_k(outcome)

`alpha` below one lets the answer move. A model that was right for a year and is wrong now
should lose its weight, and under `alpha = 1` the accumulated evidence would keep it in front
long after it stopped deserving to be.
"""

const MIN_RELIABILITY_LOG_WEIGHT = -46.0

"""
    ModelReliability(names; forgetting = 0.99)

Weights over the named models, starting equal.
"""
mutable struct ModelReliability
    names::Vector{ModelName}
    log_weights::Vector{Float64}
    forgetting::Float64
    scores::Vector{Float64}
    n_scored::Int

    function ModelReliability(
            names::AbstractVector{ModelName}; forgetting::Real = 0.99,
        )
        labels = convert(Vector{ModelName}, names)
        isempty(labels) && throw(ArgumentError("need at least one model"))
        length(unique(labels)) == length(labels) ||
            throw(ArgumentError(string("duplicate model names: ", labels)))
        0 < forgetting <= 1 || throw(
            ArgumentError(string("forgetting must lie in (0, 1], got ", forgetting)),
        )
        size = length(labels)
        return new(
            labels, fill(-log(size), size), Float64(forgetting), zeros(Float64, size), 0,
        )
    end
end

n_models(reliability::ModelReliability) = length(reliability.names)

"""
    reliabilities(reliability)

The current weights, summing to one, in the order the models were named.
"""
reliabilities(reliability::ModelReliability) = exp.(reliability.log_weights)

"""
    reliability_belief(reliability)

The weights as a [`LabelledCategorical`](@ref), carrying which model is which.
"""
reliability_belief(reliability::ModelReliability) =
    LabelledCategorical(copy(reliability.names), reliabilities(reliability))

"""
    mean_log_scores(reliability)

Each model's average log score over everything it has been scored on.

Reported beside the weights because they answer different questions: the weight says what the
fusion layer currently believes, discounted; this says how the model has actually done.
"""
mean_log_scores(reliability::ModelReliability) =
    reliability.n_scored == 0 ? zeros(Float64, n_models(reliability)) :
    reliability.scores ./ reliability.n_scored

"""
    score!(reliability, densities)

Fold one bar's log densities into the weights.

`densities` is one log density per model, in the order the models were named, for the outcome
that has just been realised. A model that could not predict this bar contributes `-Inf`, which
costs it weight without ever removing it: the floor keeps a beaten model able to come back, and
in linear space it would underflow to zero and be dead for good.
"""
function score!(reliability::ModelReliability, densities::AbstractVector{<:Real})
    values = convert(Vector{Float64}, densities)
    length(values) == n_models(reliability) || throw(
        ArgumentError(
            string(
                "got ", length(values), " densities for ", n_models(reliability), " models",
            ),
        ),
    )
    for value in values
        isnan(value) && throw(ArgumentError("a log density must not be NaN"))
    end

    size = n_models(reliability)
    updated = Vector{Float64}(undef, size)
    largest = -Inf
    for index in 1:size
        value = reliability.forgetting * reliability.log_weights[index] + values[index]
        updated[index] = value
        value > largest && (largest = value)
    end
    isfinite(largest) ||
        throw(ArgumentError("no model could account for the outcome"))

    total = 0.0
    for index in 1:size
        total += exp(updated[index] - largest)
    end
    offset = largest + log(total)
    for index in 1:size
        reliability.log_weights[index] = max(
            updated[index] - offset, MIN_RELIABILITY_LOG_WEIGHT,
        )
    end

    correction = log(sum(exp, reliability.log_weights))
    for index in 1:size
        reliability.log_weights[index] -= correction
        isfinite(values[index]) && (reliability.scores[index] += values[index])
    end
    reliability.n_scored += 1
    return reliability
end

"""
    reset!(reliability)

Return every model to an equal share.
"""
function reset!(reliability::ModelReliability)
    size = n_models(reliability)
    fill!(reliability.log_weights, -log(size))
    fill!(reliability.scores, 0.0)
    reliability.n_scored = 0
    return reliability
end

parameters(reliability::ModelReliability) = Dict{String, Any}(
    "models" => String[slug(name) for name in reliability.names],
    "weights" => reliabilities(reliability),
    "forgetting" => reliability.forgetting,
    "n_scored" => reliability.n_scored,
    "mean_log_scores" => mean_log_scores(reliability),
)

Base.show(io::IO, reliability::ModelReliability) = @printf(
    io, "<ModelReliability n=%d scored=%d leader=%s>",
    n_models(reliability), reliability.n_scored,
    slug(reliability.names[argmax(reliability.log_weights)])
)
