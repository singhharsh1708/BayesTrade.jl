"""
The contract every probabilistic model implements.

Four operations, deliberately few:

`fit!`
    Batch estimation over history. May be expensive; runs offline.
`update!`
    One recursive step from a single observation. Must be closed-form and cheap, because
    this is what runs in the trading loop.
`predict`
    A full [`ProbabilisticResult`](@ref) for a point in time.
`uncertainty`
    A single number summarising how much the model currently doubts itself, so the fusion
    layer and the risk engine can reason about models they do not otherwise understand.

Models know about features and their own parameters. They do not know about portfolios,
orders, limits or brokers. A model that needs portfolio state to make a prediction is the
portfolio context model, and it receives that state as a feature like anything else.

Julia has no field inheritance, so fitted state is composed rather than inherited: each
model holds a [`FitState`](@ref) and exposes it through [`fit_state`](@ref). That is more
honest than inheritance would have been, since it makes the shared state a visible field
rather than something a subclass silently acquires.
"""

"""
    NotFittedError

Raised when a model is asked to predict before it has seen any data.

Returning a prior-only guess instead would be worse: it looks like a prediction and quietly
pollutes the record with beliefs the model never actually formed.
"""
struct NotFittedError <: Exception
    model::String
end

Base.showerror(io::IO, error::NotFittedError) = print(
    io, "NotFittedError: ", error.model,
    " has not been fitted; call fit! before predict",
)

"""
    FitState

Shared bookkeeping for a fitted model: whether it has been fitted, over what window, and
from how many observations.
"""
Base.@kwdef mutable struct FitState
    fitted::Bool = false
    n_observations::Int = 0
    fitted_at::Union{DateTime, Nothing} = nothing
    train_start::Union{DateTime, Nothing} = nothing
    train_end::Union{DateTime, Nothing} = nothing
end

"""
    ProbabilisticModel

Abstract supertype for every module in the fusion layer.

A concrete model must define [`fit_state`](@ref), [`model_name`](@ref),
[`model_semver`](@ref), [`parameters`](@ref), [`uncertainty`](@ref), and at least one of
`fit!` and `update!`.
"""
abstract type ProbabilisticModel end

"""
    fit_state(model)

The model's [`FitState`](@ref). Required of every concrete model.
"""
fit_state(model::ProbabilisticModel) = throw(
    ArgumentError("$(typeof(model)) must define fit_state"),
)

"""
    model_name(model)

Which of the six fusion modules this model is.
"""
model_name(model::ProbabilisticModel) = throw(
    ArgumentError("$(typeof(model)) must define model_name"),
)

"""
    model_semver(model)

The model's own version, changed by hand when its mathematics change.

Manual because the parameter hash captures a change in fitted *values*, not a change in what
those values mean.
"""
model_semver(model::ProbabilisticModel) = throw(
    ArgumentError("$(typeof(model)) must define model_semver"),
)

"""
    parameters(model)

Fitted parameters, in a form that hashes stably and reads in a log.
"""
parameters(model::ProbabilisticModel) = throw(
    ArgumentError("$(typeof(model)) must define parameters"),
)

"""
    uncertainty(model)

A scalar summary of current epistemic uncertainty, on the model's own scale.

Comparable across time for one model. Not comparable across models, which is exactly why
the fusion layer learns reliabilities instead of trusting this number as a weight.
"""
uncertainty(model::ProbabilisticModel) = throw(
    ArgumentError("$(typeof(model)) must define uncertainty"),
)

"""
    fit!(model, observations)

Estimate parameters from history. Implementations must call [`mark_fitted!`](@ref).
"""
fit!(model::ProbabilisticModel, observations) = throw(
    ArgumentError("$(typeof(model)) must define fit!"),
)

"""
    update!(model, observation)

Fold one new observation into the posterior.

Must be closed form. Anything requiring sampling belongs in `fit!` and the offline path.
"""
update!(model::ProbabilisticModel, observation) = throw(
    ArgumentError("$(typeof(model)) must define update!"),
)

"""
    predict(model, features; symbol, as_of, horizon_bars)

Return the predictive distribution and the provenance behind it.
"""
predict(model::ProbabilisticModel, features; kwargs...) = throw(
    ArgumentError("$(typeof(model)) must define predict"),
)

is_fitted(model::ProbabilisticModel) = fit_state(model).fitted
n_observations(model::ProbabilisticModel) = fit_state(model).n_observations

"""
    mark_fitted!(model; n_observations, fitted_at, train_start, train_end)

Record that a model has been fitted, and over what.
"""
function mark_fitted!(
        model::ProbabilisticModel;
        n_observations::Integer,
        fitted_at::Union{DateTime, Nothing} = nothing,
        train_start::Union{DateTime, Nothing} = nothing,
        train_end::Union{DateTime, Nothing} = nothing,
    )
    state = fit_state(model)
    state.fitted = true
    state.n_observations = Int(n_observations)
    state.fitted_at = fitted_at
    state.train_start = train_start
    state.train_end = train_end
    return model
end

"""
    reset!(model)

Discard fitted state and return to the prior.

Concrete models extend this to clear their own parameters, and should call the generic
method to clear the shared bookkeeping.
"""
function reset!(model::ProbabilisticModel)
    state = fit_state(model)
    state.fitted = false
    state.n_observations = 0
    state.fitted_at = nothing
    state.train_start = nothing
    state.train_end = nothing
    return model
end

"""
    require_fitted(model)

Throw [`NotFittedError`](@ref) unless the model has been fitted.
"""
function require_fitted(model::ProbabilisticModel)
    is_fitted(model) || throw(NotFittedError(string(nameof(typeof(model)))))
    return nothing
end

"""
    params_hash(model)

Content hash of the fitted parameters, or `nothing` when unfitted.

`hash` is salted per session and floats print inconsistently, neither of which is acceptable
for an identifier that has to mean the same thing next month, so the digest is taken over a
canonical rendering with floats rounded to twelve places. A parameter differing only by
floating-point noise must not present as a different model version.
"""
function params_hash(model::ProbabilisticModel)
    is_fitted(model) || return nothing
    return stable_hash(parameters(model))
end

"""
    PACKAGE_VERSION

The version this build of the package reports about itself, read from its own Project.toml.

Read once at load rather than hardcoded, because a constant that has to be edited alongside the
manifest is a constant that will disagree with it.
"""
const PACKAGE_VERSION = let
    project = joinpath(dirname(dirname(@__DIR__)), "Project.toml")
    version = v"0.0.0"
    if isfile(project)
        for line in eachline(project)
            if startswith(line, "version")
                parsed = tryparse(VersionNumber, strip(split(line, "=")[2], [' ', '"']))
                parsed === nothing || (version = parsed)
                break
            end
        end
    end
    version
end

"""
    stable_hash(payload)

Hex digest of `payload`, identical across sessions and machines.
"""
stable_hash(payload) = bytes2hex(sha256(canonical(payload)))[1:32]

canonical(value::AbstractFloat) =
    isnan(value) ? "nan" :
    isinf(value) ? (value > 0 ? "inf" : "-inf") :
    string(round(value, digits = 12) + 0.0)
canonical(value::Integer) = string(value)
canonical(value::Bool) = string(value)
canonical(value::AbstractString) = string("\"", String(value), "\"")
canonical(value::Symbol) = canonical(string(value))
canonical(::Nothing) = "null"
function canonical(value::AbstractDict)
    keys_sorted = sort(collect(keys(value)), by = string)
    entries = String[
        string(canonical(string(k)), ":", canonical(value[k])) for k in keys_sorted
    ]
    return string("{", join(entries, ","), "}")
end

# A comprehension rather than a broadcast: broadcasting a function over an abstractly typed
# container does not infer, and this runs on every model version stamp.
canonical(value::AbstractVector) =
    string("[", join(String[canonical(item) for item in value], ","), "]")
canonical(value::Tuple) = canonical(collect(value))
canonical(value) = canonical(string(value))

"""
    model_version(model)

Identity stamped onto every prediction this model produces.
"""
function model_version(model::ProbabilisticModel)
    state = fit_state(model)
    return ModelVersion(
        name = model_name(model),
        version = model_semver(model),
        fitted_at = state.fitted_at,
        params_hash = params_hash(model),
        train_start = state.train_start,
        train_end = state.train_end,
    )
end

function Base.show(io::IO, model::ProbabilisticModel)
    state = fit_state(model)
    status = state.fitted ? "fitted on $(state.n_observations) observations" : "unfitted"
    return print(
        io, '<', nameof(typeof(model)), ' ', slug(model_name(model)),
        '@', model_semver(model), " (", status, ")>",
    )
end
