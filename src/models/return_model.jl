"""
The Bayesian return model.

A conjugate linear regression from point-in-time features to the forward log return, with the
pieces that turn a regression into something a trading system can use:

* the design row is built from a fixed, stored column order, so a model fitted on one ordering
  can never be scored on another
* features are standardised with statistics frozen at fitting time, never recomputed
* the horizon is part of the model's identity, so a model fitted on five-bar returns refuses
  to answer a question about twenty-bar returns
* every prediction carries the model version and the number of observations behind it

What it does not do is decide anything. It reports a distribution over the forward return and
stops there.
"""

const INTERCEPT = :intercept
const MIN_TRAINING_ROWS = 30

"""
    HorizonMismatchError

Raised when a model is asked about a horizon it was not fitted for.

Silently answering would be worse than failing: a five-bar model's predictive scale is roughly
half a twenty-bar model's, so the answer would look reasonable and be wrong by a factor of two
in exactly the quantity the risk engine reads.
"""
struct HorizonMismatchError <: Exception
    message::String
end

Base.showerror(io::IO, error::HorizonMismatchError) =
    print(io, "HorizonMismatchError: ", error.message)

"""
    ResponseScalePolicy

What units the response is measured in at a point in time.

A regression assumes its noise scale is constant. On returns that is false in a way that
matters: the same coefficients describe a market whose bars are five times wider in a crisis,
and a model fitted across both is fitted to neither. Dividing the response by a
point-in-time scale before fitting, and multiplying the predictive back afterwards, lets one
set of coefficients describe both regimes.

A policy rather than a flag, so the arithmetic that divides and multiplies is written once and
the choice of scale is a type. The model is parameterised on it, which is what lets the plain
model and a scaled one be different types with different versions rather than one type with a
field nobody can see in a log.
"""
abstract type ResponseScalePolicy end

"""
    ConstantScale()

Leave the response alone, and let the regression estimate one noise scale.

The default, and the honest choice when there is no volatility estimate to hand: a constant
that is wrong in both regimes is at least visible in the residual scale, whereas a bad
point-in-time scale is not.
"""
struct ConstantScale <: ResponseScalePolicy end

"""
    VolatilityScale(column; floor = 1.0e-4, annualised = true)

Measure the response in units of the volatility this bar is expected to have.

`column` names a point-in-time feature, never a volatility model handed in at prediction time.
That is structural rather than stylistic: a feature is built from bars that have closed, so a
scaled model cannot reach forward even by accident.

`floor` is not a fudge. A volatility feature is exactly zero on a flat window, and a scale of
zero is not a scale: it would divide a real return by nothing when fitting and collapse the
predictive to a point when predicting. The floor is what a flat window is worth, and it is
recorded in the model's parameters so a reader can see which one was used.
"""
struct VolatilityScale <: ResponseScalePolicy
    column::Symbol
    floor::Float64
    annualised::Bool

    function VolatilityScale(
            column::Symbol; floor::Real = 1.0e-4, annualised::Bool = true,
        )
        floor > 0 || throw(ArgumentError(string("floor must be positive, got ", floor)))
        return new(column, Float64(floor), annualised)
    end
end

"""
    policy_columns(policy)

Feature columns the policy reads, beyond the model's own.
"""
policy_columns(::ConstantScale) = Symbol[]
policy_columns(policy::VolatilityScale) = Symbol[policy.column]

"""
    default_residual_scale(policy)

Where the noise is expected to sit, in whatever units the policy leaves the response in.

A scaled response is a return divided by its own expected volatility, so it sits near one
rather than near a couple of per cent. Carrying the unscaled default across would put the
prior about sixty times too tight against a measured residual scale of 1.23, which a long
window overwhelms and a short one does not.
"""
default_residual_scale(::ConstantScale) = 0.02
default_residual_scale(::VolatilityScale) = 1.0

"""
    policy_parameters(policy)

The policy, in a form that hashes stably and reads in a log, or `nothing` when there is
nothing to record.

Part of `parameters`, so two models with identical regression state but different scaling do
not share a parameter hash. They do not make the same predictions and must not claim the same
identity.

An unscaled model records nothing at all, rather than recording that it is unscaled. The
difference matters: `parameters` feeds the hash a bundle is verified against, so adding a key
to every plain model would change every plain model's identity and every file written before
policies existed would stop loading against the model it describes.
"""
policy_parameters(::ConstantScale) = nothing
policy_parameters(policy::VolatilityScale) = Dict{String, Any}(
    "kind" => "volatility",
    "column" => string(policy.column),
    "floor" => policy.floor,
    "annualised" => policy.annualised,
)

"""
    BayesianReturnModel

Forward log return as a Student-t posterior predictive.

The intercept is fitted rather than assumed zero, and it is not standardised. On a return
model it absorbs the unconditional drift, so leaving it out would force the features to
explain a level they have nothing to do with.
"""
mutable struct BayesianReturnModel{P <: ResponseScalePolicy} <: ProbabilisticModel
    feature_names::Vector{Symbol}
    horizon_bars::Int
    regression::BayesianLinearModel
    scaler::Union{FeatureScaler, Nothing}
    policy::P
    state::FitState

    function BayesianReturnModel(
            feature_names::AbstractVector{Symbol};
            horizon_bars::Integer,
            residual_scale::Union{Real, Nothing} = nothing,
            coefficient_scale::Real = 0.5,
            prior_shape::Real = 2.0,
            forgetting::Real = 1.0,
            prior::Union{NormalInverseGammaPrior, Nothing} = nothing,
            policy::P = ConstantScale(),
        ) where {P <: ResponseScalePolicy}
        isempty(feature_names) &&
            throw(ArgumentError("a return model needs at least one feature"))
        tally = Dict{Symbol, Int}()
        for name in feature_names
            tally[name] = get(tally, name, 0) + 1
        end
        duplicates = sort!(Symbol[name for (name, seen) in tally if seen > 1])
        isempty(duplicates) || throw(
            ArgumentError(
                string("duplicate feature names: ", join(string.(duplicates), ", ")),
            ),
        )
        horizon_bars >= 1 ||
            throw(ArgumentError(string("horizon_bars must be at least 1, got ", horizon_bars)))

        width = length(feature_names) + 1
        if prior !== nothing && n_features(prior) != width
            throw(
                ArgumentError(
                    string(
                        "prior describes ", n_features(prior), " columns, model has ",
                        width, " including the intercept",
                    ),
                ),
            )
        end
        noise = residual_scale === nothing ? default_residual_scale(policy) : residual_scale
        resolved = prior === nothing ?
            weakly_informative_prior(
                width; residual_scale = noise,
                coefficient_scale = coefficient_scale, shape = prior_shape,
            ) : prior

        return new{P}(
            convert(Vector{Symbol}, feature_names), Int(horizon_bars),
            BayesianLinearModel(resolved; forgetting = forgetting), nothing, policy,
            FitState(),
        )
    end
end

fit_state(model::BayesianReturnModel) = model.state
model_name(::BayesianReturnModel) = MOMENTUM
# The version is the mathematics, not the fitted values, so a scaled model is a different
# version of the model rather than a different fit of it. Left at 0.1.0 for the plain one so
# every bundle already on disk still verifies.
model_semver(::BayesianReturnModel{ConstantScale}) = v"0.1.0"
model_semver(::BayesianReturnModel{VolatilityScale}) = v"0.2.0"

"""
    design_columns(model)

The design columns, intercept first. This order is part of the model.
"""
design_columns(model::BayesianReturnModel) = vcat(INTERCEPT, model.feature_names)

"""
    response_scale(model, features)

Units the response is measured in at this point in time.

Delegated to the model's policy, so the divide-then-multiply arithmetic around it is written
once. Under [`ConstantScale`](@ref) it is one and every path below reduces to the plain model.
"""
response_scale(model::BayesianReturnModel, features::FeatureVector) =
    response_scale(model.policy, features)

response_scale(::ConstantScale, ::FeatureVector) = 1.0

function response_scale(policy::VolatilityScale, features::FeatureVector)
    value = require(features, policy.column)
    isfinite(value) || throw(
        ArgumentError(string(policy.column, " is ", value, ", which is not a scale")),
    )
    scale = policy.annualised ? deannualise(value) : value
    return max(scale, policy.floor)
end

"""
    fit!(model, observations)

Estimate the scaler and the posterior from a training set.

The scaler is fitted on this window only and frozen. Recomputing it later would tell the model
something about the period it is being asked to predict.
"""
function fit!(model::BayesianReturnModel, observations::AbstractVector{TrainingExample})
    length(observations) >= MIN_TRAINING_ROWS || throw(
        ArgumentError(
            string(
                "need at least ", MIN_TRAINING_ROWS, " training rows, got ",
                length(observations),
            ),
        ),
    )
    check_horizons(model, observations)

    stamps = DateTime[example.features.as_of for example in observations]
    # Row i of n carries weight forgetting^(n - i), so the order is not presentation: rows
    # out of order are silently weighted as though they arrived when they did not.
    issorted(stamps) ||
        throw(ArgumentError("training rows must be in chronological order"))

    width = length(model.feature_names)
    raw = Matrix{Float64}(undef, length(observations), width)
    responses = Vector{Float64}(undef, length(observations))
    for (index, example) in enumerate(observations)
        row = design_row(example.features, model.feature_names)
        for column in 1:width
            raw[index, column] = row[column]
        end
        responses[index] = example.label.forward_log_return /
            response_scale(model, example.features)
    end

    # The scaler is built and the design validated before either is installed. Assigning
    # the scaler first would leave a window the regression goes on to refuse having already
    # replaced the standardisation the current posterior was fitted under, so a refused
    # refit would change what the model predicts.
    scaler = fit_scaler(model.feature_names, raw)
    design = build_design(model, raw, scaler)
    all(isfinite, responses) || throw(ArgumentError("responses must be finite"))

    model.scaler = scaler
    fit!(model.regression, design, responses)

    return mark_fitted!(
        model;
        n_observations = length(observations),
        fitted_at = maximum(stamps),
        train_start = minimum(stamps),
        train_end = maximum(stamps),
    )
end

"""
    update!(model, observation)

Fold one realised outcome into the posterior. Rank-one, no refit.
"""
function update!(model::BayesianReturnModel, observation::TrainingExample)
    require_fitted(model)
    check_horizons(model, [observation])
    require_later(model, observation.features.as_of)
    raw = design_row(observation.features, model.feature_names)
    update!(
        model.regression,
        build_row(model, raw),
        observation.label.forward_log_return / response_scale(model, observation.features),
    )
    model.state.n_observations += 1
    model.state.train_end = observation.features.as_of
    return model
end

"""
    require_later(model, as_of)

Refuse a bar the model has already moved past.

Absorbing one twice counts it twice, and absorbing an older one silently rewinds the training
window the model reports having been fitted over. Neither raises anything on its own, and both
leave a posterior that no sequence of bars could have produced.
"""
function require_later(model::ProbabilisticModel, as_of::DateTime)
    train_end = fit_state(model).train_end
    train_end === nothing && return nothing
    as_of > train_end || throw(
        ArgumentError(
            string(
                "model has already absorbed up to ", train_end, ", given a bar at ", as_of,
            ),
        ),
    )
    return nothing
end

"""
    predict(model, features; symbol, as_of, horizon_bars)

The predictive distribution over the forward log return.
"""
function predict(
        model::BayesianReturnModel, features::FeatureVector;
        symbol::AbstractString, as_of::DateTime, horizon_bars::Integer = 1,
    )
    require_fitted(model)
    horizon_bars == model.horizon_bars || throw(
        HorizonMismatchError(
            string(
                "model was fitted for ", model.horizon_bars, "-bar returns, asked for ",
                horizon_bars,
            ),
        ),
    )

    raw = design_row(features, model.feature_names)
    distribution, epistemic = predict(model.regression, build_row(model, raw))
    scale = response_scale(model, features)
    scaled = scale == 1.0 ? distribution : rescale(distribution, scale)

    return ProbabilisticResult(
        model = model_version(model),
        symbol = symbol,
        as_of = as_of,
        horizon_bars = model.horizon_bars,
        distribution = scaled,
        n_observations = n_observations(model),
        epistemic_variance = epistemic * scale^2,
        diagnostics = Dict{Symbol, Float64}(
            :residual_scale => residual_scale(model.regression),
            :effective_sample_size => effective_sample_size(model.regression),
            :response_scale => scale,
        ),
    )
end

"""
    rescale(distribution, factor)

Return a location-scale Student-t to the response's own units.
"""
rescale(distribution, factor::Real) =
    student_t(mean(distribution) * factor, distribution.σ * factor, dof(distribution.ρ))

"""
    can_predict(model, features)

Whether this vector has every column the model needs.

A warming-up vector is a normal state early in a backtest, so callers ask rather than catching
an exception in the loop.
"""
can_predict(model::BayesianReturnModel, features::FeatureVector) =
    is_fitted(model) &&
    all(name -> haskey(features, name), model.feature_names) &&
    all(name -> haskey(features, name), policy_columns(model.policy))

"""
    uncertainty(model)

Average coefficient uncertainty relative to the residual scale.

Dimensionless, so it stays comparable as the model is refitted on data of different
volatility. Falling means the coefficients are pinned down relative to the noise they are
trying to explain.
"""
function uncertainty(model::BayesianReturnModel)
    is_fitted(model) || return Inf
    spreads = coefficient_std(model.regression)
    return sqrt(mean(spreads .^ 2)) / residual_scale(model.regression)
end

function parameters(model::BayesianReturnModel)
    # The scaler is bound to a local before the nothing check. Narrowing a union field
    # inside an expression does not refine the field's type for the call that follows, so
    # the branch that cannot run is still analysed and still has to typecheck.
    scaler = model.scaler
    recorded = Dict{String, Any}(
        "columns" => String[string(name) for name in design_columns(model)],
        "horizon_bars" => model.horizon_bars,
        "regression" => parameters(model.regression),
        "scaler" => scaler === nothing ? nothing : parameters(scaler),
    )
    policy = policy_parameters(model.policy)
    policy === nothing || (recorded["policy"] = policy)
    return recorded
end

"""
    coefficient_report(model)

Per-column posterior summary, for reading a fitted model.

The standardised coefficient is the comparable one: it says how much the predicted return
moves per typical move in that feature. The raw one is what a reader checks against a
definition.

Under a scaling policy the raw coefficient is in units of the scaled response, not of the
return. There is no single number in return units to report, because the scale moves from bar
to bar, which is the entire point of scaling.
"""
function coefficient_report(model::BayesianReturnModel)
    require_fitted(model)
    scaler = model.scaler
    scaler === nothing && throw(NotFittedError("BayesianReturnModel"))

    means = coefficients(model.regression)
    spreads = coefficient_std(model.regression)
    raw = vcat(means[1], unscale_coefficients(scaler, means[2:end]))

    report = Dict{Symbol, Dict{String, Float64}}()
    for (index, column) in enumerate(design_columns(model))
        report[column] = Dict{String, Float64}(
            "standardised" => means[index],
            "standardised_std" => spreads[index],
            "raw" => raw[index],
            "z" => spreads[index] > 0 ? means[index] / spreads[index] : 0.0,
        )
    end
    return report
end

function reset!(model::BayesianReturnModel)
    reset!(model.regression)
    model.scaler = nothing
    return invoke(reset!, Tuple{ProbabilisticModel}, model)
end

"""
    restore!(model; scaler, regression_state, n_observations, ...)

Adopt a previously fitted state. Used by the loader, not by fitting.
"""
function restore!(
        model::BayesianReturnModel;
        scaler::FeatureScaler,
        regression_state::AbstractDict,
        n_observations::Integer,
        fitted_at::Union{DateTime, Nothing} = nothing,
        train_start::Union{DateTime, Nothing} = nothing,
        train_end::Union{DateTime, Nothing} = nothing,
    )
    scaler.names == model.feature_names || throw(
        ArgumentError(
            string(
                "scaler describes ", scaler.names, ", model expects ", model.feature_names,
            ),
        ),
    )
    load_state!(model.regression, regression_state)
    model.scaler = scaler
    return mark_fitted!(
        model; n_observations = n_observations, fitted_at = fitted_at,
        train_start = train_start, train_end = train_end,
    )
end

function require_fitted(model::BayesianReturnModel)
    (is_fitted(model) && model.scaler !== nothing) ||
        throw(NotFittedError("BayesianReturnModel"))
    return nothing
end

function build_design(model::BayesianReturnModel, raw::Matrix{Float64})
    scaler = model.scaler
    scaler === nothing && throw(NotFittedError("BayesianReturnModel"))
    return build_design(model, raw, scaler)
end

function build_design(
        ::BayesianReturnModel, raw::Matrix{Float64}, scaler::FeatureScaler,
    )
    scaled = transform(scaler, raw)
    design = Matrix{Float64}(undef, Base.size(scaled, 1), Base.size(scaled, 2) + 1)
    for row in 1:Base.size(scaled, 1)
        design[row, 1] = 1.0
        for column in 1:Base.size(scaled, 2)
            design[row, column + 1] = scaled[row, column]
        end
    end
    return design
end

function build_row(model::BayesianReturnModel, raw::Vector{Float64})
    scaler = model.scaler
    scaler === nothing && throw(NotFittedError("BayesianReturnModel"))
    scaled = transform_row(scaler, raw)
    row = Vector{Float64}(undef, length(scaled) + 1)
    row[1] = 1.0
    for index in eachindex(scaled)
        row[index + 1] = scaled[index]
    end
    return row
end

function check_horizons(
        model::BayesianReturnModel, observations::AbstractVector{TrainingExample},
    )
    mismatched = sort!(
        unique(
            Int[
                example.label.horizon_bars for example in observations
                    if example.label.horizon_bars != model.horizon_bars
            ]
        )
    )
    isempty(mismatched) || throw(
        HorizonMismatchError(
            string(
                "model expects ", model.horizon_bars, "-bar labels, training set contains ",
                mismatched,
            ),
        ),
    )
    return nothing
end
