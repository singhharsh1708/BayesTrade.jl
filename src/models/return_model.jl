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
    BayesianReturnModel

Forward log return as a Student-t posterior predictive.

The intercept is fitted rather than assumed zero, and it is not standardised. On a return
model it absorbs the unconditional drift, so leaving it out would force the features to
explain a level they have nothing to do with.
"""
mutable struct BayesianReturnModel <: ProbabilisticModel
    feature_names::Vector{Symbol}
    horizon_bars::Int
    regression::BayesianLinearModel
    scaler::Union{FeatureScaler, Nothing}
    state::FitState

    function BayesianReturnModel(
            feature_names::AbstractVector{Symbol};
            horizon_bars::Integer,
            residual_scale::Real = 0.02,
            coefficient_scale::Real = 0.5,
            prior_shape::Real = 2.0,
            forgetting::Real = 1.0,
            prior::Union{NormalInverseGammaPrior, Nothing} = nothing,
        )
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
        resolved = prior === nothing ?
            weakly_informative_prior(
                width; residual_scale = residual_scale,
                coefficient_scale = coefficient_scale, shape = prior_shape,
            ) : prior

        return new(
            convert(Vector{Symbol}, feature_names), Int(horizon_bars),
            BayesianLinearModel(resolved; forgetting = forgetting), nothing, FitState(),
        )
    end
end

fit_state(model::BayesianReturnModel) = model.state
model_name(::BayesianReturnModel) = MOMENTUM
model_semver(::BayesianReturnModel) = v"0.1.0"

"""
    design_columns(model)

The design columns, intercept first. This order is part of the model.
"""
design_columns(model::BayesianReturnModel) = vcat(INTERCEPT, model.feature_names)

"""
    response_scale(model, features)

Units the response is measured in at this point in time.

One here: the plain model assumes the noise scale is constant and lets the regression estimate
it. A model that knows better defines its own method, and everything else — the conjugate
update, the version hash, the predictive — follows unchanged.
"""
response_scale(::BayesianReturnModel, ::FeatureVector) = 1.0

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

    stamps = DateTime[example.features.as_of for example in observations]
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
    is_fitted(model) && all(name -> haskey(features, name), model.feature_names)

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
    return Dict{String, Any}(
        "columns" => String[string(name) for name in design_columns(model)],
        "horizon_bars" => model.horizon_bars,
        "regression" => parameters(model.regression),
        "scaler" => scaler === nothing ? nothing : parameters(scaler),
    )
end

"""
    coefficient_report(model)

Per-column posterior summary, for reading a fitted model.

The standardised coefficient is the comparable one: it says how much the predicted return
moves per typical move in that feature. The raw one is what a reader checks against a
definition.
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
