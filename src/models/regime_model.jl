"""
The market regime model.

The third module in the fusion layer. It wraps [`RegimeFilter`](@ref) in the contract the rest
of the system speaks, and reports two things a decision layer wants and cannot get from the
other models: which regime this probably is, and how sure that is.

Like the volatility model it emits a distribution over the forward log return rather than over
a label, so the existing walk-forward and calibration machinery scores it unchanged. A model
whose only output is a state probability cannot be falsified against anything the market
prints, and would sit in the fusion layer accountable to nobody.
"""

const MIN_REGIME_ROWS = 120

"""
    RegimeSource

Where the filter's observation comes from.
"""
abstract type RegimeSource end

"""
    BarReturnSource(column = :log_return_1)

One bar's log return, read from a point-in-time feature.
"""
struct BarReturnSource <: RegimeSource
    column::Symbol

    BarReturnSource(column::Symbol = :log_return_1) = new(column)
end

source_columns(source::BarReturnSource) = Symbol[source.column]
source_name(::BarReturnSource) = "bar_return"

"""
    observe(source, features)

The return for this bar, or `nothing` when it cannot be read.
"""
function observe(source::BarReturnSource, features::FeatureVector)
    haskey(features, source.column) || return nothing
    value = require(features, source.column)
    return isfinite(value) ? value : nothing
end

"""
    MarketRegimeModel

Forward return as a mixture over regimes, with the regime posterior reported beside it.
"""
mutable struct MarketRegimeModel{S <: RegimeSource} <: ProbabilisticModel
    source::S
    horizon_bars::Int
    prior::RegimePrior
    filter::RegimeFilter
    state::FitState

    function MarketRegimeModel(
            source::S = BarReturnSource();
            horizon_bars::Integer = 1,
            prior::RegimePrior = RegimePrior(),
        ) where {S <: RegimeSource}
        horizon_bars >= 1 ||
            throw(ArgumentError(string("horizon_bars must be at least 1, got ", horizon_bars)))
        empty = RegimeParameters(
            prior; centre = 0.0, drift_spread = prior.drift_scale,
            persistence = prior.persistence, mean_variance = 1.0e-4,
            dispersion = prior.dispersion, n_rows = 0,
        )
        return new{S}(source, Int(horizon_bars), prior, RegimeFilter(empty), FitState())
    end
end

fit_state(model::MarketRegimeModel) = model.state
model_name(::MarketRegimeModel) = REGIME
model_semver(::MarketRegimeModel) = v"0.1.0"
feature_names(model::MarketRegimeModel) = source_columns(model.source)

"""
    fit!(model, observations)

Estimate the four scalars from the window, then run the filter forward over it.

Estimated on the readable bars and filtered over every bar, so a gap decays the belief without
being counted as a quiet return. Nothing is installed until the whole window is through, so a
window that is refused leaves the model exactly as it was.
"""
function fit!(model::MarketRegimeModel, observations::AbstractVector{TrainingExample})
    length(observations) >= MIN_REGIME_ROWS || throw(
        ArgumentError(
            string(
                "need at least ", MIN_REGIME_ROWS, " training rows, got ",
                length(observations),
            ),
        ),
    )
    check_horizons(model, observations)

    stamps = DateTime[example.features.as_of for example in observations]
    issorted(stamps) ||
        throw(ArgumentError("training rows must be in chronological order"))

    readable = Float64[]
    for example in observations
        value = observe(model.source, example.features)
        value === nothing || push!(readable, value)
    end
    length(readable) >= MIN_REGIME_ROWS || throw(
        ArgumentError(
            string(
                "only ", length(readable), " of ", length(observations),
                " rows carried a readable ", join(string.(feature_names(model)), ", "),
            ),
        ),
    )

    replacement = RegimeFilter(
        estimate_regime_parameters(readable; prior = model.prior),
    )
    for example in observations
        absorb!(replacement, model.source, example.features)
    end

    model.filter = replacement
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

Absorb one bar. One propagate and one correct, no refit.
"""
function update!(model::MarketRegimeModel, observation::TrainingExample)
    require_fitted(model)
    check_horizons(model, [observation])
    require_later(model, observation.features.as_of)
    absorb!(model.filter, model.source, observation.features)
    model.state.n_observations += 1
    model.state.train_end = observation.features.as_of
    return model
end

"""
    absorb!(filter, source, features)

The single place a bar reaches the belief, so the batch and live paths cannot diverge.
"""
function absorb!(
        filter::RegimeFilter, source::RegimeSource, features::FeatureVector,
    )
    value = observe(source, features)
    value === nothing ? skip_observation!(filter) : observe_return!(filter, value)
    return filter
end

function check_horizons(
        model::MarketRegimeModel, observations::AbstractVector{TrainingExample},
    )
    for example in observations
        example.label.horizon_bars == model.horizon_bars || throw(
            HorizonMismatchError(
                string(
                    "model answers for ", model.horizon_bars, "-bar returns, given a ",
                    example.label.horizon_bars, "-bar label",
                ),
            ),
        )
    end
    return nothing
end

"""
    predict(model, features; symbol, as_of, horizon_bars)

The predictive over the forward log return, with the regime posterior in the diagnostics.

Pure: it reads no feature values and absorbs nothing, so it cannot see the bar it is asked
about or double-count one the live loop has already given it.
"""
function predict(
        model::MarketRegimeModel, features::FeatureVector;
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

    filter = model.filter
    horizon = model.horizon_bars
    distribution = predict_return(filter; horizon_bars = horizon)
    split = variance_decomposition(filter; horizon_bars = horizon)
    epistemic = split.state + split.parameter
    total = split.aleatoric + epistemic
    belief = regime_probabilities(filter)

    return ProbabilisticResult(
        model = model_version(model),
        symbol = symbol,
        as_of = as_of,
        horizon_bars = horizon,
        distribution = distribution,
        n_observations = n_observations(model),
        epistemic_variance = epistemic,
        diagnostics = Dict{Symbol, Float64}(
            :bull => belief[1],
            :bear => belief[2],
            :sideways => belief[3],
            :confidence => regime_confidence(filter),
            :persistence => filter.parameters.persistence,
            :expected_duration => 1 / max(1 - filter.parameters.persistence, 1.0e-12),
            :epistemic_share => epistemic / total,
            :state_share => split.state / total,
        ),
    )
end

"""
    regime_belief(model)

The regime posterior, carrying its own labels.
"""
regime_belief(model::MarketRegimeModel) = regime_belief(model.filter)
regime_probabilities(model::MarketRegimeModel) = regime_probabilities(model.filter)
regime_confidence(model::MarketRegimeModel) = regime_confidence(model.filter)
most_likely_regime(model::MarketRegimeModel) = most_likely_regime(model.filter)

"""
    can_predict(model, features)

Always true once fitted: the prediction is a function of the belief, not of this bar.
"""
can_predict(model::MarketRegimeModel, ::FeatureVector) = is_fitted(model)

"""
    uncertainty(model)

Normalised entropy of the regime posterior, in `[0, 1]`.

Dimensionless, so it stays comparable across markets and does not have to be rescaled to sit
beside the other models in the fusion layer.
"""
function uncertainty(model::MarketRegimeModel)
    is_fitted(model) || return Inf
    return normalised_entropy(regime_belief(model))
end

function reset!(model::MarketRegimeModel)
    reset!(model.filter)
    return invoke(reset!, Tuple{ProbabilisticModel}, model)
end

function parameters(model::MarketRegimeModel)
    recorded = parameters(model.filter)
    recorded["source"] = source_name(model.source)
    recorded["columns"] = String[string(name) for name in feature_names(model)]
    recorded["horizon_bars"] = model.horizon_bars
    return recorded
end

"""
    restore!(model; parameters, filter_state, n_observations, ...)

Put a saved belief and its parameters back into a model built from the same prior.
"""
function restore!(
        model::MarketRegimeModel;
        parameters::RegimeParameters,
        filter_state::AbstractDict,
        n_observations::Integer,
        fitted_at::Union{DateTime, Nothing} = nothing,
        train_start::Union{DateTime, Nothing} = nothing,
        train_end::Union{DateTime, Nothing} = nothing,
    )
    filter = RegimeFilter(parameters)
    load_state!(filter, filter_state)
    model.filter = filter
    return mark_fitted!(
        model; n_observations = n_observations, fitted_at = fitted_at,
        train_start = train_start, train_end = train_end,
    )
end

Base.show(io::IO, model::MarketRegimeModel) = @printf(
    io, "<MarketRegimeModel %s p=%.3f horizon=%d n=%d>",
    slug(most_likely_regime(model.filter)), maximum(regime_probabilities(model.filter)),
    model.horizon_bars, n_observations(model)
)
