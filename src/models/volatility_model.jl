"""
The Bayesian volatility model.

The second module in the fusion layer, and the first thing to exercise the model interface
from somewhere other than the return model. It wraps [`DiscountedVarianceFilter`](@ref) in the
contract the rest of the system speaks: fit, update, predict, report uncertainty.

What it emits is a distribution over the forward *return*, not over volatility. That is
deliberate. A volatility forecast cannot be falsified against a number the market prints,
whereas a return predictive can, and the calibration machinery scores exactly that. The
volatility posterior sits beside it for anyone who wants to read it.

Two consequences worth stating plainly, because they look like defects otherwise:

* The predictive is centred, so `probability_positive` is one half by construction and the
  directional half of a calibration report carries no information about this model. It is the
  spread that is being forecast, and the spread is what should be scored.
* The model reads its observation from a point-in-time *feature*, never from the label. That
  is what makes `update!` honest in a live loop: the quantity it absorbs is one the market has
  already printed.
"""

const MIN_VOLATILITY_ROWS = 60

"""
    VarianceSource

Where a variance observation comes from, and what it is worth.

A dispatch point rather than a container: each source knows which feature it reads and how
many equivalent observations that reading carries. A squared return is worth exactly one. A
range-based estimator carries several bars' worth of information from a single bar, which is
the whole reason to want one, and the conjugate update takes that weight directly.
"""
abstract type VarianceSource end

"""
    SquaredReturnSource(column = :log_return_1)

The squared deviation of one bar's return, worth one observation.

The conservative choice, and the only one shipped so far. Range estimators are more efficient
per bar but are biased low under overnight gaps, and entering a biased estimate at several
observations' worth of weight produces a posterior that is both wrong and confident, which is
the exact failure this layer exists to prevent.
"""
struct SquaredReturnSource <: VarianceSource
    column::Symbol

    SquaredReturnSource(column::Symbol = :log_return_1) = new(column)
end

"""
    source_columns(source)

Feature columns this source needs.
"""
source_columns(source::SquaredReturnSource) = Symbol[source.column]

"""
    source_name(source)

How the source records itself in a model file.
"""
source_name(::SquaredReturnSource) = "squared_return"

"""
    observe(source, features, centre)

The variance estimate and its weight, or `nothing` when this bar cannot be read.

`nothing` is a real answer rather than a failure. A bar whose feature is missing is a bar the
model must not learn from, and the alternative of substituting a zero is actively dangerous: a
halted feed absorbed as a run of zero returns drives the variance estimate down, and narrow is
the one direction a risk system must never be wrong in.
"""
function observe(source::SquaredReturnSource, features::FeatureVector, centre::Float64)
    haskey(features, source.column) || return nothing
    value = require(features, source.column)
    isfinite(value) || return nothing
    return abs2(value - centre), 1.0
end

"""
    BayesianVolatilityModel

Forward return as a centred, heavy-tailed predictive whose scale is filtered from the market.

The scale is not a fitted constant. It is a posterior that decays old evidence at a rate the
model is itself uncertain about, which is what lets the same model be right in a quiet market
and in a violent one without being refitted by hand.
"""
mutable struct BayesianVolatilityModel{S <: VarianceSource} <: ProbabilisticModel
    source::S
    horizon_bars::Int
    filter::DiscountedVarianceFilter
    state::FitState

    function BayesianVolatilityModel(
            source::S = SquaredReturnSource();
            horizon_bars::Integer = 1,
            annual_volatility::Real = 0.3,
            prior_shape::Real = 2.0,
            discounts = DEFAULT_DISCOUNTS,
            weight_forgetting::Real = 0.98,
            centre::Real = 0.0,
            prior::Union{InverseGammaPrior, Nothing} = nothing,
        ) where {S <: VarianceSource}
        horizon_bars >= 1 ||
            throw(ArgumentError(string("horizon_bars must be at least 1, got ", horizon_bars)))
        resolved = prior === nothing ?
            variance_prior(
                volatility_scale = deannualise(annual_volatility), shape = prior_shape,
            ) : prior
        return new{S}(
            source, Int(horizon_bars),
            DiscountedVarianceFilter(
                resolved; discounts = discounts,
                weight_forgetting = weight_forgetting, centre = centre,
            ),
            FitState(),
        )
    end
end

fit_state(model::BayesianVolatilityModel) = model.state
model_name(::BayesianVolatilityModel) = VOLATILITY
model_semver(::BayesianVolatilityModel) = v"0.1.0"

"""
    feature_names(model)

Columns the model reads.
"""
feature_names(model::BayesianVolatilityModel) = source_columns(model.source)

"""
    fit!(model, observations)

Absorb a training window in order, starting again from the prior.

One forward pass. Order matters and is not an implementation detail: the filter discounts, so
the same rows in a different order describe a different market.
"""
function fit!(
        model::BayesianVolatilityModel, observations::AbstractVector{TrainingExample},
    )
    length(observations) >= MIN_VOLATILITY_ROWS || throw(
        ArgumentError(
            string(
                "need at least ", MIN_VOLATILITY_ROWS, " training rows, got ",
                length(observations),
            ),
        ),
    )
    check_horizons(model, observations)

    stamps = DateTime[example.features.as_of for example in observations]
    issorted(stamps) ||
        throw(ArgumentError("training rows must be in chronological order"))

    # Swapped in only once the window is through. Rebuilding in place leaves a failed refit
    # answering from the prior under the old fit's identity, and errs narrow.
    replacement = DiscountedVarianceFilter(
        model.filter.prior;
        discounts = model.filter.discounts,
        weight_forgetting = model.filter.weight_forgetting,
        centre = model.filter.centre,
    )
    for example in observations
        absorb!(replacement, model.source, example.features)
    end

    # A window it read no bar of is not a fit: the posterior is the prior, and stamping it
    # with the row count records a belief the model never formed.
    n_absorbed(replacement) > 0 || throw(
        ArgumentError(
            string(
                "none of the ", length(observations), " training rows carried a readable ",
                join(string.(feature_names(model)), ", "),
            ),
        ),
    )

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

Absorb one bar. Closed form, no refit.
"""
function update!(model::BayesianVolatilityModel, observation::TrainingExample)
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

Read one bar through the source and give it to the filter, or age the filter past it.

The single place a bar reaches a posterior, so the missing-bar policy cannot differ between
the batch path and the live one. It takes the filter rather than the model because the batch
path absorbs into a replacement filter that is not the model's yet.
"""
function absorb!(
        filter::DiscountedVarianceFilter, source::VarianceSource, features::FeatureVector,
    )
    reading = observe(source, features, filter.centre)
    if reading === nothing
        skip_observation!(filter)
    else
        estimate, weight = reading
        observe_variance!(filter, estimate; weight = weight)
    end
    return filter
end

"""
    check_horizons(model, observations)

Refuse a training set whose labels are not the horizon this model answers for.
"""
function check_horizons(
        model::BayesianVolatilityModel, observations::AbstractVector{TrainingExample},
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

The predictive distribution over the forward log return.

Pure: it reads no feature values and absorbs nothing. The prediction is a function of the
posterior alone, which is what makes it structurally impossible for this model to peek at the
bar it is being asked about, or to count a bar twice that the live loop has already given it.

The cost is staleness, and it is real. Under walk-forward the state is whatever the last
`fit!` left, so refitting less often means predicting from an older posterior. There is a test
that measures what that costs in log score rather than describing it.
"""
function predict(
        model::BayesianVolatilityModel, features::FeatureVector;
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
    distribution = return_predictive(filter; horizon_bars = model.horizon_bars)
    epistemic = variance_inflation(filter; horizon_bars = model.horizon_bars)
    total = plugin_variance(filter; horizon_bars = model.horizon_bars) + epistemic

    return ProbabilisticResult(
        model = model_version(model),
        symbol = symbol,
        as_of = as_of,
        horizon_bars = model.horizon_bars,
        distribution = distribution,
        n_observations = n_observations(model),
        epistemic_variance = epistemic,
        diagnostics = Dict{Symbol, Float64}(
            :bar_volatility => expected_volatility(filter),
            :annualised_volatility => annualise(expected_volatility(filter)),
            :effective_sample_size => effective_sample_size(filter),
            :predictive_df => predictive_df(filter),
            :epistemic_share => epistemic / total,
            :expected_discount => expected_discount(filter),
            :discount_entropy => discount_entropy(filter),
            :discount_disagreement => discount_disagreement(filter),
        ),
    )
end

"""
    predictive_df(filter)

Weighted mean degrees of freedom of the return predictive.

A diagnostic rather than a parameter: the mixture is not a Student-t and has no single degrees
of freedom, but a reader wants to know roughly how heavy the tails currently are.
"""
function predictive_df(filter::DiscountedVarianceFilter)
    total = 0.0
    for index in 1:n_components(filter)
        total += exp(filter.log_weights[index]) * 2 * evolved_shape(filter, index)
    end
    return total
end

"""
    can_predict(model, features)

Whether the model can answer for this vector.

Always true once fitted, and that is not an oversight. The prediction does not read the
features, so a warming-up vector does not stop it: the posterior is about the market, not
about this bar.
"""
can_predict(model::BayesianVolatilityModel, ::FeatureVector) = is_fitted(model)

"""
    uncertainty(model)

Posterior standard deviation of `log sigma`.

On the log scale, so it is a relative uncertainty and stays comparable between a quiet market
and a violent one. It does not fall to zero with more data: under a discount you never become
certain about a moving target, and reporting otherwise would be a lie the fusion layer would
act on.
"""
function uncertainty(model::BayesianVolatilityModel)
    is_fitted(model) || return Inf
    return volatility_uncertainty(model.filter)
end

"""
    reset!(model)

Forget the data and return to the prior.
"""
function reset!(model::BayesianVolatilityModel)
    reset!(model.filter)
    return invoke(reset!, Tuple{ProbabilisticModel}, model)
end

function parameters(model::BayesianVolatilityModel)
    recorded = parameters(model.filter)
    recorded["source"] = source_name(model.source)
    recorded["columns"] = String[string(name) for name in feature_names(model)]
    recorded["horizon_bars"] = model.horizon_bars
    return recorded
end

"""
    restore!(model; filter_state, n_observations, fitted_at, train_start, train_end)

Put a saved posterior back into a model built from the same configuration.
"""
function restore!(
        model::BayesianVolatilityModel;
        filter_state::AbstractDict,
        n_observations::Integer,
        fitted_at::Union{DateTime, Nothing} = nothing,
        train_start::Union{DateTime, Nothing} = nothing,
        train_end::Union{DateTime, Nothing} = nothing,
    )
    load_state!(model.filter, filter_state)
    return mark_fitted!(
        model; n_observations = n_observations, fitted_at = fitted_at,
        train_start = train_start, train_end = train_end,
    )
end

Base.show(io::IO, model::BayesianVolatilityModel) = @printf(
    io, "<BayesianVolatilityModel horizon=%d n=%d volatility=%.4g annual>",
    model.horizon_bars, n_observations(model),
    annualise(expected_volatility(model.filter))
)
