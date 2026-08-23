"""
Walk-forward evaluation.

Scoring a model on the data it was fitted to measures nothing. It will look calibrated whether
or not it is, because the residuals it is being judged on are the ones it minimised. Every
honest number in this project comes from here.

The subtlety that makes this harder than it looks is the **embargo**. A training example at
time `t` is not usable until its label has been realised, which for a horizon of `h` bars is
`h` bars later. Fitting on every example up to `t` and then predicting at `t` uses labels that
had not happened yet, and the leak is completely invisible in the output: the backtest simply
looks better than it is.

So the training window always stops `h` bars short of the prediction point. That costs a
handful of rows and removes the single most common way a walk-forward backtest lies.
"""

"""
    WalkForwardConfig

How the evaluation steps through history.
"""
Base.@kwdef struct WalkForwardConfig
    initial_train::Int = 250
    refit_every::Int = 20
    max_train::Union{Int, Nothing} = nothing
    embargo_bars::Union{Int, Nothing} = nothing

    function WalkForwardConfig(initial_train, refit_every, max_train, embargo_bars)
        initial_train >= 2 || throw(
            ArgumentError(
                string("initial_train must be at least 2, got ", initial_train),
            ),
        )
        refit_every >= 1 ||
            throw(ArgumentError(string("refit_every must be at least 1, got ", refit_every)))
        if max_train !== nothing && max_train < initial_train
            throw(
                ArgumentError(
                    string(
                        "max_train ", max_train, " is below initial_train ", initial_train,
                    ),
                ),
            )
        end
        if embargo_bars !== nothing && embargo_bars < 0
            throw(
                ArgumentError(
                    string("embargo_bars must be non-negative, got ", embargo_bars),
                ),
            )
        end
        return new(initial_train, refit_every, max_train, embargo_bars)
    end
end

"""
    PredictionRecord

One out-of-sample prediction and what followed it.

The constructor enforces the embargo. A record whose training data outran its own prediction
cannot be built, so the guarantee holds even if the loop that fills it in has a bug.
"""
struct PredictionRecord{D}
    symbol::String
    as_of::DateTime
    realised_at::DateTime
    predictive::D
    outcome::Float64
    model::ModelVersion
    train_rows::Int
    train_realised_through::DateTime

    function PredictionRecord(;
            symbol::AbstractString,
            as_of::DateTime,
            realised_at::DateTime,
            predictive::D,
            outcome::Real,
            model::ModelVersion,
            train_rows::Integer,
            train_realised_through::DateTime,
        ) where {D}
        train_realised_through <= as_of || throw(
            ArgumentError(
                string(
                    "prediction at ", as_of, " used a label realised at ",
                    train_realised_through,
                ),
            ),
        )
        return new{D}(
            String(symbol), as_of, realised_at, predictive, Float64(outcome),
            model, Int(train_rows), train_realised_through,
        )
    end
end

went_up(record::PredictionRecord) = record.outcome > 0
prediction_error(record::PredictionRecord) = record.outcome - mean(record.predictive)

"""
    walk_forward(factory, examples, config)

Refit forward through history, predicting only ahead of what was known.

`factory` builds a fresh model for each refit rather than updating one in place, so a rolling
window genuinely forgets and an expanding one genuinely starts from the prior.
"""
function walk_forward(
        factory, examples::AbstractVector{TrainingExample},
        config::WalkForwardConfig = WalkForwardConfig(),
    )
    isempty(examples) && return PredictionRecord[]

    horizons = sort!(unique(Int[example.label.horizon_bars for example in examples]))
    length(horizons) == 1 ||
        throw(ArgumentError(string("examples mix horizons ", horizons)))
    embargo = config.embargo_bars === nothing ? first(horizons) : config.embargo_bars

    first_index = config.initial_train + embargo + 1
    first_index > length(examples) && return PredictionRecord[]

    records = PredictionRecord[]
    model = nothing
    fitted_rows = 0
    fitted_through = nothing

    for index in first_index:length(examples)
        train_stop = index - embargo - 1
        if model === nothing || (index - first_index) % config.refit_every == 0
            start = config.max_train === nothing ? 1 :
                max(1, train_stop - config.max_train + 1)
            window = examples[start:train_stop]
            model = factory()
            fit!(model, window)
            fitted_rows = length(window)
            fitted_through = last(window).label.realised_at
        end

        example = examples[index]
        result = predict(
            model, example.features;
            symbol = example.features.symbol,
            as_of = example.features.as_of,
            horizon_bars = example.label.horizon_bars,
        )
        push!(
            records,
            PredictionRecord(
                symbol = example.features.symbol,
                as_of = example.features.as_of,
                realised_at = example.label.realised_at,
                predictive = result.distribution,
                outcome = example.label.forward_log_return,
                model = result.model,
                train_rows = fitted_rows,
                train_realised_through = fitted_through,
            ),
        )
    end
    return records
end

outcomes(records::AbstractVector{<:PredictionRecord}) =
    Float64[record.outcome for record in records]

predictives(records::AbstractVector{<:PredictionRecord}) =
    [record.predictive for record in records]
