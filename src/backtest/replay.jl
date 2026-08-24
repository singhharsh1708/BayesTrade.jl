"""
Deterministic replay of the same code path the live loop will run.

The point is not to produce a performance number. It is that a backtest and a live session
should differ in exactly one respect, where the bars come from, and in no other. Anything the
backtest does that live cannot do is a lie the backtest tells, and the expensive ones are all
the same lie: seeing a bar before it closed.

Three things make that structural rather than a matter of care:

* the clock only moves forward, and every prediction is stamped with the moment it was made
* features are built through the same engine the live loop uses, which cannot read past `as_of`
* a model is scored on a bar before that bar is given to it, never after

What comes out is a record per bar: what was believed, what happened, and which models were
behind it. Turning that into a decision is the next layer's job, not this one's.
"""

"""
    ReplayRecord

One bar of a replay: the fused belief, the outcome, and what it cost to be wrong.
"""
struct ReplayRecord{D <: UnivariateDistribution}
    symbol::String
    as_of::DateTime
    realised_at::DateTime
    prediction::FusedPrediction{D}
    outcome::Float64
    log_score::Float64

    function ReplayRecord(
            prediction::FusedPrediction{D}, realised_at::DateTime, outcome::Real,
        ) where {D}
        realised_at > prediction.as_of || throw(
            ArgumentError(
                string(
                    "an outcome at ", realised_at, " cannot settle a prediction made at ",
                    prediction.as_of,
                ),
            ),
        )
        realised = Float64(outcome)
        isfinite(realised) || throw(ArgumentError("an outcome must be finite"))
        return new{D}(
            prediction.symbol, prediction.as_of, realised_at, prediction, realised,
            logpdf(prediction.distribution, realised),
        )
    end
end

predictives(records::Vector{<:ReplayRecord}) =
    [record.prediction.distribution for record in records]
outcomes(records::Vector{<:ReplayRecord}) = Float64[record.outcome for record in records]

"""
    ReplayConfig

How the replay is run.

`warmup` bars train the models before anything is scored. `refit_every` bars they are fitted
again from scratch; in between they absorb each bar recursively, which is what the live loop
does and is the reason `update!` exists at all.

Nothing is absorbed or scored until its outcome has actually been realised, which at a horizon
beyond one bar is several bars after the prediction was made.
"""
struct ReplayConfig
    warmup::Int
    refit_every::Int
    horizon_bars::Int

    function ReplayConfig(; warmup::Integer = 500, refit_every::Integer = 50, horizon_bars::Integer = 1)
        warmup >= MIN_REGIME_ROWS || throw(
            ArgumentError(
                string("warmup must be at least ", MIN_REGIME_ROWS, ", got ", warmup),
            ),
        )
        refit_every >= 1 ||
            throw(ArgumentError(string("refit_every must be positive, got ", refit_every)))
        horizon_bars >= 1 ||
            throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
        return new(Int(warmup), Int(refit_every), Int(horizon_bars))
    end
end

"""
    ReplayReport

Everything a replay produced, and the calibration of what it believed.
"""
struct ReplayReport{R <: ReplayRecord}
    symbol::String
    records::Vector{R}
    reliability::ModelReliability
    calibration::CalibrationReport
    n_examples::Int
    n_skipped::Int
end

Base.length(report::ReplayReport) = length(report.records)

"""
    replay(factories, examples; config, forgetting)

Walk a labelled series bar by bar, fusing what the models believe and scoring it after.

`factories` is one zero-argument constructor per model. They are called at every refit, so a
replay never carries a model fitted on a window it is about to be scored on.

The ordering inside the loop is the whole point and is worth reading in order: settle whatever
has actually been realised by now, refit if it is time, then predict from the state as it
stands. A prediction is recorded immediately and scored later, when its outcome exists.

At a horizon of one bar the settlement is always the previous bar and the queue is invisible.
At five bars it is not: scoring a prediction the moment it is made would hand the weights, and
the return model's own update, a week of information that had not happened yet.
"""
function replay(
        factories::Tuple, examples::AbstractVector{TrainingExample};
        config::ReplayConfig = ReplayConfig(), forgetting::Real = 0.99,
    )
    isempty(factories) && throw(ArgumentError("a replay needs at least one model"))
    # The first training window has to be a full warm-up and still stop short of the first
    # prediction by the horizon, so the replay starts that much later.
    required = config.warmup + config.horizon_bars + 1
    length(examples) >= required || throw(
        ArgumentError(
            string("need at least ", required, " rows to replay, got ", length(examples)),
        ),
    )
    for example in examples
        example.label.horizon_bars == config.horizon_bars || throw(
            HorizonMismatchError(
                string(
                    "replaying a ", config.horizon_bars, "-bar horizon against a ",
                    example.label.horizon_bars, "-bar label",
                ),
            ),
        )
    end
    stamps = DateTime[example.features.as_of for example in examples]
    issorted(stamps) || throw(ArgumentError("examples must be in chronological order"))

    models = map(factory -> factory(), factories)
    names = ModelName[model_name(model) for model in models]
    reliability = ModelReliability(names; forgetting = forgetting)

    records = ReplayRecord[]
    pending = Tuple{Int, Any}[]
    skipped = 0
    fitted_at = 0
    absorbed_through = 0

    for index in required:length(examples)
        now = examples[index].features.as_of

        # An outcome is usable only once it has happened. At a horizon of one bar that is
        # the next bar and the distinction is invisible, which is exactly why it has to be
        # written down: at five bars, scoring a prediction the moment it is made would feed
        # the weights, and the return model's own update, a week of future.
        while !isempty(pending)
            (position, results) = first(pending)
            examples[position].label.realised_at <= now || break
            popfirst!(pending)
            outcome = examples[position].label.forward_log_return
            score_fusion!(reliability, results, outcome)
            if position > absorbed_through
                for model in models
                    update!(model, examples[position])
                end
                absorbed_through = position
            end
        end

        if fitted_at == 0 || index - fitted_at >= config.refit_every
            # The training window stops short of every prediction by the horizon, so no row
            # in it carries a label that had not been realised by now.
            stop = index - config.horizon_bars - 1
            if stop >= config.warmup
                window = view(examples, 1:stop)
                for model in models
                    fit!(model, window)
                end
                fitted_at = index
                absorbed_through = stop
                filter!(entry -> first(entry) > stop, pending)
            end
        end

        example = examples[index]
        results = map(
            model -> predict(
                model, example.features; symbol = example.features.symbol,
                as_of = example.features.as_of, horizon_bars = config.horizon_bars,
            ),
            models,
        )

        outcome = example.label.forward_log_return
        if isfinite(outcome)
            push!(
                records,
                ReplayRecord(
                    fuse(reliability, results), example.label.realised_at, outcome,
                ),
            )
            push!(pending, (index, results))
        else
            skipped += 1
        end
    end

    typed = [record for record in records]
    return ReplayReport(
        first(examples).features.symbol, typed, reliability,
        assess(predictives(typed), outcomes(typed)),
        length(examples), skipped,
    )
end

"""
    summarise(report)

The replay in a form a person can read.
"""
function summarise(report::ReplayReport)
    lines = String[
        string(
            report.symbol, ": ", length(report), " scored bars of ", report.n_examples,
            " labelled rows",
        ),
        "",
        summarise(report.calibration),
        "",
        "  model reliability",
    ]
    weights = reliabilities(report.reliability)
    scores = mean_log_scores(report.reliability)
    pad = maximum(length(slug(name)) for name in report.reliability.names)
    for (index, name) in enumerate(report.reliability.names)
        push!(
            lines,
            @sprintf(
                "    %s  weight %6.2f%%   mean log score %+8.4f",
                rpad(slug(name), pad), 100 * weights[index], scores[index]
            ),
        )
    end
    return join(lines, "\n")
end

Base.show(io::IO, report::ReplayReport) = @printf(
    io, "<ReplayReport %s bars=%d models=%d>",
    report.symbol, length(report), n_models(report.reliability)
)
