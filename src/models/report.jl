"""
Fitting a return model and reporting whether it can be believed.

This ties the phase together: build a labelled training set, score it walk-forward with the
embargo in place, report the calibration and the fitted coefficients, then hand back a model
fitted on everything.

The order matters. Fitting first and scoring second would be scoring in sample, which measures
nothing. The evaluation runs on data the model never saw, and only then is a final model fitted
on the whole history for use going forward.
"""

"""
    FitReport

What a fit run produced: the model, the out-of-sample calibration, and the rows behind both.
"""
struct FitReport{M <: BayesianReturnModel}
    symbol::String
    model::M
    calibration::CalibrationReport
    n_examples::Int
    n_scored::Int
end

"""
    fit_return_model(engine, symbol; horizon_bars, feature_names, config, forgetting)

Fit and score a return model over a symbol's history.

Returns a [`FitReport`](@ref). Raises if there is not enough history to train on the requested
window and still leave something to score, because a fit with nothing held out is a fit whose
quality is unknown.
"""
function fit_return_model(
        engine::FeatureEngine, symbol::AbstractString;
        horizon_bars::Integer = 5,
        feature_names::Union{AbstractVector{Symbol}, Nothing} = nothing,
        config::WalkForwardConfig = WalkForwardConfig(),
        forgetting::Real = 1.0,
    )
    columns_wanted = feature_names === nothing ? columns(engine.features) :
        convert(Vector{Symbol}, feature_names)
    examples = build_training_set(engine, symbol; horizon_bars = horizon_bars)

    required = config.initial_train + horizon_bars + 50
    length(examples) >= required || throw(
        ArgumentError(
            string(
                length(examples), " labelled rows is not enough to train on ",
                config.initial_train, " and still score out of sample; ", required,
                " are needed",
            ),
        ),
    )

    factory = () -> BayesianReturnModel(
        columns_wanted; horizon_bars = horizon_bars, forgetting = forgetting,
    )
    records = walk_forward(factory, examples, config)
    calibration = assess(predictives(records), outcomes(records))

    model = factory()
    fit!(model, examples)

    return FitReport(
        String(symbol), model, calibration, length(examples), length(records),
    )
end

"""
    summarise(report)

The fit report as a person would read it.
"""
function summarise(report::FitReport)
    model = report.model
    lines = String[
        string(
            report.symbol, ": ", report.n_examples, " labelled rows, ",
            length(model.feature_names), " features, ", model.horizon_bars,
            "-bar horizon",
        ),
        "",
        summarise(report.calibration),
        "",
        string(
            "fitted ", identifier(model_version(model)), " on ",
            n_observations(model), " rows",
        ),
        @sprintf(
            "  residual scale %.5g, uncertainty %.4f",
            residual_scale(model.regression), uncertainty(model)
        ),
        "",
    ]

    coefficients = coefficient_report(model)
    columns_ordered = design_columns(model)
    pad = maximum(length(string(column)) for column in columns_ordered)
    push!(lines, string("  ", rpad("column", pad), "  standardised          std       z"))
    for column in columns_ordered
        values = coefficients[column]
        push!(
            lines,
            string(
                "  ", rpad(string(column), pad),
                @sprintf(
                    "  %12.6g  %11.6g  %6.2f",
                    values["standardised"], values["standardised_std"], values["z"]
                ),
            ),
        )
    end
    return join(lines, '\n')
end
