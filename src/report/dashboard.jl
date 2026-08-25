"""
The dashboard payload.

The dashboard is a separate program in a separate language, so the contract between them is
JSON and nothing else. No Julia type crosses this line, which is what lets the dashboard be
rewritten, replaced or thrown away without touching anything that decides a trade.

What it carries is deliberately the uncomfortable half. Any dashboard can draw an equity curve.
This one carries calibration, model disagreement, the share of each prediction that is
reducible uncertainty, and every refusal with the reason attached, because those are the
numbers that say whether the equity curve means anything.
"""

const DASHBOARD_SCHEMA_VERSION = 1

"""
    dashboard_payload(report; limits, portfolio, sector, generated_at)

Everything the dashboard draws, from a replay that has already happened.

Takes the replay rather than the models, so the payload can only contain things that were
actually predicted at the time. There is no path here that could evaluate a model against a bar
it has already seen.
"""
function dashboard_payload(
        report::ReplayReport;
        limits::RiskLimits = RiskLimits(),
        book::Union{Portfolio, Nothing} = nothing,
        sector::Union{AbstractString, Nothing} = nothing,
        generated_at::DateTime,
    )
    records = report.records
    isempty(records) && throw(ArgumentError("an empty replay has nothing to show"))

    series = Vector{Dict{String, Any}}(undef, length(records))
    for (index, record) in enumerate(records)
        prediction = record.prediction
        spread = std(prediction)
        series[index] = Dict{String, Any}(
            "as_of" => string(record.as_of),
            "mean" => mean(prediction),
            "sd" => spread,
            "lower" => quantile(prediction.distribution, 0.05),
            "upper" => quantile(prediction.distribution, 0.95),
            "outcome" => record.outcome,
            "log_score" => record.log_score,
            "probability_up" => probability_positive(prediction),
            "epistemic_share" => epistemic_share(prediction),
            "disagreement" => prediction.diagnostics[:disagreement],
            "disagreement_share" => disagreement_share(prediction),
            "agreement" => slug(model_agreement(prediction)),
            "weights" => collect(prediction.weights.probabilities),
        )
    end

    calibration = report.calibration
    coverage = [
        Dict{String, Any}(
                "level" => point.level, "empirical" => point.empirical,
                "error" => coverage_error(point),
            ) for point in calibration.coverage
    ]

    decisions = Vector{Dict{String, Any}}()
    if book !== nothing
        for record in records
            intent = decide(record.prediction, limits)
            # Volatility is supplied the same way the live session supplies it. Without it the
            # gate is skipped, and the panel then shows a ruling weaker than the one the
            # system would actually have produced, which is the opposite of what a decision
            # log is for. Turnover cannot be supplied here: a replay record carries the
            # prediction and not the bar, so the liquidity gate stays visibly SKIPPED rather
            # than being invented.
            spread = annualise(std(record.prediction))
            ruling = sector === nothing ?
                review(intent, book, limits; annualised_volatility = spread) :
                review(
                    intent, book, limits; sector = sector, annualised_volatility = spread,
                )
            push!(
                decisions,
                Dict{String, Any}(
                    "as_of" => string(record.as_of),
                    "action" => slug(intent.action),
                    "reason" => intent.reason === nothing ? nothing : slug(intent.reason),
                    "requested" => ruling.requested_weight,
                    "approved" => ruling.approved_weight,
                    "failures" => String[string(check.name) for check in failures(ruling)],
                ),
            )
        end
    end

    return Dict{String, Any}(
        "schema" => DASHBOARD_SCHEMA_VERSION,
        "generated_at" => string(generated_at),
        "symbol" => report.symbol,
        "n_examples" => report.n_examples,
        "n_scored" => length(records),
        "models" => String[slug(name) for name in report.reliability.names],
        "reliabilities" => reliabilities(report.reliability),
        "mean_log_scores" => mean_log_scores(report.reliability),
        "calibration" => Dict{String, Any}(
            "n" => calibration.n,
            "interval_error" => interval_calibration_error(calibration),
            "expected_calibration_error" => calibration.expected_calibration_error,
            "brier_score" => calibration.brier_score,
            "pit_ks" => calibration.pit_ks_statistic,
            "mean_log_score" => calibration.mean_log_score,
            "sharpness" => calibration.sharpness,
            "bias" => calibration.bias,
            "overconfident" => is_overconfident(calibration),
            "coverage" => coverage,
        ),
        "series" => series,
        "decisions" => decisions,
        "limits" => Dict{String, Any}(
            "max_position_weight" => limits.max_position_weight,
            "max_portfolio_exposure" => limits.max_portfolio_exposure,
            "min_probability_positive" => limits.min_probability_positive,
            "max_probability_large_loss" => limits.max_probability_large_loss,
            "max_model_uncertainty" => limits.max_model_uncertainty,
            "risk_budget_per_trade" => limits.risk_budget_per_trade,
        ),
    )
end

"""
    write_dashboard(payload, path)

Write a payload where the dashboard can read it.
"""
function write_dashboard(payload::AbstractDict, path::AbstractString)
    mkpath(dirname(abspath(path)))
    open(path, "w") do handle
        JSON3.pretty(handle, payload)
        println(handle)
    end
    return path
end
