"""
Whether the model is healthy, as distinct from whether the process is.

`ops/health.jl` answers "can this session act": is the feed alive, is the journal writable, is the
account in a state that permits trading. Every one of those can be true while the thing actually
making the decisions has quietly stopped working.

This answers the other question. A model whose intervals have stopped covering, whose posterior
has gone numerically strange, or whose regime belief is oscillating every bar is a model that
should not be sizing positions, and none of that shows up as an unhealthy process.

It is a diagnostic that can refuse. Where `ops/health.jl` gates on the machinery, this gates on
the mathematics, and the same rule applies to both: a check that cannot measure its input fails.
"""

"""
    ModelCheck

One named judgement about a model, with the number behind it.
"""
struct ModelCheck
    name::Symbol
    status::RiskCheckStatus
    detail::String
    observed::Float64
    allowed::Float64
end

"""
    ModelHealth

Every check, and whether the model may be trusted to size a position.
"""
struct ModelHealth
    as_of::DateTime
    checks::Vector{ModelCheck}
    healthy::Bool
end

model_problems(health::ModelHealth) =
    ModelCheck[check for check in health.checks if check.status === FAIL]

trustworthy(health::ModelHealth) = health.healthy

Base.show(io::IO, health::ModelHealth) = @printf(
    io, "<ModelHealth %s %s%s>", health.as_of, health.healthy ? "HEALTHY" : "UNHEALTHY",
    health.healthy ? "" :
        string(" [", join(String[string(c.name) for c in model_problems(health)], ", "), "]"),
)

"""
    MODEL_HEALTH_BOUNDS

Where each model check turns from pass to fail.

Conventional, and stated here rather than scattered through the function so that changing one is
a visible decision. None of them is derived from anything: the brief is explicit that a threshold
invented to make a system pass is worse than no threshold, so these are set where a person would
call the behaviour clearly wrong rather than where the current system happens to sit.
"""
const MODEL_HEALTH_BOUNDS = (
    interval_error = 0.15,          # coverage off by fifteen points is not a calibration wobble
    brier = 0.3,                   # worse than always saying fifty percent
    uncertainty = 1.0,              # an annualised spread of one hundred percent
    min_observations = 30,
    max_disagreement_share = 0.9,  # the pool is almost entirely models contradicting each other
)

"""
    assess_model_health(; as_of, calibration, uncertainty, n_observations, disagreement, fitted)

Judge a model from what is known about it.

Every argument is optional because every one of them can genuinely be absent, and absence is not
the same as a pass. A model with no calibration yet is recorded as SKIPPED and a model whose
calibration cannot be computed is recorded as FAIL, because the second one is a model that has
stopped being measurable while still being used.
"""
function assess_model_health(;
        as_of::DateTime,
        fitted::Bool,
        calibration::Union{CalibrationReport, Nothing} = nothing,
        uncertainty::Union{Real, Nothing} = nothing,
        n_observations::Union{Integer, Nothing} = nothing,
        disagreement_share::Union{Real, Nothing} = nothing,
    )
    checks = ModelCheck[]
    pass(name, detail, observed, allowed) =
        push!(checks, ModelCheck(name, PASS, detail, Float64(observed), Float64(allowed)))
    fail(name, detail, observed, allowed) =
        push!(checks, ModelCheck(name, FAIL, detail, Float64(observed), Float64(allowed)))
    skip(name, detail) =
        push!(checks, ModelCheck(name, SKIPPED, detail, NaN, NaN))

    fitted ? pass(:fitted, "the model is fitted", 1.0, 1.0) :
        fail(:fitted, "the model is not fitted", 0.0, 1.0)

    if calibration === nothing
        skip(:calibration, "no calibration yet")
        skip(:accuracy, "no calibration yet")
    else
        error = interval_calibration_error(calibration)
        isfinite(error) ?
            (
                error <= MODEL_HEALTH_BOUNDS.interval_error ?
                pass(:calibration, "intervals hold", error, MODEL_HEALTH_BOUNDS.interval_error) :
                fail(
                    :calibration, string("interval error ", round(error; digits = 4)),
                    error, MODEL_HEALTH_BOUNDS.interval_error,
                )
            ) :
            fail(:calibration, "interval error is not a number", NaN, MODEL_HEALTH_BOUNDS.interval_error)

        brier = calibration.brier_score
        isfinite(brier) ?
            (
                brier <= MODEL_HEALTH_BOUNDS.brier ?
                pass(:accuracy, "directional accuracy holds", brier, MODEL_HEALTH_BOUNDS.brier) :
                fail(
                    :accuracy, string("Brier score ", round(brier; digits = 4)),
                    brier, MODEL_HEALTH_BOUNDS.brier,
                )
            ) :
            fail(:accuracy, "Brier score is not a number", NaN, MODEL_HEALTH_BOUNDS.brier)
    end

    if uncertainty === nothing
        skip(:posterior, "no uncertainty supplied")
    elseif !isfinite(uncertainty)
        # The state that matters most. An infinite or NaN spread is a model that has come apart
        # numerically, and it is exactly the state a threshold comparison would let through.
        fail(:posterior, "the posterior is not finite", NaN, MODEL_HEALTH_BOUNDS.uncertainty)
    elseif uncertainty > MODEL_HEALTH_BOUNDS.uncertainty
        fail(
            :posterior, string("uncertainty ", round(uncertainty; digits = 4)),
            uncertainty, MODEL_HEALTH_BOUNDS.uncertainty,
        )
    else
        pass(:posterior, "posterior is usable", uncertainty, MODEL_HEALTH_BOUNDS.uncertainty)
    end

    if n_observations === nothing
        skip(:evidence, "no observation count supplied")
    else
        n_observations >= MODEL_HEALTH_BOUNDS.min_observations ?
            pass(
                :evidence, "enough evidence", n_observations,
                MODEL_HEALTH_BOUNDS.min_observations,
            ) :
            fail(
                :evidence, string(n_observations, " observations"), n_observations,
                MODEL_HEALTH_BOUNDS.min_observations,
            )
    end

    if disagreement_share === nothing
        skip(:agreement, "no disagreement supplied")
    elseif !isfinite(disagreement_share)
        fail(:agreement, "disagreement is not a number", NaN, MODEL_HEALTH_BOUNDS.max_disagreement_share)
    elseif disagreement_share > MODEL_HEALTH_BOUNDS.max_disagreement_share
        fail(
            :agreement, string("the pool is ", round(100 * disagreement_share; digits = 1), "% disagreement"),
            disagreement_share, MODEL_HEALTH_BOUNDS.max_disagreement_share,
        )
    else
        pass(
            :agreement, "the pool is not purely disagreement", disagreement_share,
            MODEL_HEALTH_BOUNDS.max_disagreement_share,
        )
    end

    return ModelHealth(as_of, checks, !any(check -> check.status === FAIL, checks))
end

"""
    model_health_report(health)

The report as a person reads it.
"""
function model_health_report(health::ModelHealth)
    buffer = IOBuffer()
    println(buffer, "Model Health")
    println(buffer, "-"^40)
    for check in health.checks
        label = check.status === PASS ? "PASS" :
            check.status === FAIL ? "FAIL" : "SKIP"
        @printf(buffer, "%-16s %-6s %s\n", string(check.name), label, check.detail)
    end
    println(buffer, "-"^40)
    println(buffer, "Overall: ", health.healthy ? "HEALTHY" : "UNHEALTHY")
    return String(take!(buffer))
end
