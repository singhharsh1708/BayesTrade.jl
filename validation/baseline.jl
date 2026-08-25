#!/usr/bin/env julia
#
#   julia --project=validation validation/baseline.jl
#
# Section 1 of the validation brief: measure what the system does today, before anything is
# changed, so that every later claim of improvement or regression has something to be measured
# against. Nothing here asserts. It records.

include(joinpath(@__DIR__, "harness.jl"))

const SIZES = (small = 500, medium = 2_000, large = 10_000)

"""
    pipeline_latencies(examples)

One pass along the whole chain, timing each stage separately.

Timed one call at a time rather than as a whole loop, because a total tells you the system is
slow and nothing about where. The stages are the ones that run per bar in a live session.
"""
forecast(model, example) = predict(
    model, example.features; symbol = example.features.symbol,
    as_of = example.features.as_of, horizon_bars = 1,
)

function pipeline_latencies(examples)
    warm = examples[1:min(600, length(examples))]
    models = ProbabilisticModel[factory() for factory in standard_factories()]
    for model in models
        fit!(model, warm)
    end
    later = examples[min(601, length(examples))]

    results = Dict{String, Any}()
    for model in models
        name = string(slug(model_name(model)))
        results[string("update_", name)] = timed(() -> update!(deepcopy(model), later))
        results[string("predict_", name)] = timed(() -> forecast(model, later))
    end

    reliability = ModelReliability(ModelName[model_name(m) for m in models])
    predictions = Tuple(forecast(model, later) for model in models)
    results["fuse"] = timed(() -> fuse(reliability, predictions))

    fused = fuse(reliability, predictions)
    limits = RiskLimits()
    results["decide"] = timed(() -> decide(fused, limits))

    intent = decide(fused, limits)
    book = Portfolio(equity = 1.0e6, cash = 1.0e6, as_of = later.features.as_of)
    results["risk_review"] = timed(() -> review(intent, book, limits))

    price = Quote(
        "VALID", later.features.as_of, 1000.0; volume = 100_000.0,
    )
    broker = PaperBroker(starting_cash = 1.0e6)
    ruling = review(intent, book, limits)
    order = order_from_ruling(broker, ruling, price, 1.0e6)
    if order !== nothing
        results["paper_execution"] = timed(
            () -> place_order!(PaperBroker(starting_cash = 1.0e6), order, price),
        )
    end
    return results
end

function main()
    started = now(UTC)
    baseline = Dict{String, Any}(
        "recorded_at" => string(started),
        "julia" => string(VERSION),
        "commit" => strip(read(`git rev-parse --short HEAD`, String)),
        "cpu_threads" => Sys.CPU_THREADS,
        "machine" => string(Sys.MACHINE),
    )

    heading("dataset scaling")
    scaling = Dict{String, Any}()
    for (label, n_bars) in pairs(SIZES)
        series = fixed_series(n_bars = n_bars)
        features = timed(() -> training_set(series); samples = 3, warmup = 1)
        examples = training_set(series)
        entry = Dict{String, Any}(
            "bars" => n_bars,
            "examples" => length(examples),
            "feature_seconds" => rounded(features.seconds),
            "feature_bytes" => features.bytes,
            "bars_per_second" => rounded(n_bars / features.seconds, 1),
        )
        scaling[string(label)] = entry
        @printf(
            "%-8s %6d bars  %6d examples  %8.3f s  %8.1f bars/s  %8.1f MB\n",
            label, n_bars, length(examples), features.seconds,
            n_bars / features.seconds, features.bytes / 1024^2,
        )
    end
    baseline["feature_generation"] = scaling

    heading("per-bar latency")
    series = fixed_series()
    examples = training_set(series)
    latency = pipeline_latencies(examples)
    stages = Dict{String, Any}()
    for stage in sort(collect(keys(latency)))
        measurement = latency[stage]
        stages[stage] = Dict{String, Any}(
            "microseconds" => rounded(measurement.seconds * 1.0e6, 2),
            "bytes" => measurement.bytes,
        )
        @printf(
            "%-28s %10.2f us  %10d bytes\n",
            stage, measurement.seconds * 1.0e6, measurement.bytes,
        )
    end
    baseline["latency"] = stages

    heading("walk-forward and calibration")
    report = replay(
        standard_factories(), examples;
        config = ReplayConfig(warmup = 500, refit_every = 25),
    )
    calibration = report.calibration
    coverage = Dict{String, Any}(
        string(point.level) => rounded(point.empirical, 4)
            for point in calibration.coverage
    )
    baseline["calibration"] = Dict{String, Any}(
        "n" => calibration.n,
        "scored" => length(report.records),
        "interval_error" => rounded(interval_calibration_error(calibration)),
        "expected_calibration_error" => rounded(calibration.expected_calibration_error),
        "brier_score" => rounded(calibration.brier_score),
        "pit_ks" => rounded(calibration.pit_ks_statistic),
        "mean_log_score" => rounded(calibration.mean_log_score),
        "sharpness" => rounded(calibration.sharpness),
        "bias" => rounded(calibration.bias),
        "overconfident" => is_overconfident(calibration),
        "coverage" => coverage,
    )
    println(summarise(calibration))
    for point in calibration.coverage
        @printf(
            "  nominal %.2f  empirical %.4f  error %+.4f\n",
            point.level, point.empirical, point.empirical - point.level,
        )
    end

    heading("decisions and risk")
    limits = RiskLimits()
    book = Portfolio(equity = 1.0e6, cash = 1.0e6, as_of = last(examples).features.as_of)
    actions = Dict{String, Int}()
    reasons = Dict{String, Int}()
    approved_total = 0.0
    requested_total = 0.0
    failures_seen = Dict{String, Int}()
    for record_ in report.records
        intent = decide(record_.prediction, limits)
        action = string(slug(intent.action))
        actions[action] = get(actions, action, 0) + 1
        if intent.reason !== nothing
            key = string(slug(intent.reason))
            reasons[key] = get(reasons, key, 0) + 1
        end
        ruling = review(intent, book, limits)
        requested_total += ruling.requested_weight
        approved_total += ruling.approved_weight
        for check in failures(ruling)
            key = string(check.name)
            failures_seen[key] = get(failures_seen, key, 0) + 1
        end
    end
    baseline["decisions"] = Dict{String, Any}(
        "actions" => actions, "decline_reasons" => reasons,
        "requested_weight_total" => rounded(requested_total, 4),
        "approved_weight_total" => rounded(approved_total, 4),
        "risk_failures" => failures_seen,
    )
    for (action, count) in sort(collect(actions), by = first)
        @printf("  %-12s %5d\n", action, count)
    end
    for (reason, count) in sort(collect(reasons), by = first)
        @printf("    declined: %-24s %5d\n", reason, count)
    end

    heading("paper broker accounting")
    session = PaperTradingSession(
        series.symbol, standard_factories(), standard_features();
        horizon_bars = 1, warmup = 500, refit_every = 50,
        interval = Day(1), interval_label = "1d", max_silence = Day(3),
    )
    elapsed = @elapsed for bar in series.bars
        on_tick!(session, Quote(series.symbol, bar.timestamp, bar.close; volume = bar.volume))
    end
    session_result = session_report(session)
    baseline["paper_session"] = merge(
        Dict{String, Any}(string(k) => v for (k, v) in session_result),
        Dict{String, Any}(
            "wall_seconds" => rounded(elapsed, 3),
            "bars_per_second" => rounded(length(series.bars) / elapsed, 1),
        ),
    )
    for key in sort(collect(keys(session_result)))
        println("  ", rpad(key, 22), session_result[key])
    end
    @printf("  %-22s %.2f s  (%.0f bars/s)\n", "wall time", elapsed, length(series.bars) / elapsed)

    baseline["total_seconds"] = rounded(
        (now(UTC) - started).value / 1000, 2,
    )
    path = record("baseline", baseline)
    heading("written")
    println(path)
    return nothing
end

main()
