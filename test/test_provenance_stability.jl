# Sections 20, 23 and 25: is a run reconstructable, does the arithmetic fail safely, and is the
# model itself healthy as distinct from the process being healthy.

@testset "a run can be identified and reproduced" begin
    base() = run_manifest(
        label = "baseline", dataset = "ar1-2000", seed = 20260825,
        features = ["log_return_1", "volatility_20"],
        models = ["momentum", "volatility"],
        configuration = Dict("warmup" => 500, "refit_every" => 25),
        created_at = DateTime(2026, 8, 25, 14, 30),
    )

    @testset "the same run is the same identifier" begin
        @test base().run_id == base().run_id
        # Order must not matter: a manifest that hashes differently because a feature list was
        # sorted differently answers "is this reproducible" with a false no.
        shuffled = run_manifest(
            label = "baseline", dataset = "ar1-2000", seed = 20260825,
            features = ["volatility_20", "log_return_1"],
            models = ["volatility", "momentum"],
            configuration = Dict("refit_every" => 25, "warmup" => 500),
            created_at = DateTime(2026, 8, 25, 14, 30),
        )
        @test shuffled.run_id == base().run_id
    end

    @testset "the clock is recorded and not hashed" begin
        # A run repeated tomorrow with the same code, data and configuration is the same run.
        later = run_manifest(
            label = "baseline", dataset = "ar1-2000", seed = 20260825,
            features = ["log_return_1", "volatility_20"],
            models = ["momentum", "volatility"],
            configuration = Dict("warmup" => 500, "refit_every" => 25),
            created_at = DateTime(2027, 1, 1, 9, 0),
        )
        @test later.run_id == base().run_id
        @test later.created_at != base().created_at
        @test manifest_payload(later)["created_at"] == "2027-01-01T09:00:00"
    end

    @testset "anything that would change the answer changes the identifier" begin
        original = base()
        variants = (
            run_manifest(
                label = "baseline", dataset = "ar1-4000", seed = 20260825,
                features = ["log_return_1", "volatility_20"],
                models = ["momentum", "volatility"],
                configuration = Dict("warmup" => 500, "refit_every" => 25),
                created_at = DateTime(2026, 8, 25, 14, 30),
            ),
            run_manifest(
                label = "baseline", dataset = "ar1-2000", seed = 1,
                features = ["log_return_1", "volatility_20"],
                models = ["momentum", "volatility"],
                configuration = Dict("warmup" => 500, "refit_every" => 25),
                created_at = DateTime(2026, 8, 25, 14, 30),
            ),
            run_manifest(
                label = "baseline", dataset = "ar1-2000", seed = 20260825,
                features = ["log_return_1"],
                models = ["momentum", "volatility"],
                configuration = Dict("warmup" => 500, "refit_every" => 25),
                created_at = DateTime(2026, 8, 25, 14, 30),
            ),
            run_manifest(
                label = "baseline", dataset = "ar1-2000", seed = 20260825,
                features = ["log_return_1", "volatility_20"],
                models = ["momentum"],
                configuration = Dict("warmup" => 500, "refit_every" => 25),
                created_at = DateTime(2026, 8, 25, 14, 30),
            ),
            run_manifest(
                label = "baseline", dataset = "ar1-2000", seed = 20260825,
                features = ["log_return_1", "volatility_20"],
                models = ["momentum", "volatility"],
                configuration = Dict("warmup" => 501, "refit_every" => 25),
                created_at = DateTime(2026, 8, 25, 14, 30),
            ),
        )
        for variant in variants
            @test variant.run_id != original.run_id
        end
        @test length(unique(v.run_id for v in variants)) == length(variants)
    end

    @testset "the label reads like a name and the payload writes like a record" begin
        manifest = base()
        @test run_label(manifest) ==
            string("2026-08-25_BAYES_v", PACKAGE_VERSION, "_ar1-2000_", first(manifest.run_id, 8))
        payload = manifest_payload(manifest)
        for key in (
                "run_id", "label", "package_version", "dataset", "configuration",
                "seed", "features", "models", "created_at", "schema",
            )
            @test haskey(payload, key)
        end
        @test JSON3.read(JSON3.write(payload), Dict{String, Any})["run_id"] == manifest.run_id
        @test occursin("RunManifest", sprint(show, manifest))
        @test_throws ArgumentError run_manifest(
            label = "", dataset = "x", created_at = DateTime(2026, 1, 1),
        )
        @test_throws ArgumentError run_manifest(
            label = "x", dataset = "", created_at = DateTime(2026, 1, 1),
        )
    end
end

@testset "numerical stability" begin
    @testset "a degenerate series is refused, not absorbed" begin
        # This found a real one. `Inf > 0` is true, so a positivity check alone accepted an
        # infinite price, which then satisfied high >= low and low <= open <= high and
        # constructed without complaint. The same defect was fixed for `Quote` in the earlier
        # audit and missed here, so an infinite bar could enter through the store even though
        # it could not enter through the feed.
        for bad in (0.0, -1.0, Inf, -Inf, NaN)
            @test_throws ArgumentError Bar(
                "N", DateTime(2026, 1, 2), bad, bad, bad, bad, 1000.0,
            )
        end
        # And one bad field is enough, in any position.
        @test_throws ArgumentError Bar(
            "N", DateTime(2026, 1, 2), 100.0, Inf, 99.0, 100.0, 1000.0,
        )
        @test_throws ArgumentError Bar(
            "N", DateTime(2026, 1, 2), Inf, 101.0, 99.0, 100.0, 1000.0,
        )
        @test_throws ArgumentError Bar(
            "N", DateTime(2026, 1, 2), 100.0, 101.0, 99.0, 100.0, Inf,
        )
        @test_throws ArgumentError Bar(
            "N", DateTime(2026, 1, 2), 100.0, 101.0, 99.0, 100.0, NaN,
        )
        @test_throws ArgumentError Bar(
            "N", DateTime(2026, 1, 2), 100.0, 99.0, 101.0, 100.0, 1000.0,
        )   # high below low
        # A valid bar still constructs, which is the half that stops an over-tight guard.
        @test Bar("N", DateTime(2026, 1, 2), 100.0, 101.0, 99.0, 100.5, 0.0) isa Bar
    end

    @testset "a variance of zero does not produce an infinite anything" begin
        # A flat stretch is an ordinary market, not an error, and the arithmetic has to survive
        # it: a zero variance in a denominator is where a system produces its first NaN.
        filter = DiscountedVarianceFilter(variance_prior(volatility_scale = 0.015))
        for _ in 1:500
            observe_variance!(filter, 0.0)
        end
        @test isfinite(noise_variance(filter))
        @test noise_variance(filter) > 0
        @test isfinite(expected_volatility(filter))
        @test isfinite(volatility_uncertainty(filter))
        @test predictive_df(filter) > 2
    end

    @testset "very small and very large numbers both survive" begin
        prior = weakly_informative_prior(2; residual_scale = 0.02, coefficient_scale = 0.5)
        for magnitude in (1.0e-12, 1.0e-6, 1.0, 1.0e6, 1.0e12)
            model = BayesianLinearModel(prior)
            for index in 1:100
                update!(model, [1.0, magnitude * (index / 100)], magnitude * 0.001)
            end
            distribution, epistemic = predict(model, [1.0, magnitude])
            @test isfinite(mean(distribution))
            @test isfinite(scale(distribution))
            @test scale(distribution) > 0
            @test isfinite(epistemic)
            @test epistemic >= 0
        end
    end

    @testset "one observation and none at all are both answerable" begin
        prior = weakly_informative_prior(1; residual_scale = 0.02)
        empty_model = BayesianLinearModel(prior)
        @test isfinite(mean(predict(empty_model, [1.0])[1]))
        update!(empty_model, [1.0], 0.01)
        @test isfinite(scale(predict(empty_model, [1.0])[1]))

        # And nothing at all to assess is refused rather than answered.
        @test_throws ArgumentError assess(Normal[], Float64[])
    end

    @testset "a very long sequence does not drift into nonsense" begin
        filter = DiscountedVarianceFilter(variance_prior(volatility_scale = 0.015))
        generator = MersenneTwister(99)
        for _ in 1:50_000
            observe_variance!(filter, (0.015 * randn(generator))^2)
        end
        @test isfinite(noise_variance(filter))
        @test 0.005 < sqrt(noise_variance(filter)) < 0.05
        @test predictive_df(filter) > 2
        @test sum(filter.weights) > 0
    end

    @testset "a non-finite feature never reaches a model" begin
        for bad in (NaN, Inf, -Inf)
            model = BayesianLinearModel(
                weakly_informative_prior(2; residual_scale = 0.02),
            )
            @test_throws ArgumentError predict(model, [1.0, bad])
        end
    end
end

@testset "model health, as distinct from process health" begin
    healthy_arguments = (
        as_of = DateTime(2026, 8, 25, 15, 30), fitted = true, uncertainty = 0.2,
        n_observations = 500, disagreement_share = 0.1,
    )

    @testset "a working model is reported working" begin
        health = assess_model_health(; healthy_arguments...)
        @test trustworthy(health)
        @test isempty(model_problems(health))
        @test occursin("HEALTHY", model_health_report(health))
        @test occursin("HEALTHY", sprint(show, health))
    end

    @testset "a posterior that has come apart is caught, not compared" begin
        # The state a threshold comparison lets through: NaN is not greater than any bound, so
        # a check written as a comparison passes a model that has stopped being a model.
        for broken in (NaN, Inf, -Inf)
            health = assess_model_health(;
                healthy_arguments..., uncertainty = broken,
            )
            @test !trustworthy(health)
            @test :posterior in Set(check.name for check in model_problems(health))
        end
    end

    @testset "an unfitted model is not a healthy one" begin
        health = assess_model_health(; healthy_arguments..., fitted = false)
        @test !trustworthy(health)
        @test :fitted in Set(check.name for check in model_problems(health))
    end

    @testset "too little evidence and too much disagreement both fail" begin
        thin = assess_model_health(; healthy_arguments..., n_observations = 5)
        @test !trustworthy(thin)
        @test :evidence in Set(check.name for check in model_problems(thin))

        split = assess_model_health(; healthy_arguments..., disagreement_share = 0.97)
        @test !trustworthy(split)
        @test :agreement in Set(check.name for check in model_problems(split))
    end

    @testset "absent is recorded as absent, never as passing" begin
        health = assess_model_health(
            as_of = DateTime(2026, 8, 25, 15, 30), fitted = true,
        )
        skipped = Set(
            check.name for check in health.checks if check.status === SKIPPED
        )
        @test :calibration in skipped
        @test :posterior in skipped
        @test :evidence in skipped
        @test :agreement in skipped
        for check in health.checks
            check.status === SKIPPED && @test isnan(check.observed)
        end
        # Nothing measured is not the same as nothing wrong, but it is also not a failure: the
        # session's own health checks are what refuse to start a model that has no history.
        @test trustworthy(health)
    end

    @testset "a real calibration report drives the calibration checks" begin
        series = generate_series(
            AR1Returns(phi = 0.4, annual_drift = 0.05);
            symbol = "MH", n_bars = 800, seed = 17, start = Date(2024, 1, 2),
        )
        engine = FeatureEngine(
            InMemoryBarStore(series.bars),
            FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
        )
        examples = build_training_set(engine, "MH"; horizon_bars = 1)
        report = replay(
            (() -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),),
            examples; config = ReplayConfig(warmup = 400, refit_every = 50),
        )
        health = assess_model_health(
            as_of = last(examples).features.as_of, fitted = true,
            calibration = report.calibration, uncertainty = 0.2,
            n_observations = length(report.records),
        )
        @test trustworthy(health)
        names = Set(check.name for check in health.checks)
        @test :calibration in names
        @test :accuracy in names
        for check in health.checks
            check.status === SKIPPED && continue
            @test isfinite(check.observed)
        end
    end
end
