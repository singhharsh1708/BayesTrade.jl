# The replay engine. Most of these tests are about what it must NOT do.

function rp_examples(; n_bars = 2_500, seed = 11, horizon = 1, process = RegimeSwitchingReturns())
    series = generate_series(
        process; symbol = "RELIANCE", n_bars = n_bars, seed = seed, start = Date(2005, 1, 1),
    )
    engine = FeatureEngine(
        InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    return build_training_set(engine, "RELIANCE"; horizon_bars = horizon)
end

rp_factories(horizon = 1) = (
    () -> BayesianReturnModel([:log_return_1]; horizon_bars = horizon),
    () -> BayesianVolatilityModel(; horizon_bars = horizon),
    () -> MarketRegimeModel(; horizon_bars = horizon),
)

rp_config(; kwargs...) = ReplayConfig(; warmup = 800, refit_every = 100, kwargs...)

@testset "replay" begin
    examples = rp_examples()
    report = replay(rp_factories(), examples; config = rp_config())

    @testset "it scores every bar after the warm-up" begin
        @test length(report) == length(examples) - 800 - 1
        @test report.n_examples == length(examples)
        @test report.n_skipped == 0
        @test report.symbol == "RELIANCE"
        @test n_models(report.reliability) == 3
        @test occursin("ReplayReport", sprint(show, report))
        @test occursin("model reliability", summarise(report))
        @test occursin("scored bars", summarise(report))
    end

    @testset "no prediction can see the bar it is predicting" begin
        # The failure this whole layer exists to prevent, asserted on every record.
        for record in report.records
            @test record.realised_at > record.as_of
            @test record.prediction.as_of == record.as_of
            @test record.prediction.horizon_bars == 1
        end
        stamps = [record.as_of for record in report.records]
        @test issorted(stamps)
        @test allunique(stamps)
        @test first(stamps) == examples[802].features.as_of
    end

    @testset "changing the future cannot change the past" begin
        # The structural test for look-ahead: corrupt every bar after a point and the
        # records before it must be identical, bit for bit.
        cut = 1_200
        tampered = copy(examples)
        for index in (cut + 1):length(tampered)
            example = tampered[index]
            tampered[index] = TrainingExample(
                example.features,
                Label(
                    symbol = example.label.symbol,
                    as_of = example.label.as_of,
                    realised_at = example.label.realised_at,
                    horizon_bars = example.label.horizon_bars,
                    forward_log_return = 0.5,
                    max_adverse_excursion = -0.5,
                    max_favourable_excursion = 0.5,
                ),
            )
        end
        altered = replay(rp_factories(), tampered; config = rp_config())
        early = cut - 800 - 1
        for index in 1:early
            @test altered.records[index].outcome == report.records[index].outcome
            @test mean(altered.records[index].prediction) ==
                mean(report.records[index].prediction)
            @test altered.records[index].log_score == report.records[index].log_score
        end
    end

    @testset "the same replay twice is the same replay" begin
        again = replay(rp_factories(), examples; config = rp_config())
        @test length(again) == length(report)
        for (left, right) in zip(again.records, report.records)
            @test left.outcome == right.outcome
            @test left.log_score == right.log_score
            @test mean(left.prediction) == mean(right.prediction)
        end
        @test reliabilities(again.reliability) == reliabilities(report.reliability)
    end

    @testset "what it believed was calibrated" begin
        @test report.calibration.n == length(report)
        @test interval_calibration_error(report.calibration) < 0.05
        @test report.calibration.pit_ks_statistic < 0.05
        @test predictives(report.records) == [r.prediction.distribution for r in report.records]
        @test outcomes(report.records) == [r.outcome for r in report.records]
        for record in report.records[1:100:end]
            @test isfinite(record.log_score)
            @test record.log_score ≈ logpdf(record.prediction.distribution, record.outcome)
        end
    end

    @testset "pooling ends up near the best model without being told which" begin
        scores = mean_log_scores(report.reliability)
        pooled = mean(record.log_score for record in report.records)
        @test pooled > mean(scores)
        @test pooled > maximum(scores) - 0.05
        @test sum(reliabilities(report.reliability)) ≈ 1.0
        # The weights moved off equal shares, which is the layer doing anything at all.
        @test maximum(reliabilities(report.reliability)) > 0.4
    end

    @testset "refitting less often is cheaper and no better" begin
        rare = replay(
            rp_factories(), examples;
            config = ReplayConfig(warmup = 800, refit_every = 10_000),
        )
        @test length(rare) == length(report)
        @test mean(record.log_score for record in rare.records) <=
            mean(record.log_score for record in report.records) + 0.05
    end

    @testset "a single model replays too" begin
        one = replay(
            (() -> BayesianVolatilityModel(; horizon_bars = 1),), examples;
            config = rp_config(),
        )
        @test n_models(one.reliability) == 1
        @test reliabilities(one.reliability) ≈ [1.0]
        @test all(record -> n_models(record.prediction) == 1, one.records)
    end

    @testset "a replay that cannot be run is refused" begin
        @test_throws ArgumentError replay((), examples; config = rp_config())
        @test_throws ArgumentError replay(rp_factories(), examples[1:100]; config = rp_config())
        @test_throws ArgumentError replay(
            rp_factories(), reverse(examples); config = rp_config(),
        )
        @test_throws HorizonMismatchError replay(
            rp_factories(5), rp_examples(; horizon = 5);
            config = rp_config(),
        )
        @test_throws ArgumentError ReplayConfig(warmup = 10)
        @test_throws ArgumentError ReplayConfig(refit_every = 0)
        @test_throws ArgumentError ReplayConfig(horizon_bars = 0)
    end

    @testset "a record cannot settle before it was predicted" begin
        record = first(report.records)
        @test_throws ArgumentError ReplayRecord(
            record.prediction, record.as_of, record.outcome,
        )
        @test_throws ArgumentError ReplayRecord(
            record.prediction, record.realised_at, NaN,
        )
    end

    @testset "an outcome is not used before it has happened" begin
        # At a horizon of one bar this is invisible: the outcome of bar t is bar t+1 and
        # settling immediately is settling on time. At five bars it is not. Corrupting one
        # label must leave every prediction made before that label was realised untouched,
        # and scoring it the moment it was forecast would move the very next one.
        horizon = 5
        long = rp_examples(; horizon = horizon)
        config = rp_config(horizon_bars = horizon)
        clean = replay(rp_factories(horizon), long; config = config)

        cut = 1_100
        tampered = copy(long)
        tampered[cut] = TrainingExample(
            long[cut].features,
            Label(
                symbol = long[cut].label.symbol,
                as_of = long[cut].label.as_of,
                realised_at = long[cut].label.realised_at,
                horizon_bars = horizon,
                forward_log_return = 0.9,
                max_adverse_excursion = -0.9,
                max_favourable_excursion = 0.9,
            ),
        )
        altered = replay(rp_factories(horizon), tampered; config = config)
        @test length(altered) == length(clean)

        settles_at = long[cut].label.realised_at
        tampered_at = long[cut].features.as_of
        checked = 0
        for (left, right) in zip(altered.records, clean.records)
            left.as_of < settles_at || break
            # The tampered bar's own record carries the tampered outcome, which is the
            # point of tampering. What must not move is any prediction.
            left.as_of == tampered_at || @test left.outcome == right.outcome
            @test mean(left.prediction) == mean(right.prediction)
            @test var(left.prediction) == var(right.prediction)
            checked += 1
        end
        # The corrupted label sits several bars before it settles, so there are predictions
        # in between that a premature settlement would have moved.
        @test checked > cut - 800 - horizon
        @test any(record -> record.as_of >= settles_at, altered.records)
    end

    @testset "a longer horizon replays at that horizon" begin
        long = replay(
            rp_factories(5), rp_examples(; horizon = 5);
            config = rp_config(horizon_bars = 5),
        )
        @test all(record -> record.prediction.horizon_bars == 5, long.records)
        @test interval_calibration_error(long.calibration) < 0.1
        # Five bars of return is a wider question than one.
        @test long.calibration.sharpness > report.calibration.sharpness
    end
end
