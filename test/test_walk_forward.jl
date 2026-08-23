# Walk-forward evaluation, including the leak it exists to prevent.

const WF_FEATURES = [:log_return_1]
const WF_HORIZON = 5

function wf_examples(
        process = AR1Returns(phi = 0.35, annual_drift = 0.0);
        n_bars = 3_000, horizon = WF_HORIZON, seed = 31,
    )
    series = generate_series(
        process; symbol = "SYNTH", n_bars = n_bars, seed = seed, start = Date(2010, 1, 1),
    )
    engine = FeatureEngine(InMemoryBarStore(series.bars), FeatureSet(Feature[LogReturn(1)]))
    return build_training_set(engine, "SYNTH"; horizon_bars = horizon)
end

wf_factory(horizon = WF_HORIZON) =
    () -> BayesianReturnModel(WF_FEATURES; horizon_bars = horizon)

wf_records(; examples = wf_examples(), kwargs...) = walk_forward(
    wf_factory(), examples,
    WalkForwardConfig(; initial_train = 500, refit_every = 100, kwargs...),
)

@testset "walk-forward" begin
    @testset "embargo" begin
        @testset "no prediction uses a label that had not been realised" begin
            # The leak this harness exists to prevent, asserted on every record.
            for record in wf_records()
                @test record.train_realised_through <= record.as_of
            end
        end

        @testset "the embargo defaults to the label horizon" begin
            examples = wf_examples()
            records = walk_forward(
                wf_factory(), examples,
                WalkForwardConfig(initial_train = 500, refit_every = 100),
            )
            @test first(records).as_of == examples[500 + WF_HORIZON + 1].features.as_of
        end

        @testset "a longer embargo starts later" begin
            early = wf_records()
            late = wf_records(embargo_bars = 50)
            @test first(late).as_of > first(early).as_of
            @test length(late) < length(early)
        end

        @testset "a record cannot be built that violates the embargo" begin
            record = first(wf_records())
            @test_throws ArgumentError PredictionRecord(
                symbol = record.symbol,
                as_of = record.as_of,
                realised_at = record.realised_at,
                predictive = record.predictive,
                outcome = record.outcome,
                model = record.model,
                train_rows = record.train_rows,
                train_realised_through = record.realised_at,
            )
        end
    end

    @testset "stepping" begin
        @testset "it predicts every bar after the warm-up" begin
            examples = wf_examples()
            @test length(wf_records(; examples = examples)) ==
                length(examples) - 500 - WF_HORIZON
        end

        @testset "predictions come out in order and never repeat" begin
            stamps = [record.as_of for record in wf_records()]
            @test issorted(stamps)
            @test allunique(stamps)
        end

        @testset "an expanding window grows and a rolling one does not" begin
            expanding = wf_records()
            rolling = wf_records(max_train = 600)
            @test last(expanding).train_rows > first(expanding).train_rows
            @test maximum(record.train_rows for record in rolling) == 600
            @test all(record -> record.train_rows <= 600, rolling)
        end

        @testset "refitting less often produces fewer distinct models" begin
            examples = wf_examples(; n_bars = 800)
            frequent = walk_forward(
                wf_factory(), examples,
                WalkForwardConfig(initial_train = 300, refit_every = 1),
            )
            occasional = walk_forward(
                wf_factory(), examples,
                WalkForwardConfig(initial_train = 300, refit_every = 100),
            )
            @test length(unique(record.model.params_hash for record in occasional)) <
                length(unique(record.model.params_hash for record in frequent))
        end

        @testset "a history shorter than the warm-up produces nothing" begin
            @test isempty(
                walk_forward(
                    wf_factory(), wf_examples(; n_bars = 300),
                    WalkForwardConfig(initial_train = 5_000),
                ),
            )
            @test isempty(walk_forward(wf_factory(), TrainingExample[]))
        end

        @testset "mixed horizons are refused" begin
            mixed = vcat(
                wf_examples(; horizon = 1)[1:100], wf_examples(; horizon = 5)[1:100],
            )
            @test_throws ArgumentError walk_forward(wf_factory(), mixed)
        end
    end

    @testset "configuration" begin
        @test_throws ArgumentError WalkForwardConfig(initial_train = 1)
        @test_throws ArgumentError WalkForwardConfig(initial_train = 500, max_train = 100)
        @test_throws ArgumentError WalkForwardConfig(embargo_bars = -1)
        @test_throws ArgumentError WalkForwardConfig(refit_every = 0)
    end

    @testset "out-of-sample calibration" begin
        @testset "a correctly specified model is calibrated out of sample" begin
            # The claim the whole project rests on, measured on data the model never saw.
            records = wf_records()
            report = assess(predictives(records), outcomes(records))
            @test report.n == length(records)
            @test interval_calibration_error(report) < 0.05
            @test report.expected_calibration_error < 0.1
        end

        @testset "it finds no edge in an unpredictable market" begin
            records = wf_records(; examples = wf_examples(GaussianReturns(annual_drift = 0.0)))
            report = assess(predictives(records), outcomes(records))
            @test report.brier_score ≈ 0.25 atol = 0.02
            @test interval_calibration_error(report) < 0.05
        end

        @testset "a predictable market is forecast better than a coin" begin
            records = wf_records()
            @test assess(predictives(records), outcomes(records)).brier_score < 0.25
        end

        @testset "the directional probability moves with the signal" begin
            records = wf_records()
            rising = filter(r -> probability_positive(r.predictive) > 0.55, records)
            falling = filter(r -> probability_positive(r.predictive) < 0.45, records)
            @test length(rising) > 50
            @test length(falling) > 50
            @test count(went_up, rising) / length(rising) >
                count(went_up, falling) / length(falling)
        end

        @testset "errors are reported against the predictive mean" begin
            record = first(wf_records())
            @test prediction_error(record) ≈ record.outcome - mean(record.predictive)
        end
    end
end
