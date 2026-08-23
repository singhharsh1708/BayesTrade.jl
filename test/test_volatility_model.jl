# The volatility model, and the first exercise of the model interface by something other than
# the return model.

const VM_ANNUAL = 0.3

function vm_series(; n_bars = 3_000, seed = 5, moving = true)
    process = moving ?
        StochasticVolatilityReturns(
            annual_drift = 0.0, annual_volatility = VM_ANNUAL,
            persistence = 0.97, volatility_of_volatility = 0.15,
        ) : GaussianReturns(annual_drift = 0.0, annual_volatility = VM_ANNUAL)
    return generate_series(
        process; symbol = "RELIANCE", n_bars = n_bars, seed = seed,
        start = Date(2012, 1, 1),
    )
end

function vm_examples(; horizon = 1, kwargs...)
    series = vm_series(; kwargs...)
    engine = FeatureEngine(InMemoryBarStore(series.bars), FeatureSet(Feature[LogReturn(1)]))
    return build_training_set(engine, "RELIANCE"; horizon_bars = horizon)
end

vm_model(; kwargs...) = BayesianVolatilityModel(; horizon_bars = 1, kwargs...)

function vm_fitted(; examples = vm_examples(), kwargs...)
    model = vm_model(; kwargs...)
    fit!(model, examples)
    return model
end

vm_predict(model, example) = predict(
    model, example.features; symbol = "RELIANCE",
    as_of = example.features.as_of, horizon_bars = model.horizon_bars,
)

@testset "volatility model" begin
    @testset "the model contract" begin
        @testset "it refuses to predict before it has been fitted" begin
            model = vm_model()
            example = first(vm_examples())
            @test !is_fitted(model)
            @test !can_predict(model, example.features)
            @test uncertainty(model) === Inf
            @test params_hash(model) === nothing
            @test_throws NotFittedError vm_predict(model, example)
            @test_throws NotFittedError update!(model, example)
            @test occursin(
                "has not been fitted",
                sprint(showerror, NotFittedError("BayesianVolatilityModel")),
            )
        end

        @testset "a fitted model reports who it is" begin
            model = vm_fitted()
            version = model_version(model)
            @test model_name(model) === VOLATILITY
            @test model_semver(model) == v"0.1.0"
            @test length(params_hash(model)) == 32
            @test occursin(r"^volatility@0\.1\.0\+[0-9a-f]{8}$", identifier(version))
            @test occursin("BayesianVolatilityModel", sprint(show, model))
        end

        @testset "absorbing a bar changes the model's identity" begin
            examples = vm_examples()
            model = vm_fitted(; examples = examples)
            before = params_hash(model)
            update!(model, last(examples))
            @test params_hash(model) != before
        end

        @testset "it answers only for the horizon it was fitted for" begin
            model = vm_fitted()
            example = first(vm_examples())
            @test_throws HorizonMismatchError predict(
                model, example.features; symbol = "RELIANCE",
                as_of = example.features.as_of, horizon_bars = 5,
            )
            @test_throws HorizonMismatchError fit!(
                vm_model(), vm_examples(; horizon = 5),
            )
        end

        @testset "too little history is refused rather than fitted blind" begin
            @test_throws ArgumentError fit!(vm_model(), vm_examples()[1:10])
            @test_throws ArgumentError fit!(vm_model(), TrainingExample[])
            @test_throws ArgumentError BayesianVolatilityModel(; horizon_bars = 0)
        end

        @testset "rows out of order are refused" begin
            examples = vm_examples()[1:200]
            @test_throws ArgumentError fit!(vm_model(), reverse(examples))
        end

        @testset "a refused refit leaves the model exactly as it was" begin
            # The dangerous shape: reset the posterior, then throw, and leave FitState
            # describing a fit that no longer exists. require_fitted would pass, predict
            # would answer from the prior, and the answer would be stamped with the old
            # window's row count. It also errs narrow, which is the direction that matters.
            examples = vm_examples()
            model = vm_fitted(; examples = examples[1:500])
            before_hash = params_hash(model)
            before_volatility = expected_volatility(model.filter)
            before_end = fit_state(model).train_end

            @test_throws ArgumentError fit!(model, reverse(examples[501:800]))
            @test params_hash(model) == before_hash
            @test expected_volatility(model.filter) == before_volatility
            @test n_absorbed(model.filter) == 500
            @test n_observations(model) == 500
            @test fit_state(model).train_end == before_end

            @test_throws HorizonMismatchError fit!(model, vm_examples(; horizon = 5)[1:200])
            @test params_hash(model) == before_hash
            @test n_absorbed(model.filter) == 500

            @test_throws ArgumentError fit!(model, examples[1:10])
            @test params_hash(model) == before_hash
        end

        @testset "a window it cannot read at all is not a fit" begin
            # The posterior would be the prior exactly, and stamping that with two hundred
            # observations would put a belief in the record the model never formed.
            blanked = TrainingExample[
                TrainingExample(
                        FeatureVector(
                            symbol = example.features.symbol, as_of = example.features.as_of,
                        ),
                        example.label,
                    ) for example in vm_examples()[1:200]
            ]
            model = vm_model()
            @test_throws ArgumentError fit!(model, blanked)
            @test !is_fitted(model)
        end

        @testset "it absorbs the feature, never the label" begin
            # What makes update! honest in a live loop: the quantity it learns from is one
            # the market has already printed, not one that needs the next h bars to exist.
            examples = vm_examples()[1:300]
            relabelled = TrainingExample[
                TrainingExample(
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
                    ) for example in examples
            ]
            plain = vm_fitted(; examples = examples)
            altered = vm_fitted(; examples = relabelled)
            @test params_hash(altered) == params_hash(plain)
            @test altered.filter.squares == plain.filter.squares
        end

        @testset "the diagnostics are a fixed set" begin
            # Pinned so the dictionary cannot silently gain or lose a key between versions,
            # which anything reading a stored prediction would have no way to notice.
            result = vm_predict(vm_fitted(), last(vm_examples()))
            @test Set(keys(result.diagnostics)) == Set(
                [
                    :bar_volatility, :annualised_volatility, :effective_sample_size,
                    :predictive_df, :epistemic_share, :expected_discount,
                    :discount_entropy, :discount_disagreement,
                ],
            )
            @test result.diagnostics[:annualised_volatility] ≈ VM_ANNUAL rtol = 0.35
            @test 0 < result.diagnostics[:epistemic_share] < 0.5
            @test result.diagnostics[:predictive_df] > 2
            @test result.horizon_bars == 1
            @test result.n_observations == n_observations(vm_fitted())
        end

        @testset "the counters count what they say" begin
            # update! must increment rather than call mark_fitted!, which sets and would
            # clobber a batch count of five hundred down to one.
            examples = vm_examples()
            model = vm_fitted(; examples = examples[1:500])
            @test n_observations(model) == 500
            for example in examples[501:503]
                update!(model, example)
            end
            @test n_observations(model) == 503
            @test fit_state(model).train_end == examples[503].features.as_of
            @test fit_state(model).train_start == examples[1].features.as_of
        end

        @testset "reset returns it to the prior" begin
            model = vm_fitted()
            reset!(model)
            @test !is_fitted(model)
            @test n_observations(model) == 0
            @test n_absorbed(model.filter) == 0
            @test uncertainty(model) === Inf
            @test fit_state(model).fitted_at === nothing
        end
    end

    @testset "predict is a function of the posterior and nothing else" begin
        # Structural, not incidental. A prediction that read the feature vector could see the
        # bar it is being asked about; one that absorbed it would count a bar the live loop
        # has already given it.
        examples = vm_examples()
        model = vm_fitted(; examples = examples)
        example = last(examples)

        weights = copy(model.filter.weights)
        squares = copy(model.filter.squares)
        log_weights = copy(model.filter.log_weights)

        baseline = vm_predict(model, example)
        inflated = FeatureVector(
            symbol = example.features.symbol, as_of = example.features.as_of,
            data_as_of = example.features.data_as_of,
            values = Dict{Symbol, Float64}(
                name => 1_000 * value for (name, value) in example.features.values
            ),
            n_bars = example.features.n_bars,
        )
        empty = FeatureVector(
            symbol = example.features.symbol, as_of = example.features.as_of,
        )

        for vector in (inflated, empty)
            other = predict(
                model, vector; symbol = "RELIANCE",
                as_of = example.features.as_of, horizon_bars = 1,
            )
            @test mean(other.distribution) == mean(baseline.distribution)
            @test std(other.distribution) == std(baseline.distribution)
            @test other.epistemic_variance == baseline.epistemic_variance
        end

        @test model.filter.weights == weights
        @test model.filter.squares == squares
        @test model.filter.log_weights == log_weights
        @test n_absorbed(model.filter) == length(examples)
        @test can_predict(model, empty)
    end

    @testset "a bar it cannot read is skipped, never guessed at" begin
        # A missing feature and a zero return are not the same thing, and the difference is
        # the one a risk system cannot afford to get wrong: absorbing a halted feed as a run
        # of zeros drives the estimate down, and narrow is the dangerous direction.
        examples = vm_examples()[1:400]
        blanked = TrainingExample[
            TrainingExample(
                    FeatureVector(
                        symbol = example.features.symbol, as_of = example.features.as_of,
                    ),
                    example.label,
                ) for example in examples[301:400]
        ]

        read_through = vm_fitted(; examples = examples)
        skipped = vm_fitted(; examples = vcat(examples[1:300], blanked))
        zeroed = vm_fitted(; examples = examples)
        for _ in 1:100
            observe_variance!(zeroed.filter, 0.0)
        end

        @test n_skipped(skipped.filter) == 100
        @test n_absorbed(skipped.filter) == 300
        @test n_observations(skipped) == 400
        @test residual_scale(skipped.filter) > residual_scale(zeroed.filter)
        @test residual_scale(skipped.filter) ≈ sqrt(
            expected_noise_variance(skipped.filter.prior),
        ) rtol = 0.35
        @test isfinite(residual_scale(read_through.filter))
    end

    @testset "scored by the existing machinery, unmodified" begin
        # The claim this whole design rests on: because the predictive is over the forward
        # return, walk_forward and assess need no knowledge of this model at all. Neither
        # file is touched by this change.
        examples = vm_examples()
        records = walk_forward(
            () -> vm_model(), examples,
            WalkForwardConfig(initial_train = 500, refit_every = 1),
        )
        @test length(records) == length(examples) - 500 - 1

        report = assess(predictives(records), outcomes(records))
        @test report.n == length(records)
        @test interval_calibration_error(report) < 0.05
        @test report.pit_ks_statistic < 0.06
        for record in records[1:50:end]
            @test isfinite(logpdf(record.predictive, record.outcome))
            @test isfinite(std(record.predictive))
            @test 0 < cdf(record.predictive, record.outcome) < 1
        end

        @testset "adapting beats never forgetting" begin
            frozen = walk_forward(
                () -> vm_model(discounts = (1.0,)), examples,
                WalkForwardConfig(initial_train = 500, refit_every = 1),
            )
            frozen_report = assess(predictives(frozen), outcomes(frozen))
            @test report.mean_log_score > frozen_report.mean_log_score + 0.15
        end

        @testset "staleness has a price and here it is" begin
            stale = walk_forward(
                () -> vm_model(), examples,
                WalkForwardConfig(initial_train = 500, refit_every = 20),
            )
            @test assess(predictives(stale), outcomes(stale)).mean_log_score <
                report.mean_log_score
        end

        @testset "the directional half carries nothing, on purpose" begin
            # Pinned in CI so a future reader cannot mistake a Brier score of a quarter for
            # a finding. The model forecasts spread; it says nothing about direction.
            for record in records[1:100:end]
                @test probability_positive(record.predictive) ≈ 0.5 atol = 1.0e-9
            end
            @test report.brier_score ≈ 0.25 atol = 1.0e-9
        end
    end

    @testset "a longer horizon is a wider predictive, not a different model" begin
        # Every other test here fits one-bar labels, so the horizon path was never
        # exercised at all: predict could have ignored model.horizon_bars entirely and
        # nothing would have noticed.
        examples = vm_examples(; horizon = 5)
        model = vm_fitted(; examples = examples, horizon_bars = 5)
        result = vm_predict(model, last(examples))
        @test result.horizon_bars == 5

        # The same posterior read at two horizons. A model that ignored its own horizon
        # would emit the one-bar predictive here and this ratio would be one.
        near = return_predictive(model.filter; horizon_bars = 1)
        @test std(result.distribution) / std(near) ≈ sqrt(5) rtol = 1.0e-9
        @test result.epistemic_variance ≈
            5 * variance_inflation(model.filter; horizon_bars = 1) rtol = 1.0e-12
        @test result.diagnostics[:predictive_df] ≈ predictive_df(model.filter)

        @testset "and it is scored at that horizon" begin
            far = walk_forward(
                () -> vm_model(horizon_bars = 5), examples,
                WalkForwardConfig(initial_train = 500, refit_every = 1),
            )
            close = walk_forward(
                () -> vm_model(), vm_examples(; horizon = 1),
                WalkForwardConfig(initial_train = 500, refit_every = 1),
            )
            @test all(record -> record.model.name === VOLATILITY, far)
            @test all(record -> record.realised_at > record.as_of, far)

            far_report = assess(predictives(far), outcomes(far))
            close_report = assess(predictives(close), outcomes(close))
            @test interval_calibration_error(far_report) < 0.1
            @test far_report.sharpness > 1.8 * close_report.sharpness
        end
    end

    @testset "it tracks a market that moves" begin
        moving = vm_fitted(; examples = vm_examples(; moving = true))
        steady = vm_fitted(; examples = vm_examples(; moving = false))
        @test expected_discount(moving.filter) < expected_discount(steady.filter)
        @test uncertainty(moving) > 0
    end

    @testset "uncertainty plateaus rather than vanishing" begin
        # The claim in `uncertainty`'s docstring, and it is not a hedge. A model that
        # discounts is tracking a moving target, so more data buys a better estimate of
        # today and no certainty at all about tomorrow. A model that never forgets does
        # converge, and the contrast is what makes the plateau a measurement rather than
        # an assertion about the code.
        discounting = Float64[]
        never = Float64[]
        for n_bars in (600, 1_500, 4_000)
            examples = vm_examples(; n_bars = n_bars)
            push!(discounting, uncertainty(vm_fitted(; examples = examples)))
            push!(
                never,
                uncertainty(vm_fitted(; examples = examples, discounts = (1.0,))),
            )
        end
        @test last(discounting) > 0.7 * first(discounting)
        @test last(never) < 0.5 * first(never)
        @test last(discounting) > 10 * last(never)
    end
end

@testset "volatility model persistence" begin
    @testset "a reloaded model is the same model and still learns" begin
        mktempdir() do dir
            examples = vm_examples(; n_bars = 1_200)
            original = vm_fitted(; examples = examples)
            path = joinpath(dir, "volatility.json")
            save_model(original, path)
            restored = load_model(path)

            @test restored isa BayesianVolatilityModel
            @test params_hash(restored) == params_hash(original)
            @test model_version(restored).fitted_at == model_version(original).fitted_at

            before = vm_predict(original, last(examples))
            after = vm_predict(restored, last(examples))
            @test mean(after.distribution) == mean(before.distribution)
            @test std(after.distribution) ≈ std(before.distribution)

            for example in examples[(end - 20):end]
                update!(original, example)
                update!(restored, example)
            end
            @test restored.filter.squares == original.filter.squares
            @test params_hash(restored) == params_hash(original)
        end
    end

    @testset "the configuration survives" begin
        mktempdir() do dir
            path = joinpath(dir, "volatility.json")
            original = vm_fitted(;
                examples = vm_examples(; n_bars = 800),
                discounts = (0.8, 0.95), weight_forgetting = 0.9,
                annual_volatility = 0.45, centre = 0.001,
            )
            save_model(original, path)
            restored = load_model(path)
            @test discount_grid(restored.filter) == [0.8, 0.95]
            @test restored.filter.weight_forgetting ≈ 0.9
            @test restored.filter.centre ≈ 0.001
            @test restored.filter.prior.rate ≈ original.filter.prior.rate
            @test feature_names(restored) == [:log_return_1]
        end
    end

    @testset "one format, two model types, no schema bump" begin
        # The name has been written since schema 1 and was simply never read. Reading it is
        # what lets a second model share the format, and every file already on disk keeps
        # loading, which is what this asserts.
        mktempdir() do dir
            volatility_path = joinpath(dir, "volatility.json")
            save_model(vm_fitted(; examples = vm_examples(; n_bars = 800)), volatility_path)

            momentum = BayesianReturnModel([:log_return_1]; horizon_bars = 5)
            fit!(momentum, vm_examples(; horizon = 5, n_bars = 800))
            momentum_path = joinpath(dir, "momentum.json")
            save_model(momentum, momentum_path)

            @test load_model(momentum_path) isa BayesianReturnModel
            @test load_model(volatility_path) isa BayesianVolatilityModel
            @test JSON3.read(read(momentum_path, String))["schema"] ==
                JSON3.read(read(volatility_path, String))["schema"]

            bundle = JSON3.read(read(volatility_path, String), Dict{String, Any})
            bundle["model"]["name"] = "regime"
            write(volatility_path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(volatility_path)
        end
    end

    @testset "a corrupt volatility file is caught at load" begin
        mktempdir() do dir
            path = joinpath(dir, "volatility.json")
            save_model(vm_fitted(; examples = vm_examples(; n_bars = 800)), path)
            original = read(path, String)

            for damage in (
                    bundle -> (bundle["state"]["squares"][1] += 1.0),
                    bundle -> (bundle["config"]["source"] = "range"),
                    bundle -> (bundle["config"]["columns"] = ["a", "b"]),
                    bundle -> (bundle["config"]["centre"] = "zero"),
                    bundle -> delete!(bundle, "state"),
                    bundle -> (bundle["config"]["discounts"] = [1.5]),
                )
                bundle = JSON3.read(original, Dict{String, Any})
                damage(bundle)
                write(path, JSON3.write(bundle))
                @test_throws ModelFileError load_model(path)
            end
        end
    end

    @testset "an unfitted model is not saved" begin
        mktempdir() do dir
            @test_throws ArgumentError save_model(
                vm_model(), joinpath(dir, "volatility.json"),
            )
        end
    end
end
