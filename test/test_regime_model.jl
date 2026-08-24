# The regime model: the filter wearing the model interface, and the evidence that the existing
# scoring machinery needs no knowledge of it.

function rm_examples(; n_bars = 4_000, seed = 11, horizon = 1, process = RegimeSwitchingReturns())
    series = generate_series(
        process; symbol = "RELIANCE", n_bars = n_bars, seed = seed, start = Date(2005, 1, 1),
    )
    engine = FeatureEngine(InMemoryBarStore(series.bars), FeatureSet(Feature[LogReturn(1)]))
    return build_training_set(engine, "RELIANCE"; horizon_bars = horizon), series
end

rm_model(; kwargs...) = MarketRegimeModel(; horizon_bars = 1, kwargs...)

function rm_fitted(; examples = first(rm_examples()), kwargs...)
    model = rm_model(; kwargs...)
    fit!(model, examples)
    return model
end

rm_predict(model, example) = predict(
    model, example.features; symbol = "RELIANCE",
    as_of = example.features.as_of, horizon_bars = model.horizon_bars,
)

@testset "regime model" begin
    @testset "the model contract" begin
        examples, _ = rm_examples()
        model = rm_model()
        @test !is_fitted(model)
        @test uncertainty(model) === Inf
        @test params_hash(model) === nothing
        @test_throws NotFittedError rm_predict(model, first(examples))
        @test_throws NotFittedError update!(model, first(examples))

        fitted = rm_fitted(; examples = examples)
        @test model_name(fitted) === REGIME
        @test model_semver(fitted) == v"0.1.0"
        @test occursin(r"^regime@0\.1\.0\+[0-9a-f]{8}$", identifier(model_version(fitted)))
        @test 0 <= uncertainty(fitted) <= 1
        @test occursin("MarketRegimeModel", sprint(show, fitted))

        @test_throws ArgumentError fit!(rm_model(), examples[1:10])
        @test_throws ArgumentError fit!(rm_model(), reverse(examples[1:300]))
        @test_throws HorizonMismatchError fit!(
            rm_model(), first(rm_examples(; horizon = 5)),
        )
        @test_throws ArgumentError MarketRegimeModel(; horizon_bars = 0)
    end

    @testset "a refused refit leaves the model as it was" begin
        examples, _ = rm_examples()
        model = rm_fitted(; examples = examples[1:1_000])
        before = params_hash(model)
        @test_throws ArgumentError fit!(model, reverse(examples[1_001:1_500]))
        @test params_hash(model) == before
        @test n_observations(model) == 1_000
    end

    @testset "counters count and a stale bar is refused" begin
        examples, _ = rm_examples()
        model = rm_fitted(; examples = examples[1:1_000])
        for example in examples[1_001:1_003]
            update!(model, example)
        end
        @test n_observations(model) == 1_003
        @test fit_state(model).train_end == examples[1_003].features.as_of
        @test_throws ArgumentError update!(model, examples[1_003])
        @test_throws ArgumentError update!(model, examples[500])
        @test n_observations(model) == 1_003
    end

    @testset "predict reads the belief and nothing else" begin
        examples, _ = rm_examples()
        model = rm_fitted(; examples = examples)
        example = last(examples)
        belief = regime_probabilities(model)

        baseline = rm_predict(model, example)
        blank = FeatureVector(
            symbol = example.features.symbol, as_of = example.features.as_of,
        )
        other = predict(
            model, blank; symbol = "RELIANCE", as_of = example.features.as_of,
            horizon_bars = 1,
        )
        @test mean(other.distribution) == mean(baseline.distribution)
        @test other.epistemic_variance == baseline.epistemic_variance
        @test regime_probabilities(model) == belief
        @test can_predict(model, blank)
    end

    @testset "the diagnostics carry the regime posterior" begin
        result = rm_predict(rm_fitted(), last(first(rm_examples())))
        @test Set(keys(result.diagnostics)) == Set(
            [
                :bull, :bear, :sideways, :confidence, :persistence,
                :expected_duration, :epistemic_share, :state_share,
            ],
        )
        mass = result.diagnostics[:bull] + result.diagnostics[:bear] +
            result.diagnostics[:sideways]
        @test mass ≈ 1.0
        @test 0 <= result.diagnostics[:confidence] <= 1
        @test result.diagnostics[:expected_duration] > 1
        @test 0 < result.diagnostics[:epistemic_share] < 1
        @test result.diagnostics[:state_share] <= result.diagnostics[:epistemic_share]

        model = rm_fitted()
        belief = regime_belief(model)
        @test belief isa LabelledCategorical
        @test belief.labels == collect(REGIME_STATES)
        @test uncertainty(model) ≈ normalised_entropy(belief)
        @test most_likely_regime(model) in REGIME_STATES
    end

    @testset "a bar it cannot read decays the belief rather than informing it" begin
        examples, _ = rm_examples()
        blanked = TrainingExample[
            TrainingExample(
                    FeatureVector(
                        symbol = example.features.symbol, as_of = example.features.as_of,
                    ),
                    example.label,
                ) for example in examples[501:800]
        ]
        model = rm_fitted(; examples = vcat(examples[1:500], blanked))
        @test n_skipped(model.filter) == 300
        @test n_absorbed(model.filter) == 500
        @test n_observations(model) == 800
        @test regime_probabilities(model) ≈ model.prior.stationary atol = 1.0e-4

        all_blank = TrainingExample[
            TrainingExample(
                    FeatureVector(
                        symbol = example.features.symbol, as_of = example.features.as_of,
                    ),
                    example.label,
                ) for example in examples[1:500]
        ]
        @test_throws ArgumentError fit!(rm_model(), all_blank)
    end

    @testset "scored by the existing machinery, unmodified" begin
        # Neither walk_forward.jl nor calibration.jl is touched by this change. The
        # predictive is over the forward return, so the outcome they already grade against
        # is the right one.
        examples, _ = rm_examples()
        config = WalkForwardConfig(initial_train = 500, refit_every = 5)
        records = walk_forward(() -> rm_model(), examples, config)
        report = assess(predictives(records), outcomes(records))

        @test all(record -> record.model.name === REGIME, records)
        @test interval_calibration_error(report) < 0.05
        @test report.pit_ks_statistic < 0.05
        for record in records[1:100:end]
            @test isfinite(logpdf(record.predictive, record.outcome))
            @test 0 < cdf(record.predictive, record.outcome) < 1
        end

        # Against an iid Student-t fitted on the same warm-up window.
        returns = Float64[
            require(example.features, :log_return_1) for example in examples[1:500]
        ]
        flat = [
            student_t(mean(returns), std(returns) * sqrt(3 / 5), 5.0) for _ in records
        ]
        baseline = assess(flat, outcomes(records))
        @test report.mean_log_score > baseline.mean_log_score + 0.02

        # The directional half is uninformative even though the predictive is not centred.
        @test report.brier_score ≈ 0.25 atol = 0.01
    end

    @testset "it does not manufacture regimes on a market that has none" begin
        examples, _ = rm_examples(;
            process = GaussianReturns(annual_drift = 0.0, annual_volatility = 0.25),
        )
        model = rm_fitted(; examples = examples)
        @test uncertainty(model) > 0.6
        @test regime_confidence(model) < 0.4
    end

    @testset "it identifies the regime it was never told" begin
        examples, series = rm_examples(; n_bars = 6_000)
        truth = true_states(series)
        # Aligned by timestamp rather than by arithmetic. The row count differs from the bar
        # count by both the leading bar that has no return and the trailing bar that has no
        # label, so a single offset conflates the two and would silently shift the whole
        # comparison by one bar, which is exactly how a lookahead hides inside a
        # plausible-looking accuracy number.
        at = Dict{DateTime, Int}(
            bar.timestamp => index for (index, bar) in enumerate(series.bars)
        )
        @test at[first(examples).features.as_of] == 2

        model = rm_fitted(; examples = examples[1:3_000])
        called = Int[]
        actual = Int[]
        for index in 3_001:length(examples)
            push!(called, argmax(regime_probabilities(model)))
            push!(actual, truth[at[examples[index].features.as_of]])
            update!(model, examples[index])
        end
        majority = maximum([mean(actual .== state) for state in 1:N_REGIMES])
        @test mean(called .== actual) > majority + 0.1
    end
end

@testset "regime model persistence" begin
    @testset "a reloaded model is the same model and still learns" begin
        mktempdir() do dir
            examples, _ = rm_examples(; n_bars = 2_000)
            original = rm_fitted(; examples = examples[1:(end - 20)])
            path = joinpath(dir, "regime.json")
            save_model(original, path)
            restored = load_model(path)

            @test restored isa MarketRegimeModel
            @test params_hash(restored) == params_hash(original)
            @test regime_probabilities(restored) == regime_probabilities(original)
            @test restored.filter.parameters.persistence ≈
                original.filter.parameters.persistence

            for example in examples[(end - 19):end]
                update!(original, example)
                update!(restored, example)
            end
            @test regime_probabilities(restored) == regime_probabilities(original)
            @test params_hash(restored) == params_hash(original)
        end
    end

    @testset "the transition matrix is rebuilt, never stored" begin
        mktempdir() do dir
            path = joinpath(dir, "regime.json")
            save_model(rm_fitted(; examples = first(rm_examples(; n_bars = 1_500))), path)
            bundle = JSON3.read(read(path, String))
            @test !haskey(bundle["config"], "transitions")
            @test !haskey(bundle["parameters"], "transitions")

            restored = load_model(path)
            @test transition_matrix(restored.filter.parameters) ≈
                regime_transition(
                restored.filter.parameters.persistence, restored.prior.stationary,
            )
        end
    end

    @testset "a corrupt regime file is caught at load" begin
        mktempdir() do dir
            path = joinpath(dir, "regime.json")
            save_model(rm_fitted(; examples = first(rm_examples(; n_bars = 1_500))), path)
            original = read(path, String)

            for damage in (
                    bundle -> (bundle["state"]["belief"] = [0.5, 0.6, 0.2]),
                    bundle -> (bundle["state"]["belief"] = [0.5, 0.5]),
                    bundle -> (bundle["config"]["source"] = "tick"),
                    bundle -> (bundle["config"]["prior"]["stationary"] = [0.5, 0.5]),
                    bundle -> (bundle["parameters"]["persistence"] = 1.5),
                    bundle -> (bundle["parameters"]["n_rows"] = -1),
                    bundle -> delete!(bundle, "parameters"),
                )
                bundle = JSON3.read(original, Dict{String, Any})
                damage(bundle)
                write(path, JSON3.write(bundle))
                @test_throws ModelFileError load_model(path)
            end
        end
    end

    @testset "three model types share one format" begin
        mktempdir() do dir
            regime_path = joinpath(dir, "regime.json")
            save_model(rm_fitted(; examples = first(rm_examples(; n_bars = 1_500))), regime_path)
            @test load_model(regime_path) isa MarketRegimeModel
            @test JSON3.read(read(regime_path, String))["schema"] == 1
            @test JSON3.read(read(regime_path, String))["model"]["name"] == "regime"
        end
    end

    @testset "an unfitted model is not saved" begin
        mktempdir() do dir
            @test_throws ArgumentError save_model(rm_model(), joinpath(dir, "regime.json"))
        end
    end
end
