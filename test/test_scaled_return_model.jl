# The volatility-scaled return model: one set of coefficients describing a market whose bars
# change width, and the arithmetic that makes that possible written once.

const SR_COLUMN = :volatility_20

function sr_examples(; n_bars = 2_000, seed = 5, horizon = 1, moving = true)
    process = moving ?
        StochasticVolatilityReturns(
            annual_drift = 0.0, annual_volatility = 0.3,
            persistence = 0.97, volatility_of_volatility = 0.2,
        ) : GaussianReturns(annual_drift = 0.0, annual_volatility = 0.3)
    series = generate_series(
        process; symbol = "RELIANCE", n_bars = n_bars, seed = seed,
        start = Date(2012, 1, 1),
    )
    engine = FeatureEngine(
        InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    return build_training_set(engine, "RELIANCE"; horizon_bars = horizon)
end

sr_model(; policy = ConstantScale(), horizon = 1) =
    BayesianReturnModel([:log_return_1]; horizon_bars = horizon, policy = policy)

sr_scaled(; kwargs...) = sr_model(; policy = VolatilityScale(SR_COLUMN), kwargs...)

function sr_fitted(model, examples)
    fit!(model, examples)
    return model
end

sr_predict(model, features) =
    predict(
    model, features; symbol = "RELIANCE", as_of = features.as_of,
    horizon_bars = model.horizon_bars,
)

function sr_vector(template::FeatureVector; kwargs...)
    values = copy(template.values)
    for (name, value) in kwargs
        values[name] = value
    end
    return FeatureVector(
        symbol = template.symbol, as_of = template.as_of,
        data_as_of = template.data_as_of, values = values,
        n_bars = template.n_bars,
    )
end

@testset "response scale policies" begin
    @testset "the plain model is unscaled and stays that way" begin
        example = first(sr_examples())
        @test response_scale(ConstantScale(), example.features) === 1.0
        @test response_scale(sr_model(), example.features) === 1.0
        @test policy_columns(ConstantScale()) == Symbol[]
        @test model_semver(sr_model()) == v"0.1.0"
    end

    @testset "a volatility scale reads a feature and floors it" begin
        template = last(sr_examples()).features
        policy = VolatilityScale(SR_COLUMN; floor = 1.0e-3)
        @test policy_columns(policy) == [SR_COLUMN]
        @test model_semver(sr_scaled()) == v"0.2.0"

        # The feature is annualised, and the response is a one-bar return.
        annual = require(template, SR_COLUMN)
        @test response_scale(policy, template) ≈ deannualise(annual)
        @test response_scale(
            VolatilityScale(SR_COLUMN; annualised = false), template,
        ) ≈ annual

        # A flat window prints exactly zero, and zero is not a scale.
        @test response_scale(policy, sr_vector(template; volatility_20 = 0.0)) == 1.0e-3
        @test_throws ArgumentError VolatilityScale(SR_COLUMN; floor = 0.0)
        @test_throws ArgumentError response_scale(
            policy, sr_vector(template; volatility_20 = NaN),
        )
    end

    @testset "the scaling multiplies the predictive, and does not divide it" begin
        # An inversion here leaves the model looking plausible on average and wrong in
        # every regime, which is exactly the failure that would survive a hit-rate check.
        examples = sr_examples()
        model = sr_fitted(sr_scaled(), examples)
        template = last(examples).features

        quiet = sr_predict(model, sr_vector(template; volatility_20 = 0.2))
        loud = sr_predict(model, sr_vector(template; volatility_20 = 0.4))

        @test std(loud.distribution) / std(quiet.distribution) ≈ 2.0 rtol = 1.0e-12
        @test mean(loud.distribution) / mean(quiet.distribution) ≈ 2.0 rtol = 1.0e-12
        @test loud.epistemic_variance / quiet.epistemic_variance ≈ 4.0 rtol = 1.0e-12
        @test dof(loud.distribution.ρ) === dof(quiet.distribution.ρ)
        @test loud.diagnostics[:response_scale] ≈ 2 * quiet.diagnostics[:response_scale]
    end

    @testset "a vector without the scaling column is refused before predict" begin
        # The gap the plain model leaves open: can_predict looked only at the model's own
        # features, so a missing scaling column surfaced as an exception inside the loop.
        examples = sr_examples()
        model = sr_fitted(sr_scaled(), examples)
        template = last(examples).features
        without = FeatureVector(
            symbol = template.symbol, as_of = template.as_of,
            data_as_of = template.data_as_of,
            values = Dict{Symbol, Float64}(
                :log_return_1 => require(template, :log_return_1),
            ),
            n_bars = template.n_bars,
        )
        @test can_predict(model, template)
        @test !can_predict(model, without)
        @test can_predict(sr_fitted(sr_model(), examples), without)
        @test_throws KeyError sr_predict(model, without)
    end

    @testset "policy is part of the model's identity" begin
        # Same regression state, different predictions. They must not claim the same
        # version or the same hash.
        examples = sr_examples()
        plain = sr_fitted(sr_model(), examples)
        scaled = sr_fitted(sr_scaled(), examples)
        @test params_hash(plain) != params_hash(scaled)
        @test model_semver(plain) != model_semver(scaled)
        @test identifier(model_version(plain)) != identifier(model_version(scaled))
        @test parameters(scaled)["policy"]["kind"] == "volatility"
        @test parameters(plain)["policy"]["kind"] == "constant"

        floored = sr_fitted(
            sr_model(policy = VolatilityScale(SR_COLUMN; floor = 0.5)), examples,
        )
        @test params_hash(floored) != params_hash(scaled)
    end

    @testset "scaling earns its place on a market that changes width" begin
        # The whole point. One set of coefficients, two regimes, measured out of sample.
        examples = sr_examples()
        config = WalkForwardConfig(initial_train = 500, refit_every = 20)
        plain = walk_forward(() -> sr_model(), examples, config)
        scaled = walk_forward(() -> sr_scaled(), examples, config)

        plain_report = assess(predictives(plain), outcomes(plain))
        scaled_report = assess(predictives(scaled), outcomes(scaled))

        @test interval_calibration_error(scaled_report) <
            interval_calibration_error(plain_report)
        @test scaled_report.mean_log_score > plain_report.mean_log_score + 0.3
    end

    @testset "and costs almost nothing when the width never changes" begin
        examples = sr_examples(; moving = false)
        config = WalkForwardConfig(initial_train = 500, refit_every = 20)
        plain = walk_forward(() -> sr_model(), examples, config)
        scaled = walk_forward(() -> sr_scaled(), examples, config)
        plain_report = assess(predictives(plain), outcomes(plain))
        scaled_report = assess(predictives(scaled), outcomes(scaled))
        @test scaled_report.mean_log_score > plain_report.mean_log_score - 0.1
        @test interval_calibration_error(scaled_report) < 0.05
    end

    @testset "a scaled model persists as what it is" begin
        mktempdir() do dir
            examples = sr_examples(; n_bars = 1_200)
            original = sr_fitted(sr_scaled(), examples)
            path = joinpath(dir, "scaled.json")
            save_model(original, path)
            restored = load_model(path)

            @test restored isa BayesianReturnModel{VolatilityScale}
            @test restored.policy.column === SR_COLUMN
            @test restored.policy.floor ≈ original.policy.floor
            @test restored.policy.annualised == original.policy.annualised
            @test params_hash(restored) == params_hash(original)

            template = last(examples).features
            @test std(sr_predict(restored, template).distribution) ≈
                std(sr_predict(original, template).distribution)

            bundle = JSON3.read(read(path, String), Dict{String, Any})
            bundle["config"]["policy"]["kind"] = "quadratic"
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)
        end
    end

    @testset "a file written before policies existed is an unscaled model" begin
        # Absence means constant, not an error, which is what keeps every bundle already on
        # disk loading unchanged.
        mktempdir() do dir
            path = joinpath(dir, "old.json")
            save_model(sr_fitted(sr_model(), sr_examples(; n_bars = 1_200)), path)
            bundle = JSON3.read(read(path, String), Dict{String, Any})
            delete!(bundle["config"], "policy")
            write(path, JSON3.write(bundle))

            restored = load_model(path)
            @test restored isa BayesianReturnModel{ConstantScale}
            @test model_semver(restored) == v"0.1.0"
        end
    end
end

@testset "integrity findings deferred from the volatility audit" begin
    @testset "the horizon is inside the parameter hash" begin
        examples = sr_examples(; horizon = 1)
        one_bar = sr_fitted(sr_model(), examples)
        five_bar = sr_fitted(sr_model(horizon = 5), sr_examples(; horizon = 5))
        @test params_hash(one_bar) != params_hash(five_bar)
        @test parameters(one_bar)["horizon_bars"] == 1
        @test parameters(five_bar)["horizon_bars"] == 5

        volatility = BayesianVolatilityModel(; horizon_bars = 1)
        fit!(volatility, sr_examples(; horizon = 1))
        longer = BayesianVolatilityModel(; horizon_bars = 5)
        fit!(longer, sr_examples(; horizon = 5))
        @test parameters(volatility)["horizon_bars"] !=
            parameters(longer)["horizon_bars"]
    end

    @testset "a bar the model has already passed is refused" begin
        # Absorbing one twice counts it twice; absorbing an older one rewinds the window
        # the model reports having been fitted over. Neither raises anything on its own.
        examples = sr_examples()
        for model in (
                sr_fitted(sr_model(), examples[1:500]),
                fit!(BayesianVolatilityModel(; horizon_bars = 1), examples[1:500]),
            )
            @test_throws ArgumentError update!(model, examples[500])
            @test_throws ArgumentError update!(model, examples[10])
            @test n_observations(model) == 500
            update!(model, examples[501])
            @test n_observations(model) == 501
        end
    end

    @testset "a source can read a column that is not the default" begin
        examples = sr_examples()
        model = BayesianVolatilityModel(
            SquaredReturnSource(SR_COLUMN); horizon_bars = 1, annual_volatility = 3.0,
        )
        fit!(model, examples)
        @test feature_names(model) == [SR_COLUMN]
        @test source_columns(model.source) == [SR_COLUMN]
        @test parameters(model)["columns"] == ["volatility_20"]
        # Reading a volatility column as though it were a return is nonsense as a model and
        # is not the point: what is asserted is that the column is honoured, not defaulted.
        @test n_absorbed(model.filter) == length(examples)

        mktempdir() do dir
            path = joinpath(dir, "other.json")
            save_model(model, path)
            @test feature_names(load_model(path)) == [SR_COLUMN]
        end
    end

    @testset "one row out of place is enough to be refused" begin
        # The check has to look at every pair, not only at the endpoints.
        examples = sr_examples()[1:200]
        swapped = copy(examples)
        swapped[100], swapped[101] = swapped[101], swapped[100]
        @test_throws ArgumentError fit!(BayesianVolatilityModel(; horizon_bars = 1), swapped)
        @test first(swapped).features.as_of < last(swapped).features.as_of
    end

    @testset "the integrity check cannot be turned off by deleting a line" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model(sr_fitted(sr_model(), sr_examples(; n_bars = 1_200)), path)
            bundle = JSON3.read(read(path, String), Dict{String, Any})
            delete!(bundle["model"], "params_hash")
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)
        end
    end

    @testset "a count cannot be negative" begin
        mktempdir() do dir
            path = joinpath(dir, "volatility.json")
            volatility = BayesianVolatilityModel(; horizon_bars = 1)
            fit!(volatility, sr_examples(; horizon = 1))
            save_model(volatility, path)
            original = read(path, String)

            for damage in (
                    bundle -> (bundle["model"]["n_observations"] = -5),
                    bundle -> (bundle["state"]["n_seen"] = -1),
                    bundle -> (bundle["state"]["n_skipped"] = -1),
                )
                bundle = JSON3.read(original, Dict{String, Any})
                damage(bundle)
                write(path, JSON3.write(bundle))
                @test_throws ModelFileError load_model(path)
            end
        end
    end

    @testset "the skip counter survives a round trip" begin
        mktempdir() do dir
            examples = sr_examples()[1:400]
            blanked = TrainingExample[
                TrainingExample(
                        FeatureVector(
                            symbol = example.features.symbol, as_of = example.features.as_of,
                        ),
                        example.label,
                    ) for example in examples[301:400]
            ]
            model = BayesianVolatilityModel(; horizon_bars = 1)
            fit!(model, vcat(examples[1:300], blanked))
            @test n_skipped(model.filter) == 100

            path = joinpath(dir, "skipped.json")
            save_model(model, path)
            restored = load_model(path)
            @test n_skipped(restored.filter) == 100
            @test params_hash(restored) == params_hash(model)
        end
    end
end
