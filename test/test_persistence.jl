const PERSIST_FEATURES = [:log_return_1]

function persist_examples(; n_bars = 1_500, horizon = 5)
    series = generate_series(
        AR1Returns(phi = 0.35, annual_drift = 0.0);
        symbol = "SYNTH", n_bars = n_bars, seed = 43, start = Date(2015, 1, 1),
    )
    engine = FeatureEngine(InMemoryBarStore(series.bars), FeatureSet(Feature[LogReturn(1)]))
    return build_training_set(engine, "SYNTH"; horizon_bars = horizon)
end

function persist_fitted(; kwargs...)
    model = BayesianReturnModel(PERSIST_FEATURES; horizon_bars = 5, kwargs...)
    fit!(model, persist_examples())
    return model
end

@testset "model persistence" begin
    @testset "a reloaded model predicts identically" begin
        # The property that matters: a saved model is the same model.
        mktempdir() do dir
            original = persist_fitted()
            path = joinpath(dir, "model.json")
            save_model(original, path)
            restored = load_model(path)

            for example in persist_examples()[1:100:end]
                before = predict(
                    original, example.features; symbol = "SYNTH",
                    as_of = example.features.as_of, horizon_bars = 5,
                )
                after = predict(
                    restored, example.features; symbol = "SYNTH",
                    as_of = example.features.as_of, horizon_bars = 5,
                )
                @test mean(after) ≈ mean(before)
                @test std(after) ≈ std(before)
            end
        end
    end

    @testset "the version survives the round trip" begin
        mktempdir() do dir
            original = persist_fitted()
            path = joinpath(dir, "model.json")
            save_model(original, path)
            restored = load_model(path)
            @test model_version(restored) == model_version(original)
            @test params_hash(restored) == params_hash(original)
        end
    end

    @testset "a reloaded model can still be updated" begin
        # Statistics are stored, not the posterior, so learning continues.
        mktempdir() do dir
            examples = persist_examples()
            original = persist_fitted()
            path = joinpath(dir, "model.json")
            save_model(original, path)
            restored = load_model(path)

            before = effective_sample_size(restored.regression)
            for example in examples[1:20]
                update!(original, example)
                update!(restored, example)
            end
            @test effective_sample_size(restored.regression) > before
            @test coefficients(restored.regression) ≈ coefficients(original.regression)
        end
    end

    @testset "the configuration survives" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            original = persist_fitted(
                forgetting = 0.995, residual_scale = 0.05, coefficient_scale = 0.2,
            )
            save_model(original, path)
            restored = load_model(path)
            @test restored.regression.forgetting ≈ 0.995
            @test restored.horizon_bars == 5
            @test restored.feature_names == PERSIST_FEATURES
            @test restored.regression.prior.precision ≈ original.regression.prior.precision
            @test restored.regression.prior.rate ≈ original.regression.prior.rate
        end
    end

    @testset "the file is readable json a person could diff" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model(persist_fitted(), path)
            text = read(path, String)
            @test occursin('\n', text)
            bundle = JSON3.read(text, Dict{String, Any})
            @test bundle["schema"] == 1
            @test bundle["model"]["name"] == "momentum"
            @test bundle["config"]["feature_names"] == ["log_return_1"]
        end
    end

    @testset "missing directories are created" begin
        mktempdir() do dir
            path = joinpath(dir, "nested", "deeper", "model.json")
            save_model(persist_fitted(), path)
            @test isfile(path)
        end
    end

    @testset "an unfitted model is not saved" begin
        mktempdir() do dir
            @test_throws ArgumentError save_model(
                BayesianReturnModel(PERSIST_FEATURES; horizon_bars = 5),
                joinpath(dir, "model.json"),
            )
        end
    end

    @testset "corruption is caught at load, not three layers later" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model(persist_fitted(), path)
            original = read(path, String)

            @test_throws ModelFileError load_model(joinpath(dir, "absent.json"))

            write(joinpath(dir, "bad.json"), "{not json")
            @test_throws ModelFileError load_model(joinpath(dir, "bad.json"))

            # Valid JSON, but not a bundle. Every one of these is a document a parser
            # accepts and a loader must not.
            for text in ("5", "\"a model\"", "[1, 2]", "null")
                write(joinpath(dir, "bad.json"), text)
                @test_throws ModelFileError load_model(joinpath(dir, "bad.json"))
            end

            bundle = JSON3.read(original, Dict{String, Any})
            bundle["schema"] = 99
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)

            bundle = JSON3.read(original, Dict{String, Any})
            delete!(bundle, "scaler")
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)

            bundle = JSON3.read(original, Dict{String, Any})
            bundle["config"]["horizon_bars"] = "five"
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)

            bundle = JSON3.read(original, Dict{String, Any})
            bundle["state"]["xy"] = "not a vector"
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)
        end
    end

    @testset "edited statistics fail the hash check" begin
        # The check that stops a corrupt file from silently trading.
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model(persist_fitted(), path)
            bundle = JSON3.read(read(path, String), Dict{String, Any})
            bundle["state"]["xy"][1] += 1.0
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)
        end
    end

    @testset "a scaler for other features is refused" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model(persist_fitted(), path)
            bundle = JSON3.read(read(path, String), Dict{String, Any})
            bundle["scaler"]["names"] = ["something_else"]
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)
        end
    end

    @testset "a prior of the wrong width is refused" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model(persist_fitted(), path)
            bundle = JSON3.read(read(path, String), Dict{String, Any})
            bundle["config"]["prior"]["mean"] = [0.0, 0.0, 0.0]
            bundle["config"]["prior"]["precision"] = [
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            ]
            write(path, JSON3.write(bundle))
            @test_throws ModelFileError load_model(path)
        end
    end
end

@testset "fit report" begin
    function report_engine(process = GaussianReturns(); n_bars = 1_200, seed = 11)
        series = generate_series(
            process; symbol = "RELIANCE", n_bars = n_bars, seed = seed,
            start = Date(2020, 1, 1),
        )
        return FeatureEngine(InMemoryBarStore(series.bars), minimal_feature_set())
    end

    tight() = WalkForwardConfig(initial_train = 250, refit_every = 20)

    @testset "it scores out of sample before fitting on everything" begin
        report = fit_return_model(
            report_engine(), "RELIANCE"; horizon_bars = 5, config = tight(),
        )
        @test report.symbol == "RELIANCE"
        @test report.n_scored < report.n_examples
        @test n_observations(report.model) == report.n_examples
        @test report.calibration.n == report.n_scored
    end

    @testset "it finds no edge in the synthetic market" begin
        # Independent returns by construction. Every z should be small.
        report = fit_return_model(
            report_engine(), "RELIANCE"; horizon_bars = 5, config = tight(),
        )
        coefficients = coefficient_report(report.model)
        for column in design_columns(report.model)
            column === :intercept && continue
            @test abs(coefficients[column]["z"]) < 4
        end
        @test report.calibration.brier_score ≈ 0.25 atol = 0.03
    end

    @testset "it finds the edge that is there" begin
        report = fit_return_model(
            report_engine(AR1Returns(phi = 0.35, annual_drift = 0.0); n_bars = 3_000),
            "RELIANCE"; horizon_bars = 1, config = tight(),
        )
        @test abs(coefficient_report(report.model)[:log_return_1]["z"]) > 5
        @test report.calibration.brier_score < 0.25
    end

    @testset "the summary reports calibration and coefficients" begin
        summary = summarise(
            fit_return_model(report_engine(), "RELIANCE"; horizon_bars = 5, config = tight()),
        )
        @test occursin("labelled rows", summary)
        @test occursin("interval error", summary)
        @test occursin("standardised", summary)
        @test occursin("log_return_1", summary)
    end

    @testset "too little history is refused rather than fitted blind" begin
        @test_throws ArgumentError fit_return_model(
            report_engine(; n_bars = 300), "RELIANCE"; horizon_bars = 5, config = tight(),
        )
    end

    @testset "the feature set can be narrowed" begin
        report = fit_return_model(
            report_engine(), "RELIANCE"; horizon_bars = 5, config = tight(),
            feature_names = [:log_return_1],
        )
        @test report.model.feature_names == [:log_return_1]
        @test design_columns(report.model) == [:intercept, :log_return_1]
    end
end
