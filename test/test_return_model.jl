# The Bayesian return model, against processes whose predictability is known.
#
# `AR1Returns` has an exactly known one-step relationship: the next log return is `phi` times
# this one plus independent noise. A correct model must recover `phi`. `GaussianReturns` has
# no relationship at all, and a correct model must find nothing.

const PHI = 0.4
const RETURN_FEATURES = [:log_return_1]

function return_examples(
        process = AR1Returns(phi = PHI, annual_drift = 0.0);
        n_bars = 4_000, seed = 17, horizon = 1,
        features = FeatureSet(Feature[LogReturn(1)]),
    )
    series = generate_series(
        process; symbol = "SYNTH", n_bars = n_bars, seed = seed, start = Date(2008, 1, 1),
    )
    engine = FeatureEngine(InMemoryBarStore(series.bars), features)
    return build_training_set(engine, "SYNTH"; horizon_bars = horizon), engine
end

function return_fitted(process = AR1Returns(phi = PHI, annual_drift = 0.0); kwargs...)
    examples, engine = return_examples(process)
    model = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1, kwargs...)
    fit!(model, examples)
    return model, examples, engine
end

@testset "feature scaler" begin
    scalerdesign(rows = 200; seed = 3) = hcat(
        randn(Xoshiro(seed), rows),
        5.0 .+ 0.1 .* randn(Xoshiro(seed + 1), rows),
        -2.0 .+ 10.0 .* randn(Xoshiro(seed + 2), rows),
    )
    SCALER_NAMES = [:a, :b, :c]

    @testset "it centres and scales the training window" begin
        design = scalerdesign()
        scaled = transform(fit_scaler(SCALER_NAMES, design), design)
        for column in 1:3
            @test mean(view(scaled, :, column)) ≈ 0.0 atol = 1.0e-12
            @test std(view(scaled, :, column)) ≈ 1.0
        end
    end

    @testset "a constant column is given unit scale rather than zero" begin
        design = hcat(ones(50), scalerdesign(50)[:, 1:2])
        scaler = fit_scaler(SCALER_NAMES, design)
        @test scaler.scales[1] == 1.0
        @test all(≈(0.0), view(transform(scaler, design), :, 1))
    end

    @testset "degenerate inputs are rejected" begin
        @test_throws ArgumentError fit_scaler(SCALER_NAMES, scalerdesign(1))
        @test_throws ArgumentError fit_scaler(SCALER_NAMES, scalerdesign()[:, 1:2])
        @test_throws ArgumentError FeatureScaler(SCALER_NAMES, zeros(3), zeros(3))
    end

    @testset "later rows use the training statistics, not their own" begin
        # The whole point: a later window must not recentre itself.
        scaler = fit_scaler(SCALER_NAMES, scalerdesign(200; seed = 1))
        later = scalerdesign(200; seed = 20) .+ 3.0
        scaled = transform(scaler, later)
        @test any(column -> abs(mean(view(scaled, :, column))) > 0.1, 1:3)
    end

    @testset "drifting out of the training range produces a large value" begin
        scaler = fit_scaler(SCALER_NAMES, scalerdesign())
        @test abs(transform_row(scaler, [50.0, 5.0, -2.0])[1]) > 10
    end

    @testset "transforming a row agrees with transforming a matrix" begin
        design = scalerdesign()
        scaler = fit_scaler(SCALER_NAMES, design)
        @test transform_row(scaler, design[7, :]) ≈ transform(scaler, design)[7, :]
        @test_throws ArgumentError transform_row(scaler, zeros(2))
    end

    @testset "a prediction is unchanged by the unscaling round trip" begin
        design = scalerdesign()
        scaler = fit_scaler(SCALER_NAMES, design)
        standardised = [0.4, -0.2, 0.7]
        row = design[11, :]
        through = dot(transform_row(scaler, row), standardised)
        direct = dot(row .- scaler.centres, unscale_coefficients(scaler, standardised))
        @test through ≈ direct
        @test_throws ArgumentError unscale_coefficients(scaler, zeros(2))
    end

    @testset "parameters record the frozen statistics" begin
        parameters = BayesTrade.parameters(fit_scaler(SCALER_NAMES, scalerdesign()))
        @test parameters["names"] == ["a", "b", "c"]
        @test length(parameters["centres"]) == 3
    end
end

@testset "bayesian return model" begin
    @testset "recovery" begin
        @testset "it recovers the autoregressive coefficient" begin
            model, _, _ = return_fitted()
            @test coefficient_report(model)[:log_return_1]["raw"] ≈ PHI atol = 0.05
        end

        @testset "it finds nothing in an unpredictable market" begin
            # The null case. A model that finds an edge here is fitting noise.
            model, _, _ = return_fitted(GaussianReturns(annual_drift = 0.0))
            @test abs(coefficient_report(model)[:log_return_1]["z"]) < 3
        end

        @testset "it distinguishes momentum from mean reversion" begin
            momentum, _, _ = return_fitted(AR1Returns(phi = 0.4, annual_drift = 0.0))
            reverting, _, _ = return_fitted(AR1Returns(phi = -0.4, annual_drift = 0.0))
            @test coefficient_report(momentum)[:log_return_1]["raw"] > 0.3
            @test coefficient_report(reverting)[:log_return_1]["raw"] < -0.3
        end

        @testset "the intercept absorbs the drift" begin
            model, _, _ = return_fitted(AR1Returns(phi = 0.0, annual_drift = 0.5))
            expected = 0.5 / BARS_PER_YEAR
            @test coefficient_report(model)[:intercept]["standardised"] ≈ expected atol =
                expected
        end

        @testset "it recovers the residual scale" begin
            process = AR1Returns(phi = PHI, annual_drift = 0.0, annual_volatility = 0.25)
            model, _, _ = return_fitted(process)
            @test residual_scale(model.regression) ≈ innovation_scale(process) rtol = 0.1
        end

        @testset "a stronger relationship is found more confidently" begin
            weak, _, _ = return_fitted(AR1Returns(phi = 0.05, annual_drift = 0.0))
            strong, _, _ = return_fitted(AR1Returns(phi = 0.45, annual_drift = 0.0))
            @test abs(coefficient_report(strong)[:log_return_1]["z"]) >
                abs(coefficient_report(weak)[:log_return_1]["z"])
        end
    end

    @testset "predictions" begin
        @testset "it returns a student-t predictive with provenance" begin
            model, examples, _ = return_fitted()
            result = predict(
                model, last(examples).features;
                symbol = "SYNTH", as_of = last(examples).features.as_of, horizon_bars = 1,
            )
            @test result.model.name === MOMENTUM
            @test result.model.params_hash !== nothing
            @test result.horizon_bars == 1
            @test result.n_observations == length(examples)
            @test isfinite(dof(result.distribution.ρ))
        end

        @testset "the predicted direction follows the signal" begin
            model, examples, _ = return_fitted()
            up = argmax(example -> require(example.features, :log_return_1), examples)
            down = argmin(example -> require(example.features, :log_return_1), examples)
            after_up = predict(
                model, up.features; symbol = "SYNTH", as_of = up.features.as_of,
                horizon_bars = 1,
            )
            after_down = predict(
                model, down.features; symbol = "SYNTH", as_of = down.features.as_of,
                horizon_bars = 1,
            )
            @test mean(after_up) > mean(after_down)
            @test probability_positive(after_up) > probability_positive(after_down)
        end

        @testset "predictions are modest relative to the noise" begin
            # An honest return model predicts far less than a day's typical move.
            model, examples, _ = return_fitted()
            for example in examples[1:200:end]
                result = predict(
                    model, example.features; symbol = "SYNTH",
                    as_of = example.features.as_of, horizon_bars = 1,
                )
                @test abs(mean(result)) < 3 * residual_scale(model.regression)
            end
        end

        @testset "the diagnostics carry the fitted scale" begin
            model, examples, _ = return_fitted()
            result = predict(
                model, last(examples).features; symbol = "SYNTH",
                as_of = last(examples).features.as_of, horizon_bars = 1,
            )
            @test result.diagnostics[:effective_sample_size] ≈ Float64(length(examples))
            @test result.diagnostics[:response_scale] == 1.0
            @test 0.0 <= epistemic_share(result) < 0.1
        end

        @testset "an unusual feature value widens the interval" begin
            model, examples, _ = return_fitted()
            ordinary = last(examples).features
            extreme = FeatureVector(
                symbol = ordinary.symbol, as_of = ordinary.as_of,
                data_as_of = ordinary.data_as_of,
                values = Dict(:log_return_1 => 5.0), n_bars = ordinary.n_bars,
            )
            narrow = predict(
                model, ordinary; symbol = "SYNTH", as_of = ordinary.as_of, horizon_bars = 1,
            )
            wide = predict(
                model, extreme; symbol = "SYNTH", as_of = ordinary.as_of, horizon_bars = 1,
            )
            @test width(credible_interval(wide)) > width(credible_interval(narrow))
            @test epistemic_share(wide) > epistemic_share(narrow)
        end
    end

    @testset "contract" begin
        @testset "predicting or updating before fitting is an error" begin
            model = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            examples, _ = return_examples(; n_bars = 200)
            @test !is_fitted(model)
            @test_throws NotFittedError predict(
                model, first(examples).features; symbol = "SYNTH",
                as_of = first(examples).features.as_of, horizon_bars = 1,
            )
            @test_throws NotFittedError update!(model, first(examples))
        end

        @testset "the scaler is frozen at fit time" begin
            # This is why a fit over all rows and a fit over the first hundred followed by
            # updates do not agree: they standardise against different windows, and only the
            # first hundred rows were knowable when the second model was fitted.
            examples, _ = return_examples(; n_bars = 1_000)
            model = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            fit!(model, examples[1:100])
            frozen = model.scaler
            for example in examples[101:end]
                update!(model, example)
            end
            @test model.scaler === frozen

            raw = reduce(
                vcat,
                [
                    transpose(design_row(example.features, RETURN_FEATURES))
                        for example in examples[1:100]
                ],
            )
            expected = fit_scaler(RETURN_FEATURES, raw)
            @test model.scaler.centres ≈ expected.centres
            @test model.scaler.scales ≈ expected.scales
        end

        @testset "sequential updates reach the same posterior as one batch" begin
            # Conjugacy end to end, holding the scaler fixed so only the update path varies.
            examples, _ = return_examples(; n_bars = 1_000)
            model = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            fit!(model, examples[1:100])
            for example in examples[101:end]
                update!(model, example)
            end

            scaler = model.scaler
            raw = reduce(
                vcat,
                [
                    transpose(design_row(example.features, RETURN_FEATURES))
                        for example in examples
                ],
            )
            design = hcat(ones(length(examples)), transform(scaler, raw))
            responses = Float64[example.label.forward_log_return for example in examples]

            reference = BayesianLinearModel(model.regression.prior)
            fit!(reference, design, responses)

            @test coefficients(model.regression) ≈ coefficients(reference)
            @test posterior_rate(model.regression) ≈ posterior_rate(reference)
            @test n_observations(model) == length(examples)
        end

        @testset "uncertainty falls as evidence accumulates" begin
            examples, _ = return_examples()
            small = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            fit!(small, examples[1:200])
            large = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            fit!(large, examples)
            @test uncertainty(large) < uncertainty(small)
            @test isinf(uncertainty(BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)))
        end

        @testset "reset returns the model to its prior" begin
            model, _, _ = return_fitted()
            reset!(model)
            @test !is_fitted(model)
            @test model.scaler === nothing
            @test params_hash(model) === nothing
        end

        @testset "the model version changes when the posterior does" begin
            model, examples, _ = return_fitted()
            before = params_hash(model)
            update!(model, first(examples))
            @test params_hash(model) != before
        end

        @testset "can_predict reports a warming-up vector" begin
            model, examples, engine = return_fitted()
            @test can_predict(model, last(examples).features)
            first_bar = first(load_range(engine.store, "SYNTH")).timestamp
            @test !can_predict(model, features_at(engine, "SYNTH", first_bar))
        end
    end

    @testset "horizon" begin
        @testset "asking about another horizon is refused" begin
            model, examples, _ = return_fitted()
            @test_throws HorizonMismatchError predict(
                model, last(examples).features; symbol = "SYNTH",
                as_of = last(examples).features.as_of, horizon_bars = 20,
            )
        end

        @testset "training on the wrong horizon is refused" begin
            examples, _ = return_examples(; n_bars = 1_000, horizon = 5)
            model = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            @test_throws HorizonMismatchError fit!(model, examples)
        end

        @testset "a longer horizon model predicts a wider distribution" begin
            short_rows, _ = return_examples(GaussianReturns(annual_drift = 0.0); horizon = 1)
            long_rows, _ = return_examples(GaussianReturns(annual_drift = 0.0); horizon = 20)

            short = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            fit!(short, short_rows)
            long = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 20)
            fit!(long, long_rows)

            @test residual_scale(long.regression) > 3 * residual_scale(short.regression)
        end
    end

    @testset "construction" begin
        @testset "degenerate specifications are rejected" begin
            @test_throws ArgumentError BayesianReturnModel(Symbol[]; horizon_bars = 1)
            @test_throws ArgumentError BayesianReturnModel([:a, :a]; horizon_bars = 1)
            @test_throws ArgumentError BayesianReturnModel([:a]; horizon_bars = 0)
        end

        @testset "a prior of the wrong width is rejected" begin
            @test_throws ArgumentError BayesianReturnModel(
                [:a, :b]; horizon_bars = 1,
                prior = weakly_informative_prior(2; residual_scale = 0.1),
            )
        end

        @testset "too few training rows are refused" begin
            examples, _ = return_examples(; n_bars = 200)
            model = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
            @test_throws ArgumentError fit!(model, examples[1:10])
        end

        @testset "the intercept leads the column order" begin
            @test design_columns(BayesianReturnModel([:a, :b]; horizon_bars = 1)) ==
                [:intercept, :a, :b]
        end
    end

    @testset "multiple features" begin
        multi() = return_examples(
            AR1Returns(phi = PHI, annual_drift = 0.0);
            seed = 23,
            features = FeatureSet(
                Feature[LogReturn(1), LogReturn(5), RelativeStrengthIndex(14)],
            ),
        )

        @testset "it puts its weight on the feature that carries the signal" begin
            examples, _ = multi()
            model = BayesianReturnModel(
                [:log_return_1, :log_return_5, :rsi_14]; horizon_bars = 1,
            )
            fit!(model, examples)
            report = coefficient_report(model)
            @test abs(report[:log_return_1]["z"]) > abs(report[:rsi_14]["z"])
        end

        @testset "features on wildly different scales stay comparable" begin
            # rsi_14 lives on 0-100 and log_return_1 on 0.01; standardising is what makes
            # their coefficients readable side by side.
            examples, _ = multi()
            model = BayesianReturnModel(
                [:log_return_1, :log_return_5, :rsi_14]; horizon_bars = 1,
            )
            fit!(model, examples)
            report = coefficient_report(model)
            magnitudes = Float64[
                abs(report[column]["standardised"]) for column in
                    [:log_return_1, :log_return_5, :rsi_14]
            ]
            @test maximum(magnitudes) < 100 * minimum(magnitudes)
        end

        @testset "the design column order is fixed by the model" begin
            examples, _ = multi()
            model = BayesianReturnModel(
                [:log_return_1, :log_return_5, :rsi_14]; horizon_bars = 1,
            )
            fit!(model, examples)
            @test Set(keys(coefficient_report(model))) == Set(design_columns(model))
            @test BayesTrade.parameters(model)["columns"] ==
                String[string(name) for name in design_columns(model)]
        end
    end

    @testset "forgetting tracks a market that changes" begin
        first_half = generate_series(
            AR1Returns(phi = 0.45, annual_drift = 0.0);
            symbol = "SYNTH", n_bars = 3_000, seed = 5, start = Date(2008, 1, 1),
        )
        second_half = generate_series(
            AR1Returns(phi = -0.45, annual_drift = 0.0);
            symbol = "SYNTH", n_bars = 3_000, seed = 6, start = Date(2020, 1, 1),
        )
        store = InMemoryBarStore(vcat(first_half.bars, second_half.bars))
        engine = FeatureEngine(store, FeatureSet(Feature[LogReturn(1)]))
        examples = build_training_set(engine, "SYNTH"; horizon_bars = 1)

        remembering = BayesianReturnModel(RETURN_FEATURES; horizon_bars = 1)
        fit!(remembering, examples)
        forgetting = BayesianReturnModel(
            RETURN_FEATURES; horizon_bars = 1, forgetting = 0.999,
        )
        fit!(forgetting, examples)

        @test coefficient_report(forgetting)[:log_return_1]["raw"] < -0.2
        @test abs(coefficient_report(remembering)[:log_return_1]["raw"]) < 0.15
    end
end
