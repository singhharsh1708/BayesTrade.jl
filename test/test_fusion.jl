# The fusion layer: pooling, reliability, and the arithmetic that has to be exact.

fu_t(location, scale, df = 8.0) = student_t(location, scale, df)

function fu_models(; n_bars = 2_500, seed = 11, horizon = 1)
    series = generate_series(
        RegimeSwitchingReturns(); symbol = "RELIANCE", n_bars = n_bars, seed = seed,
        start = Date(2005, 1, 1),
    )
    engine = FeatureEngine(
        InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, "RELIANCE"; horizon_bars = horizon)
    momentum = BayesianReturnModel([:log_return_1]; horizon_bars = horizon)
    volatility = BayesianVolatilityModel(; horizon_bars = horizon)
    regime = MarketRegimeModel(; horizon_bars = horizon)
    for model in (momentum, volatility, regime)
        fit!(model, examples)
    end
    return (momentum, volatility, regime), examples
end

function fu_results(models, example; symbol = "RELIANCE")
    return map(
        model -> predict(
            model, example.features; symbol = symbol,
            as_of = example.features.as_of, horizon_bars = model.horizon_bars,
        ),
        models,
    )
end

@testset "opinion pool" begin
    @testset "it is a distribution, and a correct one" begin
        pool = OpinionPool((fu_t(0.001, 0.02), fu_t(-0.002, 0.03, 5.0)), [0.6, 0.4])
        @test pool isa UnivariateDistribution
        @test n_components(pool) == 2

        # The moments follow the law of total variance rather than an average of variances.
        centre = 0.6 * 0.001 + 0.4 * -0.002
        @test mean(pool) ≈ centre
        within = 0.6 * var(fu_t(0.001, 0.02)) + 0.4 * var(fu_t(-0.002, 0.03, 5.0))
        between = 0.6 * (0.001 - centre)^2 + 0.4 * (-0.002 - centre)^2
        @test var(pool) ≈ within + between
        @test disagreement(pool) ≈ between
        @test var(pool) > within

        # Density and mass agree with the components they came from.
        for x in (-0.05, 0.0, 0.013)
            @test pdf(pool, x) ≈ 0.6 * pdf(fu_t(0.001, 0.02), x) +
                0.4 * pdf(fu_t(-0.002, 0.03, 5.0), x)
            @test logpdf(pool, x) ≈ log(pdf(pool, x))
            @test cdf(pool, x) ≈ 0.6 * cdf(fu_t(0.001, 0.02), x) +
                0.4 * cdf(fu_t(-0.002, 0.03, 5.0), x)
            @test ccdf(pool, x) ≈ 1 - cdf(pool, x)
        end
        @test cdf(pool, 100.0) ≈ 1.0
        @test cdf(pool, -100.0) ≈ 0.0 atol = 1.0e-12
    end

    @testset "the quantile inverts the distribution function" begin
        pool = OpinionPool(
            (fu_t(0.01, 0.01), fu_t(-0.02, 0.05, 4.0), fu_t(0.0, 0.02, 30.0)),
            [0.2, 0.3, 0.5],
        )
        for level in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
            @test cdf(pool, quantile(pool, level)) ≈ level atol = 1.0e-9
        end
        @test quantile(pool, 0.5) > quantile(pool, 0.4)
        @test_throws ArgumentError quantile(pool, 0.0)
        @test_throws ArgumentError quantile(pool, 1.0)

        # A pool of identical components is that component.
        same = OpinionPool((fu_t(0.0, 0.02), fu_t(0.0, 0.02)), [0.5, 0.5])
        @test quantile(same, 0.9) ≈ quantile(fu_t(0.0, 0.02), 0.9)
        @test var(same) ≈ var(fu_t(0.0, 0.02))
        @test disagreement(same) ≈ 0.0
    end

    @testset "one component is the identity" begin
        component = fu_t(0.003, 0.017, 6.0)
        pool = OpinionPool((component,), [1.0])
        @test mean(pool) ≈ mean(component)
        @test var(pool) ≈ var(component)
        @test cdf(pool, 0.01) ≈ cdf(component, 0.01)
        @test logpdf(pool, 0.01) ≈ logpdf(component, 0.01)
    end

    @testset "it samples in proportion to its weights" begin
        pool = OpinionPool((fu_t(-1.0, 0.01), fu_t(1.0, 0.01)), [0.25, 0.75])
        rng = Xoshiro(5)
        draws = [rand(rng, pool) for _ in 1:20_000]
        @test mean(draws .> 0) ≈ 0.75 atol = 0.02
        @test mean(draws) ≈ mean(pool) atol = 0.03
    end

    @testset "a pool that is not a pool is refused" begin
        component = fu_t(0.0, 0.02)
        @test_throws ArgumentError OpinionPool((component, component), [0.5])
        @test_throws ArgumentError OpinionPool((component, component), [0.5, 0.4])
        @test_throws ArgumentError OpinionPool((component, component), [1.5, -0.5])
        @test_throws ArgumentError OpinionPool((component, component), [0.0, 0.0])
        @test_throws ArgumentError OpinionPool((component,), [NaN])
    end
end

@testset "model reliability" begin
    @testset "it starts even and finds the better model" begin
        reliability = ModelReliability([MOMENTUM, VOLATILITY, REGIME])
        @test n_models(reliability) == 3
        @test reliabilities(reliability) ≈ fill(1 / 3, 3)
        @test reliability_belief(reliability).labels == [MOMENTUM, VOLATILITY, REGIME]
        @test mean_log_scores(reliability) == zeros(3)

        rng = Xoshiro(1)
        truth = Normal(0.0, 0.02)
        for _ in 1:500
            outcome = rand(rng, truth)
            score!(
                reliability,
                [
                    logpdf(Normal(0.01, 0.05), outcome),
                    logpdf(truth, outcome),
                    logpdf(Normal(0.0, 0.04), outcome),
                ],
            )
        end
        weights = reliabilities(reliability)
        @test argmax(weights) == 2
        @test weights[2] > 0.9
        @test sum(weights) ≈ 1.0
        @test argmax(mean_log_scores(reliability)) == 2
        @test reliability.n_scored == 500
    end

    @testset "a beaten model can come back" begin
        # Weights live in log space with a floor precisely so this is possible.
        reliability = ModelReliability([MOMENTUM, VOLATILITY]; forgetting = 0.95)
        rng = Xoshiro(3)
        for _ in 1:400
            outcome = 0.05 * randn(rng)
            score!(reliability, [logpdf(Normal(0.0, 0.05), outcome), logpdf(Normal(0.0, 0.005), outcome)])
        end
        beaten = reliabilities(reliability)[2]
        @test beaten < 0.01
        @test all(isfinite, reliability.log_weights)

        for _ in 1:400
            outcome = 0.005 * randn(rng)
            score!(reliability, [logpdf(Normal(0.0, 0.05), outcome), logpdf(Normal(0.0, 0.005), outcome)])
        end
        @test reliabilities(reliability)[2] > 0.9
    end

    @testset "forgetting is what lets the answer move" begin
        # Both settings get there in the end, so the claim is about speed, not destination.
        # After six hundred bars of being wrong the second model sits on the floor under
        # either setting; what differs is how long it takes to climb back.
        rng = Xoshiro(7)
        before = 0.05 .* randn(rng, 600)
        after = 0.005 .* randn(rng, 60)

        function recovery(forgetting)
            reliability = ModelReliability([MOMENTUM, VOLATILITY]; forgetting = forgetting)
            for outcome in before
                score!(
                    reliability,
                    [logpdf(Normal(0.0, 0.05), outcome), logpdf(Normal(0.0, 0.005), outcome)],
                )
            end
            beaten = reliabilities(reliability)[2]
            trail = Float64[]
            for outcome in after
                score!(
                    reliability,
                    [logpdf(Normal(0.0, 0.05), outcome), logpdf(Normal(0.0, 0.005), outcome)],
                )
                push!(trail, reliabilities(reliability)[2])
            end
            return beaten, findfirst(>(0.9), trail), trail
        end

        moving_beaten, moving_bars, moving_trail = recovery(0.95)
        static_beaten, static_bars, static_trail = recovery(1.0)

        @test moving_beaten < 1.0e-15
        @test static_beaten < 1.0e-15
        @test moving_bars !== nothing
        @test static_bars !== nothing
        @test moving_bars < static_bars
        @test moving_trail[10] > 1_000 * static_trail[10]
    end

    @testset "a model that could not answer is not scored" begin
        reliability = ModelReliability([MOMENTUM, VOLATILITY])
        score!(reliability, [logpdf(Normal(0.0, 0.02), 0.001), -Inf])
        @test reliabilities(reliability)[1] > reliabilities(reliability)[2]
        @test all(isfinite, reliability.log_weights)
        @test mean_log_scores(reliability)[2] == 0.0

        @test_throws ArgumentError score!(reliability, [0.0])
        @test_throws ArgumentError score!(reliability, [NaN, 0.0])
        @test_throws ArgumentError score!(reliability, [-Inf, -Inf])
        @test_throws ArgumentError ModelReliability(ModelName[])
        @test_throws ArgumentError ModelReliability([MOMENTUM, MOMENTUM])
        @test_throws ArgumentError ModelReliability([MOMENTUM]; forgetting = 0.0)

        reset!(reliability)
        @test reliabilities(reliability) ≈ [0.5, 0.5]
        @test reliability.n_scored == 0
    end
end

@testset "fusion" begin
    models, examples = fu_models()
    example = last(examples)
    results = fu_results(models, example)

    @testset "it pools what the models said" begin
        reliability = ModelReliability([MOMENTUM, VOLATILITY, REGIME])
        prediction = fuse(reliability, results)

        @test prediction isa FusedPrediction
        @test n_models(prediction) == 3
        @test prediction.symbol == "RELIANCE"
        @test prediction.as_of == example.features.as_of
        @test prediction.horizon_bars == 1
        @test prediction.weights.labels == [MOMENTUM, VOLATILITY, REGIME]
        @test sum(prediction.weights.probabilities) ≈ 1.0
        @test length(prediction.sources) == 3
        @test occursin("FusedPrediction", sprint(show, prediction))

        # Between the components it came from, never outside them.
        centres = [mean(result.distribution) for result in results]
        @test minimum(centres) <= mean(prediction) <= maximum(centres)
        spreads = [std(result.distribution) for result in results]
        @test minimum(spreads) <= std(prediction) <= maximum(spreads) * 1.5

        @test 0 < epistemic_share(prediction) < 1
        @test prediction.diagnostics[:n_models] == 3.0
        @test prediction.diagnostics[:disagreement] ≈ disagreement(prediction.distribution)
        @test prediction.epistemic_variance ≈
            prediction.diagnostics[:within_model_epistemic] +
            prediction.diagnostics[:disagreement]
        @test 0 <= prediction.diagnostics[:weight_entropy] <= 1
        @test isfinite(probability_positive(prediction))
        @test credible_interval(prediction).lower < mean(prediction)
    end

    @testset "an absent model costs the others nothing" begin
        reliability = ModelReliability([MOMENTUM, VOLATILITY, REGIME])
        partial = fuse(reliability, (results[1], results[3]))
        @test n_models(partial) == 2
        @test sum(partial.weights.probabilities) ≈ 1.0
        @test partial.weights.probabilities ≈ [0.5, 0.5]
        @test partial.weights.labels == [MOMENTUM, REGIME]

        single = fuse(reliability, (results[2],))
        @test n_models(single) == 1
        @test mean(single) ≈ mean(results[2].distribution)
        @test var(single) ≈ var(results[2].distribution)
        @test single.diagnostics[:disagreement] ≈ 0.0
    end

    @testset "the weights follow the reliabilities" begin
        reliability = ModelReliability([MOMENTUM, VOLATILITY, REGIME])
        for _ in 1:200
            score!(reliability, [-2.0, 5.0, -2.0])
        end
        prediction = fuse(reliability, results)
        @test argmax(prediction.weights.probabilities) == 2
        @test prediction.weights.probabilities[2] > 0.9
        # The pool then sits essentially on the model that earned it. Compared absolutely:
        # the volatility model's predictive is centred, so its mean is exactly zero and a
        # relative tolerance against it could never pass.
        @test mean(prediction) ≈ mean(results[2].distribution) atol = 1.0e-6
        @test std(prediction) ≈ std(results[2].distribution) rtol = 0.05
    end

    @testset "fusing answers to different questions is refused" begin
        reliability = ModelReliability([MOMENTUM, VOLATILITY, REGIME])
        other_symbol = predict(
            models[1], example.features; symbol = "INFY",
            as_of = example.features.as_of, horizon_bars = 1,
        )
        other_time = predict(
            models[1], example.features; symbol = "RELIANCE",
            as_of = example.features.as_of + Day(1), horizon_bars = 1,
        )
        @test_throws ArgumentError fuse(reliability, (results[2], other_symbol))
        @test_throws ArgumentError fuse(reliability, (results[2], other_time))
        @test_throws ArgumentError fuse(reliability, (results[1], results[1]))
        @test_throws ArgumentError fuse(reliability, ())
        @test_throws ArgumentError fuse(ModelReliability([VOLATILITY]), (results[1],))
    end

    @testset "scoring moves the weights toward whoever was right" begin
        reliability = ModelReliability([MOMENTUM, VOLATILITY, REGIME])
        before = reliabilities(reliability)
        score_fusion!(reliability, results, example.label.forward_log_return)
        after = reliabilities(reliability)
        @test sum(after) ≈ 1.0
        @test after != before
        @test reliability.n_scored == 1

        best = argmax(
            [
                logpdf(result.distribution, example.label.forward_log_return) for
                    result in results
            ]
        )
        @test argmax(after) == best
    end

    @testset "pooling beats the models it pools, out of sample" begin
        # The claim the layer exists to make. Scored prequentially: every bar is forecast
        # before it is used to move the weights.
        models, examples = fu_models(; n_bars = 3_000, seed = 5)
        reliability = ModelReliability([MOMENTUM, VOLATILITY, REGIME])
        pooled = Float64[]
        individual = [Float64[], Float64[], Float64[]]

        for example in examples[2_000:end]
            results = fu_results(models, example)
            outcome = example.label.forward_log_return
            push!(pooled, logpdf(fuse(reliability, results).distribution, outcome))
            for (index, result) in enumerate(results)
                push!(individual[index], logpdf(result.distribution, outcome))
            end
            score_fusion!(reliability, results, outcome)
        end

        @test all(isfinite, pooled)
        # At least as good as the average model, and close to the best one without having
        # been told in advance which that was.
        @test mean(pooled) > mean(mean.(individual))
        @test mean(pooled) > maximum(mean.(individual)) - 0.05
    end
end
