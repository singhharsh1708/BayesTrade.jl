# Conjugate linear regression, checked against the closed form it claims to implement.
#
# Where an analytic answer exists it is computed independently in the test rather than taken
# from the implementation. Where the claim is statistical, it is checked on data generated
# from known coefficients.

const TRUE_COEFFICIENTS = [0.5, -0.3, 0.2]
const NOISE = 0.1

linprior(n = 3; residual_scale = 0.1, coefficient_scale = 10.0, shape = 2.0) =
    weakly_informative_prior(
    n; residual_scale = residual_scale,
    coefficient_scale = coefficient_scale, shape = shape,
)

lindesign(n; seed = 3, features = 3) = randn(Xoshiro(seed), n, features)

linresponses(X; coefficients = TRUE_COEFFICIENTS, noise = NOISE, seed = 4) =
    X * coefficients .+ noise .* randn(Xoshiro(seed), size(X, 1))

function linfitted(n = 2_000; kwargs...)
    model = BayesianLinearModel(linprior(; kwargs...))
    X = lindesign(n)
    fit!(model, X, linresponses(X))
    return model
end

@testset "conjugate linear regression" begin
    @testset "prior" begin
        @testset "a shape at or below one leaves the noise mean undefined" begin
            @test_throws ArgumentError linprior(3; shape = 1.0)
        end

        @testset "degenerate scales are rejected" begin
            @test_throws ArgumentError weakly_informative_prior(3; residual_scale = 0.0)
            @test_throws ArgumentError weakly_informative_prior(
                3; residual_scale = 0.1, coefficient_scale = 0.0,
            )
            @test_throws ArgumentError weakly_informative_prior(0; residual_scale = 0.1)
        end

        @testset "a malformed precision is rejected" begin
            @test_throws ArgumentError NormalInverseGammaPrior(
                zeros(2), [1.0 0.5; 0.0 1.0], 2.0, 1.0,
            )
            @test_throws ArgumentError NormalInverseGammaPrior(
                zeros(2), -Matrix{Float64}(I, 2, 2), 2.0, 1.0,
            )
            @test_throws ArgumentError NormalInverseGammaPrior(
                zeros(3), Matrix{Float64}(I, 2, 2), 2.0, 1.0,
            )
            @test_throws ArgumentError NormalInverseGammaPrior(
                zeros(2), Matrix{Float64}(I, 2, 2), 2.0, 0.0,
            )
        end

        @testset "the weakly informative prior expresses the scale it was given" begin
            prior = linprior(3; residual_scale = 0.05)
            @test expected_noise_variance(prior) ≈ 0.05^2
            @test prior.mean == zeros(3)
        end

        @testset "a wider coefficient scale is a weaker prior" begin
            @test linprior(3; coefficient_scale = 0.1).precision[1, 1] >
                linprior(3; coefficient_scale = 10.0).precision[1, 1]
        end
    end

    @testset "recovery" begin
        @testset "the posterior mean recovers the true coefficients" begin
            @test coefficients(linfitted()) ≈ TRUE_COEFFICIENTS atol = 0.01
        end

        @testset "the posterior recovers the noise scale" begin
            @test residual_scale(linfitted()) ≈ NOISE rtol = 0.05
        end

        @testset "each credible interval covers its true coefficient" begin
            model = linfitted()
            spreads = coefficient_std(model)
            for (index, truth) in enumerate(TRUE_COEFFICIENTS)
                @test abs(coefficients(model)[index] - truth) < 3 * spreads[index]
            end
        end

        @testset "more data shrinks the posterior" begin
            @test coefficient_std(linfitted(200))[1] > coefficient_std(linfitted(5_000))[1]
        end

        @testset "an unfitted model sits on its prior" begin
            model = BayesianLinearModel(linprior())
            @test coefficients(model) ≈ zeros(3)
            @test noise_variance(model) ≈ expected_noise_variance(model.prior)
            @test effective_sample_size(model) == 0.0
            @test n_absorbed(model) == 0
        end

        @testset "a pure noise response leaves the coefficients near zero" begin
            # The null case a return model must get right, or it is fitting noise.
            model = BayesianLinearModel(linprior())
            X = lindesign(4_000)
            fit!(model, X, NOISE .* randn(Xoshiro(9), 4_000))
            spreads = coefficient_std(model)
            for index in 1:3
                @test abs(coefficients(model)[index]) < 3 * spreads[index]
            end
        end
    end

    @testset "closed form" begin
        @testset "the posterior matches the normal-inverse-gamma update" begin
            # Every posterior quantity, recomputed independently from the definitions.
            prior = linprior()
            model = BayesianLinearModel(prior)
            X = lindesign(200)
            y = linresponses(X)
            fit!(model, X, y)

            precision = prior.precision + transpose(X) * X
            mean = precision \ (prior.precision * prior.mean + transpose(X) * y)
            shape = prior.shape + 200 / 2
            rate = prior.rate + (
                dot(prior.mean, prior.precision * prior.mean) + dot(y, y) -
                    dot(mean, precision * mean)
            ) / 2

            @test posterior_precision(model) ≈ precision
            @test coefficients(model) ≈ mean
            @test posterior_shape(model) ≈ shape
            @test posterior_rate(model) ≈ rate
        end

        @testset "the predictive matches its definition" begin
            model = linfitted(500)
            x = [0.4, -0.2, 1.1]
            leverage = dot(x, posterior_precision(model) \ x)
            distribution, _ = predict(model, x)

            @test dof(distribution.ρ) ≈ 2 * posterior_shape(model)
            @test mean(distribution) ≈ dot(x, coefficients(model))
            @test distribution.σ ≈
                sqrt(posterior_rate(model) / posterior_shape(model) * (1 + leverage))
        end

        @testset "a one dimensional posterior matches a hand computation" begin
            # Small enough to check by hand: two observations, unit prior precision.
            prior = NormalInverseGammaPrior([0.0], reshape([1.0], 1, 1), 2.0, 1.0)
            model = BayesianLinearModel(prior)
            fit!(model, reshape([1.0, 2.0], 2, 1), [1.0, 2.0])

            @test posterior_precision(model)[1, 1] ≈ 6.0
            @test coefficients(model)[1] ≈ 5 / 6
            @test posterior_shape(model) ≈ 3.0
            @test posterior_rate(model) ≈ 1.0 + (5.0 - 25 / 6) / 2
        end
    end

    @testset "predictive" begin
        @testset "degrees of freedom grow with the data" begin
            x = [1.0, 0.0, 0.0]
            @test dof(first(predict(linfitted(100), x)).ρ) <
                dof(first(predict(linfitted(5_000), x)).ρ)
        end

        @testset "heavier tails than a variance matched normal" begin
            # An unknown noise variance puts materially more mass on a large adverse move,
            # and understating that is the one failure this system cannot afford.
            distribution, _ = predict(linfitted(60), [1.0, 0.0, 0.0])
            matched = Normal(mean(distribution), std(distribution))
            threshold = mean(distribution) + 5 * std(distribution)
            @test probability_above(distribution, threshold) >
                probability_above(matched, threshold)
        end

        @testset "extrapolation is less certain than interpolation" begin
            model = linfitted(500)
            near, near_epistemic = predict(model, [0.1, 0.0, 0.0])
            far, far_epistemic = predict(model, [20.0, 0.0, 0.0])
            @test std(far) > std(near)
            @test far_epistemic > near_epistemic
        end

        @testset "the epistemic share falls as evidence accumulates" begin
            x = [1.0, 0.5, -0.5]
            small, small_epistemic = predict(linfitted(50), x)
            large, large_epistemic = predict(linfitted(5_000), x)
            @test small_epistemic / var(small) > large_epistemic / var(large)
        end

        @testset "with no data the predictive is the prior" begin
            model = BayesianLinearModel(linprior(3; residual_scale = 0.2))
            distribution, epistemic = predict(model, zeros(3))
            @test mean(distribution) ≈ 0.0
            @test epistemic ≈ 0.0 atol = 1.0e-12
        end

        @testset "predicting many rows agrees with predicting one" begin
            model = linfitted(200)
            X = lindesign(10; seed = 99)
            @test predict_mean(model, X) ≈
                [mean(first(predict(model, X[index, :]))) for index in 1:10]
        end
    end

    @testset "conjugacy" begin
        @testset "a batch fit equals a sequence of updates" begin
            # The property the hot path depends on: online costs nothing in accuracy.
            X = lindesign(300)
            y = linresponses(X)

            batch = BayesianLinearModel(linprior())
            fit!(batch, X, y)

            incremental = BayesianLinearModel(linprior())
            for index in 1:300
                update!(incremental, X[index, :], y[index])
            end

            @test coefficients(incremental) ≈ coefficients(batch)
            @test posterior_rate(incremental) ≈ posterior_rate(batch)
            @test posterior_shape(incremental) ≈ posterior_shape(batch)
            @test coefficient_std(incremental) ≈ coefficient_std(batch)
        end

        @testset "order does not matter without forgetting" begin
            X = lindesign(300)
            y = linresponses(X)
            order = randperm(Xoshiro(1), 300)

            forward = BayesianLinearModel(linprior())
            fit!(forward, X, y)
            shuffled = BayesianLinearModel(linprior())
            fit!(shuffled, X[order, :], y[order])

            @test coefficients(shuffled) ≈ coefficients(forward) atol = 1.0e-9
        end

        @testset "refitting discards the previous batch" begin
            model = BayesianLinearModel(linprior())
            X = lindesign(500)
            y = linresponses(X)
            fit!(model, X, y)
            fit!(model, X[1:10, :], y[1:10])
            @test n_absorbed(model) == 10
            @test effective_sample_size(model) ≈ 10.0
        end

        @testset "reset returns to the prior" begin
            model = linfitted(500)
            reset!(model)
            @test coefficients(model) ≈ zeros(3)
            @test n_absorbed(model) == 0
        end
    end

    @testset "forgetting" begin
        @testset "the effective sample size converges rather than growing" begin
            model = BayesianLinearModel(linprior(); forgetting = 0.99)
            X = lindesign(5_000)
            fit!(model, X, linresponses(X))
            @test n_absorbed(model) == 5_000
            @test effective_sample_size(model) ≈ 100.0 rtol = 0.01
        end

        @testset "it tracks a coefficient that moves" begin
            # A regime change: the true coefficient flips halfway through the sample.
            rng = Xoshiro(12)
            X = randn(rng, 4_000, 1)
            y = vcat(
                X[1:2_000, 1] .+ NOISE .* randn(rng, 2_000),
                -X[2_001:end, 1] .+ NOISE .* randn(rng, 2_000),
            )

            remembering = BayesianLinearModel(linprior(1))
            fit!(remembering, X, y)
            forgetting = BayesianLinearModel(linprior(1); forgetting = 0.99)
            fit!(forgetting, X, y)

            @test coefficients(forgetting)[1] ≈ -1.0 atol = 0.1
            @test coefficients(remembering)[1] ≈ 0.0 atol = 0.1
        end

        @testset "a batch fit still equals a sequence of updates" begin
            X = lindesign(300)
            y = linresponses(X)

            batch = BayesianLinearModel(linprior(); forgetting = 0.98)
            fit!(batch, X, y)
            incremental = BayesianLinearModel(linprior(); forgetting = 0.98)
            for index in 1:300
                update!(incremental, X[index, :], y[index])
            end
            @test coefficients(incremental) ≈ coefficients(batch)
            @test posterior_rate(incremental) ≈ posterior_rate(batch)
        end

        @testset "forgetting leaves wider posteriors than remembering" begin
            X = lindesign(4_000)
            y = linresponses(X)
            remembering = BayesianLinearModel(linprior())
            fit!(remembering, X, y)
            forgetting = BayesianLinearModel(linprior(); forgetting = 0.98)
            fit!(forgetting, X, y)
            @test coefficient_std(forgetting)[1] > coefficient_std(remembering)[1]
        end

        @testset "an out of range forgetting factor is rejected" begin
            @test_throws ArgumentError BayesianLinearModel(linprior(); forgetting = 1.5)
            @test_throws ArgumentError BayesianLinearModel(linprior(); forgetting = 0.0)
        end
    end

    @testset "numerics" begin
        @testset "perfectly collinear features are regularised, not fatal" begin
            rng = Xoshiro(6)
            first_column = randn(rng, 500)
            X = hcat(first_column, first_column, randn(rng, 500))
            model = BayesianLinearModel(linprior(3; coefficient_scale = 1.0))
            fit!(model, X, linresponses(X))
            @test all(isfinite, coefficients(model))
            @test all(isfinite, coefficient_std(model))
        end

        @testset "a perfect fit does not produce a negative rate" begin
            X = lindesign(200)
            model = BayesianLinearModel(linprior())
            fit!(model, X, X * TRUE_COEFFICIENTS)
            @test posterior_rate(model) > 0
            @test residual_scale(model) > 0
        end

        @testset "a constant feature column is handled" begin
            X = hcat(ones(300), lindesign(300; features = 2))
            model = BayesianLinearModel(linprior())
            fit!(model, X, fill(0.05, 300))
            @test coefficients(model)[1] ≈ 0.05 atol = 0.01
        end
    end

    @testset "validation" begin
        @testset "a row of the wrong width is rejected" begin
            @test_throws ArgumentError update!(linfitted(50), [1.0, 2.0], 0.1)
            @test_throws ArgumentError predict(linfitted(50), zeros(5))
        end

        @testset "non-finite inputs are rejected" begin
            @test_throws ArgumentError update!(linfitted(50), zeros(3), NaN)
            @test_throws ArgumentError update!(linfitted(50), [1.0, Inf, 0.0], 0.1)
        end

        @testset "a mismatched batch is rejected" begin
            @test_throws ArgumentError fit!(
                BayesianLinearModel(linprior()), lindesign(10), zeros(9),
            )
            @test_throws ArgumentError fit!(
                BayesianLinearModel(linprior()), lindesign(10; features = 2), zeros(10),
            )
        end
    end

    @testset "provenance and state" begin
        @testset "parameters describe the fitted state" begin
            parameters = BayesTrade.parameters(linfitted(500))
            @test length(parameters["coefficients"]) == 3
            @test parameters["effective_sample_size"] ≈ 500.0
            @test parameters["forgetting"] == 1.0
        end

        @testset "the same data produces the same parameters" begin
            @test stable_hash(BayesTrade.parameters(linfitted(500))) ==
                stable_hash(BayesTrade.parameters(linfitted(500)))
        end

        @testset "state round trips" begin
            model = linfitted(500)
            restored = BayesianLinearModel(linprior())
            load_state!(restored, state(model))
            @test coefficients(restored) ≈ coefficients(model)
            @test posterior_rate(restored) ≈ posterior_rate(model)
            @test n_absorbed(restored) == n_absorbed(model)
        end

        @testset "a restored model keeps updating correctly" begin
            model = linfitted(500)
            restored = BayesianLinearModel(linprior())
            load_state!(restored, state(model))
            row = [0.3, -0.1, 0.2]
            update!(model, row, 0.05)
            update!(restored, row, 0.05)
            @test coefficients(restored) ≈ coefficients(model)
        end

        @testset "a mismatched state is refused" begin
            model = BayesianLinearModel(linprior())
            @test_throws ArgumentError load_state!(
                model,
                Dict(
                    "xx" => [[1.0, 0.0], [0.0, 1.0]], "xy" => [0.0, 0.0],
                    "yy" => 0.0, "weight" => 0.0, "n_seen" => 0,
                ),
            )
        end

        @testset "show reports the fitted scale" begin
            @test occursin("residual_scale=", sprint(show, linfitted(100)))
        end
    end
end
