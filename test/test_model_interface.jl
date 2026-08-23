# The contract is exercised through a conjugate normal-mean model with known observation
# variance, chosen because its posterior has a closed form. Every assertion below compares
# against that closed form rather than against whatever the implementation produced.

mutable struct ConjugateMeanModel <: ProbabilisticModel
    prior_mean::Float64
    prior_variance::Float64
    noise_variance::Float64
    mean::Float64
    variance::Float64
    state::FitState
end

ConjugateMeanModel(; prior_mean = 0.0, prior_variance = 1.0, noise_variance = 4.0) =
    ConjugateMeanModel(
    prior_mean, prior_variance, noise_variance, prior_mean, prior_variance, FitState(),
)

BayesTrade.fit_state(model::ConjugateMeanModel) = model.state
BayesTrade.model_name(::ConjugateMeanModel) = MOMENTUM
BayesTrade.model_semver(::ConjugateMeanModel) = v"0.1.0"
BayesTrade.parameters(model::ConjugateMeanModel) =
    Dict("mean" => model.mean, "variance" => model.variance)
BayesTrade.uncertainty(model::ConjugateMeanModel) = sqrt(model.variance)

function absorb!(model::ConjugateMeanModel, observation::Float64)
    precision = 1 / model.variance + 1 / model.noise_variance
    model.mean =
        (model.mean / model.variance + observation / model.noise_variance) / precision
    model.variance = 1 / precision
    return model
end

function BayesTrade.fit!(model::ConjugateMeanModel, observations)
    model.mean = model.prior_mean
    model.variance = model.prior_variance
    for observation in observations
        absorb!(model, Float64(observation))
    end
    return mark_fitted!(
        model; n_observations = length(observations), fitted_at = DateTime(2026, 1, 2),
    )
end

function BayesTrade.update!(model::ConjugateMeanModel, observation)
    require_fitted(model)
    absorb!(model, Float64(observation))
    model.state.n_observations += 1
    return model
end

function BayesTrade.predict(model::ConjugateMeanModel, features; symbol, as_of, horizon_bars = 1)
    require_fitted(model)
    return ProbabilisticResult(
        model = model_version(model), symbol = symbol, as_of = as_of,
        horizon_bars = horizon_bars,
        distribution = Normal(model.mean + features, sqrt(model.variance + model.noise_variance)),
        n_observations = n_observations(model),
        epistemic_variance = model.variance,
    )
end

function BayesTrade.reset!(model::ConjugateMeanModel)
    model.mean = model.prior_mean
    model.variance = model.prior_variance
    return invoke(reset!, Tuple{ProbabilisticModel}, model)
end

const OBSERVATIONS = [1.0, 3.0, 2.0, 4.0]

@testset "model interface" begin
    @testset "predicting before fitting is an error, not a prior guess" begin
        model = ConjugateMeanModel()
        @test !is_fitted(model)
        @test_throws NotFittedError predict(
            model, 0.0; symbol = "RELIANCE", as_of = DateTime(2026, 1, 2),
        )
        @test_throws NotFittedError update!(model, 1.0)
        @test occursin("has not been fitted", sprint(showerror, NotFittedError("M")))
    end

    @testset "the posterior matches the closed form" begin
        model = ConjugateMeanModel()
        fit!(model, OBSERVATIONS)
        n, sigma2, tau2 = length(OBSERVATIONS), 4.0, 1.0
        precision = 1 / tau2 + n / sigma2
        @test model.variance ≈ 1 / precision
        @test model.mean ≈ (0 / tau2 + sum(OBSERVATIONS) / sigma2) / precision
    end

    @testset "sequential updates agree with a batch fit" begin
        batch = ConjugateMeanModel()
        fit!(batch, OBSERVATIONS)

        incremental = ConjugateMeanModel()
        fit!(incremental, OBSERVATIONS[1:1])
        for observation in OBSERVATIONS[2:end]
            update!(incremental, observation)
        end

        @test uncertainty(incremental) ≈ uncertainty(batch)
        @test n_observations(incremental) == n_observations(batch)
        @test params_hash(incremental) == params_hash(batch)
    end

    @testset "uncertainty falls as evidence accumulates" begin
        model = ConjugateMeanModel()
        fit!(model, [1.0])
        before = uncertainty(model)
        for _ in 1:10
            update!(model, 1.0)
        end
        @test uncertainty(model) < before
    end

    @testset "reset returns the model to its unfitted state" begin
        model = ConjugateMeanModel()
        fit!(model, OBSERVATIONS)
        reset!(model)
        @test !is_fitted(model)
        @test n_observations(model) == 0
        @test params_hash(model) === nothing
        @test_throws NotFittedError predict(
            model, 0.0; symbol = "R", as_of = DateTime(2026, 1, 2),
        )
    end

    @testset "versioning" begin
        model = ConjugateMeanModel()
        @test params_hash(model) === nothing

        fit!(model, OBSERVATIONS)
        before = params_hash(model)
        update!(model, 5.0)
        @test params_hash(model) != before

        first, second = ConjugateMeanModel(), ConjugateMeanModel()
        fit!(first, OBSERVATIONS)
        fit!(second, OBSERVATIONS)
        @test model_version(first) == model_version(second)
    end

    @testset "every prediction carries its producing version" begin
        model = ConjugateMeanModel()
        fit!(model, OBSERVATIONS)
        result = predict(model, 0.0; symbol = "RELIANCE", as_of = DateTime(2026, 1, 2))
        @test result.model.name === MOMENTUM
        @test startswith(identifier(result.model), "momentum@0.1.0+")
        @test result.model.fitted_at == DateTime(2026, 1, 2)
        @test epistemic_share(result) > 0
    end

    @testset "an incomplete model says what it is missing" begin
        struct Incomplete <: ProbabilisticModel end
        @test_throws ArgumentError fit_state(Incomplete())
        @test_throws ArgumentError model_name(Incomplete())
        @test_throws ArgumentError parameters(Incomplete())
        @test_throws ArgumentError uncertainty(Incomplete())
        @test_throws ArgumentError fit!(Incomplete(), [1.0])
    end

    @testset "show reports the fitted state" begin
        model = ConjugateMeanModel()
        @test occursin("unfitted", sprint(show, model))
        fit!(model, OBSERVATIONS)
        @test occursin("fitted on 4 observations", sprint(show, model))
    end
end

@testset "stable hashing" begin
    @testset "key order does not change the digest" begin
        @test stable_hash(Dict("a" => 1.0, "b" => 2.0)) ==
            stable_hash(Dict("b" => 2.0, "a" => 1.0))
    end

    @testset "different values produce different digests" begin
        @test stable_hash(Dict("a" => 1.0)) != stable_hash(Dict("a" => 1.0000001))
    end

    @testset "floating point noise below the rounding point is absorbed" begin
        @test stable_hash(Dict("a" => 0.1 + 0.2)) == stable_hash(Dict("a" => 0.3))
        @test stable_hash(Dict("a" => -0.0)) == stable_hash(Dict("a" => 0.0))
    end

    @testset "non-finite values are representable and distinct" begin
        digests = Set(
            [
                stable_hash(Dict("a" => NaN)),
                stable_hash(Dict("a" => Inf)),
                stable_hash(Dict("a" => -Inf)),
            ]
        )
        @test length(digests) == 3
    end

    @testset "nested structures hash stably and order matters in sequences" begin
        left = Dict("coefficients" => [1.0, 2.0], "prior" => Dict("mean" => 0.0))
        right = Dict("prior" => Dict("mean" => 0.0), "coefficients" => [1.0, 2.0])
        @test stable_hash(left) == stable_hash(right)
        @test stable_hash(Dict("a" => [1.0, 2.0])) != stable_hash(Dict("a" => [2.0, 1.0]))
    end

    @testset "the digest matches the model version pattern" begin
        digest = stable_hash(Dict("a" => 1.0))
        @test length(digest) == 32
        @test all(c -> c in "0123456789abcdef", digest)
    end
end
