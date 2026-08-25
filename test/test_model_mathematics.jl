# Section 4 of the validation brief: check that the mathematics behaves, not that the code runs.
#
# The conjugate normal-inverse-gamma regression has closed forms for everything, so nothing here
# has to be approximated or eyeballed. Where a quantity has an analytic value, the test computes
# it independently from the textbook expression and compares.

using LinearAlgebra: I, dot

"""
    analytic_posterior(prior, X, y)

The posterior a normal-inverse-gamma regression must reach, written out from the definition
rather than taken from the implementation being tested.

    Lambda_n = Lambda_0 + X'X
    m_n      = Lambda_n^-1 (Lambda_0 m_0 + X'y)
    a_n      = a_0 + n/2
    b_n      = b_0 + (y'y + m_0' Lambda_0 m_0 - m_n' Lambda_n m_n) / 2
"""
function analytic_posterior(prior, X, y)
    precision = prior.precision + X' * X
    linear = prior.precision * prior.mean + X' * y
    mean = precision \ linear
    shape = prior.shape + length(y) / 2
    rate = prior.rate + (
        dot(y, y) + dot(prior.mean, prior.precision * prior.mean) - dot(mean, linear)
    ) / 2
    return (precision = precision, mean = mean, shape = shape, rate = rate)
end

"""
    analytic_predictive(posterior, x)

Location, scale and degrees of freedom of the Student-t posterior predictive at one row.
"""
function analytic_predictive(posterior, x)
    leverage = dot(x, posterior.precision \ x)
    return (
        location = dot(x, posterior.mean),
        scale = sqrt(posterior.rate / posterior.shape * (1 + leverage)),
        df = 2 * posterior.shape,
        leverage = leverage,
    )
end

"""
    t_dof(distribution)

Degrees of freedom of a shifted, scaled Student-t.

`Distributions.dof` is defined on the standard t and not on the `LocationScale` wrapper the
package builds, so the wrapper is unwrapped here rather than in the package: the tail weight is
a property this file needs to assert on and nothing in the package needs to read.
"""
t_dof(distribution) = dof(distribution.ρ)

mm_prior(n; residual_scale = 0.02, coefficient_scale = 0.5, shape = 2.0) =
    weakly_informative_prior(
    n; residual_scale = residual_scale,
    coefficient_scale = coefficient_scale, shape = shape,
)

"""
    mm_data(n; beta, noise, seed)

A design with an intercept and a known coefficient vector, so "did it recover beta" is a
question with an answer.
"""
function mm_data(n::Int; beta = [0.001, 0.4], noise = 0.01, seed = 11)
    generator = MersenneTwister(seed)
    X = hcat(ones(n), randn(generator, n))
    y = X * beta .+ noise .* randn(generator, n)
    return X, y
end

@testset "the conjugate regression is the regression it claims to be" begin
    @testset "with no data the predictive is the prior predictive" begin
        # Not a formality. A model that quietly starts somewhere other than its prior is a
        # model whose stated beliefs are not the ones it holds.
        prior = mm_prior(2)
        model = BayesianLinearModel(prior)
        x = [1.0, 0.7]
        distribution, epistemic = predict(model, x)

        leverage = dot(x, prior.precision \ x)
        @test mean(distribution) ≈ dot(x, prior.mean) atol = 1.0e-12
        @test t_dof(distribution) ≈ 2 * prior.shape
        @test scale(distribution) ≈ sqrt(prior.rate / prior.shape * (1 + leverage))
        @test epistemic > 0
        @test epistemic < var(distribution)
    end

    @testset "the posterior is the textbook posterior, to floating point" begin
        prior = mm_prior(2)
        X, y = mm_data(200)
        model = BayesianLinearModel(prior)
        fit!(model, X, y)

        expected = analytic_posterior(prior, X, y)
        for row in ([1.0, 0.0], [1.0, 1.5], [1.0, -2.3])
            predictive = analytic_predictive(expected, row)
            distribution, epistemic = predict(model, row)
            @test mean(distribution) ≈ predictive.location rtol = 1.0e-10
            @test scale(distribution) ≈ predictive.scale rtol = 1.0e-10
            @test t_dof(distribution) ≈ predictive.df rtol = 1.0e-12
            # The epistemic share is leverage / (1 + leverage) of the total, by construction.
            @test epistemic ≈ var(distribution) * predictive.leverage /
                (1 + predictive.leverage) rtol = 1.0e-9
        end
    end

    @testset "absorbing one at a time reaches the same place as absorbing all at once" begin
        # Exchangeability. A conjugate update that is order-dependent, or that differs between
        # its batch and streaming paths, has two different models wearing one name, and the
        # backtest uses one while the live session uses the other.
        prior = mm_prior(2)
        X, y = mm_data(150; seed = 77)

        batch = BayesianLinearModel(prior)
        fit!(batch, X, y)

        streamed = BayesianLinearModel(prior)
        for index in 1:length(y)
            update!(streamed, X[index, :], y[index])
        end

        shuffled = BayesianLinearModel(prior)
        for index in shuffle(MersenneTwister(3), 1:length(y))
            update!(shuffled, X[index, :], y[index])
        end

        for row in ([1.0, 0.4], [1.0, -1.1])
            reference = predict(batch, row)[1]
            for other in (streamed, shuffled)
                distribution = predict(other, row)[1]
                @test mean(distribution) ≈ mean(reference) rtol = 1.0e-9
                @test scale(distribution) ≈ scale(reference) rtol = 1.0e-9
                @test t_dof(distribution) ≈ t_dof(reference) rtol = 1.0e-12
            end
        end
    end

    @testset "more evidence shrinks what is unknown about the coefficients" begin
        # The claim the whole design rests on: epistemic uncertainty is the part that goes
        # away with data, and aleatoric uncertainty is the part that does not.
        prior = mm_prior(2)
        X, y = mm_data(2000; noise = 0.01, seed = 5)
        row = [1.0, 1.0]
        model = BayesianLinearModel(prior)

        epistemics = Float64[]
        totals = Float64[]
        for n in (10, 50, 200, 1000, 2000)
            fresh = BayesianLinearModel(prior)
            fit!(fresh, X[1:n, :], y[1:n])
            distribution, epistemic = predict(fresh, row)
            push!(epistemics, epistemic)
            push!(totals, var(distribution))
        end
        @test issorted(epistemics, rev = true)
        @test epistemics[end] < epistemics[1] / 50
        # The total settles on a floor rather than on zero: knowing the coefficients exactly
        # still leaves the residual variance. It settles *above* the true noise here, and the
        # testset below says exactly why.
        @test totals[end] > 0.5 * 0.01^2
        @test totals[end] < 10 * 0.01^2
    end

    @testset "the prior's coefficient penalty lands in the noise estimate" begin
        # Not a coding error. It is what the normal-inverse-gamma posterior says, and it is
        # worth a test because the consequence is surprising: a strong signal gets reported
        # as noise.
        #
        #   b_n = b_0 + (y'y + m_0' L_0 m_0 - m_n' L_n m_n) / 2
        #
        # With m_0 = 0 that difference is not the residual sum of squares. It is the residual
        # sum of squares *plus* the ridge penalty the prior charges the fitted coefficients,
        # beta' L_0 beta, and E[sigma^2] = b_n / a_n carries it. Where a coefficient is large
        # against coefficient_scale, the penalty dominates and the predictive widens.
        n, noise, beta = 2000, 0.01, [0.001, 0.4]
        X, y = mm_data(n; beta = beta, noise = noise, seed = 5)
        residual_variance = var(y .- X * (X \ y))

        ratios = Float64[]
        for coefficient_scale in (0.25, 0.5, 1.0, 2.0, 5.0, 20.0)
            prior = mm_prior(2; coefficient_scale = coefficient_scale)
            model = BayesianLinearModel(prior)
            fit!(model, X, y)
            estimate = posterior_rate(model) / posterior_shape(model)
            push!(ratios, estimate / residual_variance)
            # The penalty is beta' L_0 beta and the sum of squares is n * residual variance,
            # so the inflation is predictable rather than mysterious.
            penalty = sum(abs2, X \ y) / coefficient_scale^2
            @test estimate ≈ (n * residual_variance + penalty) /
                (n + 2 * prior.shape - 2) rtol = 0.05
        end
        # A weaker prior charges less, monotonically, and stops mattering once the penalty is
        # small against the sum of squares.
        @test issorted(ratios, rev = true)
        @test ratios[1] > 10                       # coefficient_scale 0.25 inflates tenfold
        @test 0.9 < ratios[end] < 1.1              # coefficient_scale 20 does not
    end

    @testset "standardising the design is load-bearing, not preprocessing" begin
        # The model standardises its features with statistics frozen at fitting time. That
        # reads like tidying and is not: the prior's precision is a fixed number, and whether
        # it swamps the data depends entirely on the units the design is measured in.
        #
        # A raw return column has a standard deviation near 0.015, so X'X for that column is
        # about n * 0.000225, which for two thousand rows is 0.45 against a prior precision of
        # 4. The prior wins, the coefficient is shrunk to a tenth of what the data says, and
        # the model learns almost nothing while reporting no particular difficulty.
        n, noise = 2000, 0.015
        generator = MersenneTwister(13)
        feature = noise .* randn(generator, n)
        X_raw = hcat(ones(n), feature)
        y = X_raw * [0.0, 0.35] .+ noise .* randn(generator, n)
        raw_ols = X_raw \ y

        prior = mm_prior(2)
        raw_model = BayesianLinearModel(prior)
        fit!(raw_model, X_raw, y)
        raw_recovered = coefficients(raw_model)[2] / raw_ols[2]

        scaler = fit_scaler([:intercept, :feature], X_raw)
        X_scaled = reduce(
            vcat, [permutedims(transform_row(scaler, X_raw[i, :])) for i in 1:n],
        )
        scaled_model = BayesianLinearModel(prior)
        fit!(scaled_model, X_scaled, y)
        scaled_recovered = coefficients(scaled_model)[2] / (X_scaled \ y)[2]

        @test raw_recovered < 0.2            # a tenth of the relationship survives
        @test scaled_recovered > 0.95        # essentially all of it does
        @test scaled_recovered > 4 * raw_recovered
    end

    @testset "it recovers a coefficient it was not told" begin
        prior = mm_prior(2)
        beta = [0.001, 0.4]
        errors = Float64[]
        for n in (50, 500, 5000)
            X, y = mm_data(n; beta = beta, noise = 0.01, seed = 31)
            model = BayesianLinearModel(prior)
            fit!(model, X, y)
            push!(errors, abs(coefficients(model)[2] - beta[2]))
        end
        @test issorted(errors, rev = true)
        @test errors[end] < 0.005
    end

    @testset "the degrees of freedom grow with the evidence" begin
        # 2a_n = 2a_0 + n. A predictive whose tails never thin is one that never learned.
        prior = mm_prior(2; shape = 2.0)
        X, y = mm_data(400; seed = 9)
        for n in (0, 1, 30, 400)
            model = BayesianLinearModel(prior)
            n > 0 && fit!(model, X[1:n, :], y[1:n])
            @test t_dof(predict(model, [1.0, 0.2])[1]) ≈ 2 * prior.shape + n
        end
    end

    @testset "a single observation is absorbed without pretending to know anything" begin
        prior = mm_prior(2)
        model = BayesianLinearModel(prior)
        empty_scale = scale(predict(model, [1.0, 1.0])[1])
        update!(model, [1.0, 1.0], 0.05)
        distribution, epistemic = predict(model, [1.0, 1.0])
        @test isfinite(mean(distribution))
        @test isfinite(scale(distribution))
        @test scale(distribution) > 0
        @test epistemic > 0
        # One point pulls the mean a little and does not collapse the spread.
        @test abs(mean(distribution)) < 0.05
        @test scale(distribution) > empty_scale / 3
    end

    @testset "an absurd observation does not produce an absurd model" begin
        # Fat fingers, a corrupt tick, a stale price divided by the wrong divisor. The model
        # should move, and it should still be a model afterwards.
        prior = mm_prior(2)
        for outlier in (1.0e6, -1.0e6, 1.0e12)
            model = BayesianLinearModel(prior)
            X, y = mm_data(100)
            fit!(model, X, y)
            update!(model, [1.0, 1.0], outlier)
            distribution, epistemic = predict(model, [1.0, 0.5])
            @test isfinite(mean(distribution))
            @test isfinite(scale(distribution))
            @test scale(distribution) > 0
            @test isfinite(epistemic)
            @test epistemic >= 0
            # And the damage shows up as uncertainty rather than as false confidence.
            @test scale(distribution) > 1.0
        end
    end

    @testset "the rate never goes negative through rounding" begin
        # The posterior rate is a difference of large terms and can land a hair below zero on
        # data whose residuals are genuinely almost zero. A negative rate is an imaginary
        # scale, which propagates as NaN through everything downstream.
        prior = mm_prior(2)
        model = BayesianLinearModel(prior)
        X = hcat(ones(200), collect(range(-1, 1; length = 200)))
        y = X * [0.001, 0.4]                    # exactly on the line, zero residual
        fit!(model, X, y)
        @test posterior_rate(model) > 0
        distribution, epistemic = predict(model, [1.0, 0.3])
        @test isfinite(scale(distribution))
        @test scale(distribution) > 0
        @test isfinite(epistemic)
    end

    @testset "a prior with an opinion is an opinion the posterior starts from" begin
        # Every prior the package builds through weakly_informative_prior has mean zero, which
        # makes the prior-mean term in the linear system vanish and leaves that path untested.
        # A mutation that deletes it survives the whole file otherwise.
        prior = NormalInverseGammaPrior(
            [0.002, -0.5], Matrix{Float64}(I, 2, 2) .* 4.0, 2.0, 0.0004,
        )
        model = BayesianLinearModel(prior)

        # With nothing seen, the predictive is centred where the prior says.
        empty_row = [1.0, 1.0]
        empty_distribution, _ = predict(model, empty_row)
        @test mean(empty_distribution) ≈ 0.002 - 0.5
        @test coefficients(model) ≈ prior.mean
        # And the spread is the prior's spread. The rate is computed from a cancellation that
        # only vanishes when the prior mean is carried through both of its terms, so a version
        # that drops it from one of them lands here and nowhere else.
        empty_leverage = dot(empty_row, prior.precision \ empty_row)
        @test scale(empty_distribution) ≈
            sqrt(prior.rate / prior.shape * (1 + empty_leverage)) rtol = 1.0e-12
        @test posterior_rate(model) ≈ prior.rate rtol = 1.0e-12

        # One informative batch pulls it toward the data without arriving there.
        X, y = mm_data(40; beta = [0.001, 0.4], noise = 0.01, seed = 19)
        fit!(model, X, y)
        pulled = coefficients(model)[2]
        @test pulled > prior.mean[2]           # moved toward +0.4
        @test pulled < 0.4                     # and has not got there yet
        @test posterior_rate(model) > 0

        # More of the same data carries it the rest of the way.
        X_long, y_long = mm_data(4000; beta = [0.001, 0.4], noise = 0.01, seed = 19)
        far = BayesianLinearModel(prior)
        fit!(far, X_long, y_long)
        @test abs(coefficients(far)[2] - 0.4) < abs(pulled - 0.4)
        @test abs(coefficients(far)[2] - 0.4) < 0.01
    end

    @testset "the rate clamp is load-bearing, not decoration" begin
        # The posterior rate is a difference of large terms. On a design of large magnitude
        # whose response lies exactly on the fitted plane, the cancellation is the whole
        # quantity and it lands below zero. Unclamped that is a negative variance, and its
        # square root is the NaN that propagates through every prediction afterwards.
        prior = NormalInverseGammaPrior(
            [0.0, 0.0], Matrix{Float64}(I, 2, 2) .* 1.0e-12, 2.0, 1.0e-12,
        )
        for magnitude in (1.0e6, 1.0e8, 1.0e10)
            model = BayesianLinearModel(prior)
            X = hcat(
                fill(magnitude, 300), magnitude .* collect(range(-1, 1; length = 300)),
            )
            fit!(model, X, X * [1.0, 2.0])

            # The unclamped expression, computed here the way the implementation computes it.
            linear = prior.precision * prior.mean + model.xy
            unclamped = prior.rate + (
                dot(prior.mean, prior.precision * prior.mean) + model.yy -
                    dot(coefficients(model), linear)
            ) / 2
            @test unclamped < 0                          # the cancellation really goes negative
            @test posterior_rate(model) > 0              # and the clamp catches it
            distribution, epistemic = predict(model, [magnitude, 0.0])
            @test isfinite(mean(distribution))
            @test isfinite(scale(distribution))
            @test !isnan(epistemic)
        end
    end

    @testset "reset returns the model to its prior and nowhere else" begin
        prior = mm_prior(2)
        model = BayesianLinearModel(prior)
        before = predict(model, [1.0, 0.6])
        X, y = mm_data(300)
        fit!(model, X, y)
        @test predict(model, [1.0, 0.6])[1] != before[1]
        reset!(model)
        after = predict(model, [1.0, 0.6])
        @test mean(after[1]) ≈ mean(before[1])
        @test scale(after[1]) ≈ scale(before[1])
        @test t_dof(after[1]) ≈ t_dof(before[1])
    end

    @testset "forgetting bounds how much the past can ever count for" begin
        # With a discount the effective sample size approaches 1/(1-lambda) rather than
        # growing without limit, which is what stops a five-year-old regime from outvoting
        # this month.
        prior = mm_prior(2)
        X, y = mm_data(4000; seed = 44)
        for forgetting in (0.99, 0.95)
            model = BayesianLinearModel(prior; forgetting = forgetting)
            fit!(model, X, y)
            ceiling = 1 / (1 - forgetting)
            @test t_dof(predict(model, [1.0, 0.0])[1]) < 2 * prior.shape + ceiling + 1
        end
        # Without one it grows with every observation.
        plain = BayesianLinearModel(prior)
        fit!(plain, X, y)
        @test t_dof(predict(plain, [1.0, 0.0])[1]) ≈ 2 * prior.shape + 4000
    end
end
