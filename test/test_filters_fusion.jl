# Sections 5, 6, 7 and 8 of the validation brief: the volatility filter, the regime chain, the
# fusion layer and the disagreement diagnostic.
#
# Each model is driven with a synthetic sequence whose right answer is known by construction,
# so "did it respond correctly" is a question with an answer rather than a matter of opinion.

ff_prior(; volatility = 0.015, shape = 2.0) =
    variance_prior(volatility_scale = volatility, shape = shape)

"""
    drive!(filter, returns)

Absorb a sequence and return the predictive standard deviation after each one.
"""
function drive!(filter::DiscountedVarianceFilter, returns)
    spreads = Float64[]
    for value in returns
        # The filter absorbs a variance estimate, not a return. With the centre at zero that
        # estimate is the squared return, which is the unbiased one-observation estimate the
        # inverse-gamma kernel is conjugate to.
        observe_variance!(filter, (value - filter.centre)^2)
        push!(spreads, sqrt(noise_variance(filter)))
    end
    return spreads
end

quiet(n, seed) = 0.004 .* randn(MersenneTwister(seed), n)
violent(n, seed) = 0.04 .* randn(MersenneTwister(seed), n)

@testset "the variance filter tracks the volatility it is shown" begin
    @testset "a quiet market and a violent one are told apart" begin
        calm = DiscountedVarianceFilter(ff_prior())
        storm = DiscountedVarianceFilter(ff_prior())
        drive!(calm, quiet(400, 1))
        drive!(storm, violent(400, 1))

        @test sqrt(noise_variance(calm)) < 0.008
        @test sqrt(noise_variance(storm)) > 0.03
        @test sqrt(noise_variance(storm)) > 4 * sqrt(noise_variance(calm))
    end

    @testset "a shock is absorbed quickly and a collapse is too" begin
        # The claim the posterior over the forgetting rate exists to support. A single fixed
        # decay is either too slow for the shock or too jumpy for the calm; a mixture should
        # be neither.
        filter = DiscountedVarianceFilter(ff_prior())
        drive!(filter, quiet(300, 2))
        before = sqrt(noise_variance(filter))
        shocked = drive!(filter, violent(60, 3))
        @test before < 0.01
        @test shocked[end] > 3 * before
        # Most of the move happens early rather than after two hundred bars.
        @test shocked[20] > before + 0.6 * (shocked[end] - before)

        collapsed = drive!(filter, quiet(200, 4))
        @test collapsed[end] < shocked[end] / 2
    end

    @testset "the weights move toward forgetting faster when the world changes" begin
        # Readable off the mixture: the components that forget quickly earn weight when the
        # level shifts, and the component that never forgets earns it when nothing does.
        steady = DiscountedVarianceFilter(ff_prior())
        drive!(steady, quiet(600, 5))
        steady_discount = expected_discount(steady)

        shifting = DiscountedVarianceFilter(ff_prior())
        for round in 1:6
            drive!(shifting, isodd(round) ? quiet(50, 10 + round) : violent(50, 20 + round))
        end
        shifting_discount = expected_discount(shifting)

        @test shifting_discount < steady_discount

        # Evidence has to *accumulate*. A grid that re-weights from the latest bar alone still
        # produces a plausible-looking expected discount and never learns anything: it is
        # equally uncertain after six hundred bars as after twenty. The mutation that freezes
        # the carried-forward term survives every other assertion in this file.
        #
        # Fed a constant rather than a random sequence, because the property is about how
        # evidence accumulates and not about any particular draw. `randn` does not produce the
        # same stream on every Julia version, and the first version of this assertion passed
        # on 1.10 and failed on 1.12 for that reason alone.
        long_run = DiscountedVarianceFilter(ff_prior())
        short_run = DiscountedVarianceFilter(ff_prior())
        for _ in 1:600
            observe_variance!(long_run, 0.004^2)
        end
        for _ in 1:20
            observe_variance!(short_run, 0.004^2)
        end
        @test discount_entropy(long_run) < discount_entropy(short_run) - 0.5
        @test maximum(long_run.log_weights) - minimum(long_run.log_weights) > 5
        @test 0 < shifting_discount <= 1
        @test 0 < steady_discount <= 1
        # And the mixture says how sure it is about that, rather than only what it picked.
        @test discount_entropy(shifting) >= 0
        @test discount_disagreement(shifting) >= 0
    end

    @testset "uncertainty is never suppressed to nothing" begin
        # Two structural guarantees the design leans on: the shape only ever grows above the
        # prior, so the predictive always has a finite mean and variance, and the rate floor
        # is unreachable rather than merely defensive.
        prior = ff_prior()
        filter = DiscountedVarianceFilter(prior)
        # The guarantee is structural, so it is asserted exactly rather than loosely: the
        # shape starts at the prior's and only ever grows, which is what keeps df above two
        # and the predictive mean and variance finite.
        #
        # Two shapes exist and they answer different questions. The filtered one says how
        # volatile it is now; the evolved one says how volatile the next bar will be. Both are
        # pinned, because a mutation to either survives a test that only reads the other.
        @test predictive_df(filter) ≈ 2 * prior.shape          # evolved shape
        @test predictive_df(filter) > 2
        # An inverse-gamma with shape a and rate b has mean b/(a-1), and with nothing absorbed
        # that is exactly the prior's own answer.
        @test noise_variance(filter) ≈ prior.rate / (prior.shape - 1)   # filtered shape
        @test sqrt(noise_variance(filter)) ≈ 0.015 rtol = 1.0e-9
        for component in 1:length(DEFAULT_DISCOUNTS)
            @test posterior_shape(filter, component) ≈ prior.shape
            # And the rate floor never binds, which is the claim the module makes for itself:
            # the prior sits outside the discounted statistics, so the rate can only ever be
            # the prior's rate plus something non-negative. Removing the clamp changes nothing,
            # and that is a property worth stating rather than a gap in the tests.
            @test posterior_rate(filter, component) >= prior.rate
        end
        for value in vcat(quiet(500, 6), zeros(50), violent(50, 7))
            observe_variance!(filter, value^2)
            # Above two always. Not monotone, and it should not be: weight moving to a
            # faster-forgetting component genuinely lowers the effective sample size, which
            # is the mixture doing its job rather than losing information.
            @test predictive_df(filter) > 2
            @test noise_variance(filter) > 0
            @test isfinite(noise_variance(filter))
            @test volatility_uncertainty(filter) > 0
            for component in 1:length(DEFAULT_DISCOUNTS)
                @test posterior_rate(filter, component) >= prior.rate
            end
        end
        # A flat bar is an ordinary observation here, not a degenerate case.
        flat = DiscountedVarianceFilter(ff_prior())
        for _ in 1:200
            observe_variance!(flat, 0.0)          # a flat bar is an ordinary observation
        end
        @test noise_variance(flat) > 0
        @test isfinite(expected_volatility(flat))
    end

    @testset "a skipped bar ages the filter without inventing a return" begin
        # Substituting zero would be evidence for the quiet state, and inventing evidence from
        # a gap in the data is how a model becomes confidently wrong.
        skipped = DiscountedVarianceFilter(ff_prior())
        substituted = DiscountedVarianceFilter(ff_prior())
        returns = violent(200, 8)
        for value in returns
            observe_variance!(skipped, value^2)
            observe_variance!(substituted, value^2)
        end
        for _ in 1:50
            skip_observation!(skipped)
            observe_variance!(substituted, 0.0)   # what substituting a zero return would do
        end
        @test noise_variance(skipped) > noise_variance(substituted)
        @test skipped.n_skipped == 50
        @test skipped.n_seen == length(returns)
    end

    @testset "an absurd return leaves a filter, not a wreck" begin
        for outlier in (1.0, -1.0, 100.0)
            filter = DiscountedVarianceFilter(ff_prior())
            drive!(filter, quiet(200, 9))
            observe_variance!(filter, outlier^2)
            @test isfinite(noise_variance(filter))
            @test noise_variance(filter) > 0
            @test predictive_df(filter) > 2
            @test isfinite(expected_volatility(filter))
        end
    end
end

# --------------------------------------------------------------------------------------------

"""
    regime_run(; drift, volatility, n, seed)

A stretch of returns from one regime, with the drift and spread that regime is meant to have.
"""
regime_run(; drift, volatility, n, seed) =
    drift .+ volatility .* randn(MersenneTwister(seed), n)

# The prior fixes what each state means: drift shapes (+1, -1, 0) and log-variance shapes
# (-0.5, +1.5, 0). Bull is therefore a rising *and quiet* market, bear a falling and violent
# one, sideways the middle on both. A synthetic bull with the lowest variance of the three
# gets scored as sideways, and correctly so.
bull_run(n, seed) = regime_run(drift = 0.004, volatility = 0.006, n = n, seed = seed)
bear_run(n, seed) = regime_run(drift = -0.005, volatility = 0.02, n = n, seed = seed)
flat_run(n, seed) = regime_run(drift = 0.0, volatility = 0.01, n = n, seed = seed)

"""
    fitted_filter(returns)

A regime filter with its parameters estimated from a stretch containing all three states, so
the states are separated by the data rather than by the prior alone.
"""
function fitted_filter(returns)
    parameters = estimate_regime_parameters(returns; prior = RegimePrior())
    return RegimeFilter(parameters)
end

leader(filter) = argmax(regime_probabilities(filter))

@testset "the regime chain finds the regime it is in" begin
    training = vcat(
        bull_run(300, 31), bear_run(300, 32), flat_run(300, 33),
        bull_run(300, 34), bear_run(300, 35),
    )

    @testset "each state is identified by what it looks like, not by its index" begin
        # State identity is fixed by the prior's shape vectors, which is what makes the model
        # immune to label switching: whichever component the data lands on, bull is the one
        # with the upward drift.
        filter = fitted_filter(training)
        for (label, sequence) in (
                (1, bull_run(200, 41)), (2, bear_run(200, 42)), (3, flat_run(200, 43)),
            )
            fresh = fitted_filter(training)
            for value in sequence
                observe_return!(fresh, value)
            end
            @test leader(fresh) == label
            @test regime_probabilities(fresh)[label] > 0.5
        end
    end

    @testset "the belief is always a distribution" begin
        filter = fitted_filter(training)
        for value in vcat(training, bull_run(400, 44), bear_run(400, 45))
            observe_return!(filter, value)
            belief = regime_probabilities(filter)
            @test length(belief) == N_REGIMES
            @test all(isfinite, belief)
            @test all(>=(0), belief)
            @test sum(belief) ≈ 1 atol = 1.0e-10
        end
        @test isfinite(filter.log_likelihood)
    end

    @testset "it switches rather than staying where it was" begin
        # The failure this rules out is a chain so sticky it never leaves its first state,
        # which looks stable and is useless.
        filter = fitted_filter(training)
        for value in bull_run(300, 51)
            observe_return!(filter, value)
        end
        @test leader(filter) == 1

        switched = 0
        for (index, value) in enumerate(bear_run(200, 52))
            observe_return!(filter, value)
            leader(filter) == 2 && (switched = index; break)
        end
        @test switched > 0
        @test switched < 100        # within a hundred bars, not eventually
    end

    @testset "stickiness is a parameter and it does what it says" begin
        # A = lambda I + (1 - lambda) 1 pi'. One number, no EM, and the diagonal has to
        # dominate or the chain is not a regime model at all.
        for persistence in (0.9, 0.98)
            prior = RegimePrior(persistence = persistence)
            matrix = regime_transition(persistence, collect(prior.stationary))
            @test size(matrix) == (N_REGIMES, N_REGIMES)
            for row in 1:N_REGIMES
                @test sum(matrix[row, :]) ≈ 1
                @test all(>=(0), matrix[row, :])
                @test matrix[row, row] == maximum(matrix[row, :])
            end
        end
        loose = regime_transition(0.9, collect(RegimePrior().stationary))
        tight = regime_transition(0.99, collect(RegimePrior().stationary))
        for row in 1:N_REGIMES
            @test tight[row, row] > loose[row, row]
        end
    end

    @testset "a skipped bar propagates the chain and nothing more" begin
        filter = fitted_filter(training)
        for value in bull_run(200, 61)
            observe_return!(filter, value)
        end
        before = regime_probabilities(filter)
        skip_observation!(filter)
        after = regime_probabilities(filter)
        @test sum(after) ≈ 1
        @test after != before                       # the chain aged
        @test filter.n_skipped == 1
        # It moves toward the stationary distribution rather than toward any observation.
        stationary = collect(filter.parameters.prior.stationary)
        @test sum(abs, after .- stationary) < sum(abs, before .- stationary)
    end

    @testset "a long sequence does not accumulate into nonsense" begin
        # The forward recursion runs in log space for exactly this reason.
        filter = fitted_filter(training)
        for round in 1:20
            for value in vcat(bull_run(100, 70 + round), bear_run(100, 90 + round))
                observe_return!(filter, value)
            end
        end
        belief = regime_probabilities(filter)
        @test all(isfinite, belief)
        @test sum(belief) ≈ 1 atol = 1.0e-10
        @test isfinite(filter.log_likelihood)
        @test filter.n_seen == 20 * 200
    end
end

# --------------------------------------------------------------------------------------------

"""
    stub_result(name; mean, sd, epistemic, n)

A model result with a chosen predictive, so fusion can be driven with agreement and
disagreement that were arranged rather than hoped for.
"""
function stub_result(name::ModelName; mean::Real, sd::Real, epistemic::Real = 1.0e-6, n = 500)
    return ProbabilisticResult(
        model = ModelVersion(
            name = name, version = v"1.0.0", fitted_at = DateTime(2026, 1, 1),
            params_hash = string(name, "-hash"),
        ),
        symbol = "FUSE",
        as_of = DateTime(2026, 3, 2, 15, 30),
        horizon_bars = 1,
        distribution = Normal(Float64(mean), Float64(sd)),
        n_observations = n,
        epistemic_variance = Float64(epistemic),
    )
end

ff_fuse(results...; forgetting = 0.99) = fuse(
    ModelReliability(
        ModelName[result.model.name for result in results];
        forgetting = forgetting
    ),
    Tuple(results),
)

@testset "fusion behaves when the models do not" begin
    @testset "agreement leaves the pool where the models are" begin
        agreed = ff_fuse(
            stub_result(MOMENTUM; mean = 0.004, sd = 0.015),
            stub_result(VOLATILITY; mean = 0.004, sd = 0.015),
            stub_result(REGIME; mean = 0.004, sd = 0.015),
        )
        @test mean(agreed) ≈ 0.004
        @test agreed.diagnostics[:disagreement] ≈ 0 atol = 1.0e-15
        @test disagreement_share(agreed) < 1.0e-9
        @test model_agreement(agreed) === AGREEMENT_HIGH
    end

    @testset "disagreement widens the pool beyond any single model" begin
        # The property that makes a linear opinion pool the right choice here: the mixture is
        # wider than its components when they disagree, so conflict shows up as uncertainty
        # instead of being averaged away.
        split = ff_fuse(
            stub_result(MOMENTUM; mean = 0.02, sd = 0.01),
            stub_result(VOLATILITY; mean = -0.02, sd = 0.01),
        )
        @test std(split) > 0.01
        @test split.diagnostics[:disagreement] > 0
        @test disagreement_share(split) > 0.5
        @test model_agreement(split) === AGREEMENT_LOW
        # And the disagreement is counted as reducible, because which model is right is
        # something evidence can settle.
        @test split.epistemic_variance >= split.diagnostics[:disagreement]
    end

    @testset "the agreement label spans its range and is ordered" begin
        levels = Agreement[]
        for gap in (0.0, 0.006, 0.02)
            fused = ff_fuse(
                stub_result(MOMENTUM; mean = gap, sd = 0.012),
                stub_result(VOLATILITY; mean = -gap, sd = 0.012),
            )
            push!(levels, model_agreement(fused))
        end
        @test levels == [AGREEMENT_HIGH, AGREEMENT_MEDIUM, AGREEMENT_LOW]
        # A share is a share: bounded, whatever the units of the returns.
        for scale in (1.0e-4, 1.0, 1.0e4)
            fused = ff_fuse(
                stub_result(MOMENTUM; mean = scale, sd = scale / 2),
                stub_result(VOLATILITY; mean = -scale, sd = scale / 2),
            )
            @test 0 <= disagreement_share(fused) <= 1
        end
    end

    @testset "one model is enough and says so" begin
        alone = ff_fuse(stub_result(MOMENTUM; mean = 0.003, sd = 0.014))
        @test mean(alone) ≈ 0.003
        @test alone.diagnostics[:n_models] == 1
        @test alone.diagnostics[:disagreement] ≈ 0 atol = 1.0e-15
        @test only(alone.weights.probabilities) ≈ 1
        @test model_agreement(alone) === AGREEMENT_HIGH
    end

    @testset "an unusable model is not a confident one" begin
        # The failure the brief names: an unavailable or invalid model treated as a confident
        # prediction. A model that cannot say anything has to widen the pool, never narrow it.
        confident = stub_result(MOMENTUM; mean = 0.004, sd = 0.01)
        useless = stub_result(VOLATILITY; mean = 0.0, sd = 5.0, epistemic = 25.0)
        fused = ff_fuse(confident, useless)
        @test std(fused) > std(confident.distribution)
        @test std(fused) > 1.0
        # Note what agreement does *not* say here. It reports AGREEMENT_HIGH, and correctly:
        # the two models agree about where the return will be and differ only in how sure they
        # are. Agreement is about location, confidence is a separate axis, and a reader who
        # takes a high agreement label as "the models are confident" has read it wrong. That
        # is the reason it is a diagnostic and never a gate.
        @test model_agreement(fused) === AGREEMENT_HIGH
        @test disagreement_share(fused) < 0.01
        @test fused.epistemic_variance > confident.epistemic_variance
    end

    @testset "weights stay a distribution however they are pushed" begin
        reliability = ModelReliability(ModelName[MOMENTUM, VOLATILITY, REGIME])
        results = (
            stub_result(MOMENTUM; mean = 0.004, sd = 0.012),
            stub_result(VOLATILITY; mean = 0.001, sd = 0.02),
            stub_result(REGIME; mean = -0.002, sd = 0.03),
        )
        for round in 1:2000
            # A run of outcomes that only ever suits the first model, which is the pressure
            # that would send a weight to zero or a log-weight to minus infinity.
            score_fusion!(reliability, results, 0.004)
            probabilities = fuse(reliability, results).weights.probabilities
            @test all(isfinite, probabilities)
            @test all(>=(0), probabilities)
            @test sum(probabilities) ≈ 1 atol = 1.0e-12
        end
        final = fuse(reliability, results).weights.probabilities
        @test final[1] == maximum(final)
        # Concentrated, but never exactly zero: a model on zero weight can never earn its way
        # back, and the pool would have silently become a single model.
        @test all(>(0), final)
    end

    @testset "when no model can be weighted the pool says so, uniformly" begin
        # The branch that runs when every log weight has gone to minus infinity, which is what
        # a long enough run of impossible outcomes produces. Collapsing onto whichever model
        # happens to be first would be a silent, arbitrary choice presented as a decision.
        reliability = ModelReliability(ModelName[MOMENTUM, VOLATILITY, REGIME])
        fill!(reliability.log_weights, -Inf)
        results = (
            stub_result(MOMENTUM; mean = 0.004, sd = 0.012),
            stub_result(VOLATILITY; mean = 0.001, sd = 0.02),
            stub_result(REGIME; mean = -0.002, sd = 0.03),
        )
        fused = fuse(reliability, results)
        probabilities = fused.weights.probabilities
        @test all(≈(1 / 3), probabilities)
        @test sum(probabilities) ≈ 1
        @test all(isfinite, probabilities)
        # Uniform is the honest answer: it has no idea which model to believe, and the pool
        # is wide because all three are in it.
        @test std(fused) > std(results[1].distribution)
    end

    @testset "the pool is scored on the outcome, never before it" begin
        # Weights move only when score_fusion! is called, which the replay calls after the bar
        # is realised. Fusing repeatedly without scoring must change nothing.
        reliability = ModelReliability(ModelName[MOMENTUM, VOLATILITY])
        results = (
            stub_result(MOMENTUM; mean = 0.004, sd = 0.012),
            stub_result(VOLATILITY; mean = -0.004, sd = 0.012),
        )
        first_pass = fuse(reliability, results).weights.probabilities
        for _ in 1:50
            fuse(reliability, results)
        end
        @test fuse(reliability, results).weights.probabilities == first_pass
        score_fusion!(reliability, results, 0.004)
        @test fuse(reliability, results).weights.probabilities != first_pass
    end
end
