@testset "probabilistic layer" begin
    @testset "student-t predictive" begin
        d = student_t(0.0, 1.0, 5.0)
        @test mean(d) ≈ 0.0
        @test var(d) ≈ 5 / 3
        @test_throws ArgumentError student_t(0.0, 0.0, 5.0)
        @test_throws ArgumentError student_t(0.0, 1.0, 0.0)
    end

    @testset "heavier tails than a variance-matched normal" begin
        # The property that justifies the choice: an unknown variance puts materially more
        # mass on a large adverse move, and understating that is the one failure this
        # system cannot afford.
        t = student_t(0.0, 1.0, 4.0)
        matched = Normal(0.0, std(t))
        @test probability_loss_exceeds(t, 6.0) > probability_loss_exceeds(matched, 6.0)
    end

    @testset "credible interval matches the analytic z-score" begin
        interval = credible_interval(Normal(0.0, 1.0), 0.95)
        @test interval.lower ≈ -1.959963985 atol = 1.0e-6
        @test interval.upper ≈ 1.959963985 atol = 1.0e-6
        @test width(interval) ≈ 2 * 1.959963985 atol = 1.0e-6
        @test 0.0 in interval
        @test !(3.0 in interval)
        @test_throws ArgumentError CredibleInterval(1.0, -1.0, 0.95)
        @test_throws ArgumentError credible_interval(Normal(), 1.0)
    end

    @testset "tail probabilities" begin
        d = Normal(0.0, 0.02)
        @test probability_positive(d) ≈ 0.5
        @test probability_above(d, 0.02) ≈ 1 - cdf(d, 0.02)
        @test probability_below(d, -0.02) ≈ cdf(d, -0.02)
        @test probability_loss_exceeds(d, 0.05) ≈ cdf(d, -0.05)
        @test_throws ArgumentError probability_loss_exceeds(d, -0.01)
    end

    @testset "a higher mean lowers the probability of a large loss" begin
        low = Normal(0.0, 0.1)
        high = Normal(0.05, 0.1)
        @test probability_loss_exceeds(high, 0.05) < probability_loss_exceeds(low, 0.05)
    end

    @testset "mixtures keep what an average would hide" begin
        # Two confident components pulling apart look identical to one vague component
        # once averaged, and those are very different situations for a risk engine.
        split = MixtureModel([Normal(0.06, 0.01), Normal(-0.06, 0.01)], [0.5, 0.5])
        matched = Normal(mean(split), std(split))
        @test logpdf(split, 0.0) < logpdf(matched, 0.0)
        @test logpdf(split, 0.06) > logpdf(matched, 0.06)
    end

    @testset "mixture moments against a large sample" begin
        rng = MersenneTwister(3)
        m = MixtureModel([Normal(0.02, 0.02), Normal(-0.03, 0.05)], [0.7, 0.3])
        draws = rand(rng, m, 400_000)
        @test mean(m) ≈ mean(draws) atol = 5.0e-4
        @test std(m) ≈ std(draws) rtol = 0.01
        @test probability_positive(m) ≈ count(>(0), draws) / length(draws) atol = 0.005
    end

    @testset "labelled categorical" begin
        c = LabelledCategorical(BULL => 0.68, SIDEWAYS => 0.21, BEAR => 0.11)
        @test most_likely(c) === BULL
        @test probability_of(c, BULL) ≈ 0.68
        @test probability_of(c, HIGH_VOLATILITY) == 0.0
        @test sum(c.probabilities) ≈ 1.0
    end

    @testset "categorical rejects what is not a distribution" begin
        @test_throws ArgumentError LabelledCategorical(BULL => 0.6, BEAR => 0.6)
        @test_throws ArgumentError LabelledCategorical(BULL => 1.2, BEAR => -0.2)
        @test_throws ArgumentError LabelledCategorical(BULL => 0.5, BULL => 0.5)
        @test_throws ArgumentError LabelledCategorical(Regime[], Float64[])
    end

    @testset "entropy is normalised so it compares across state counts" begin
        three = uniform_categorical([BULL, BEAR, SIDEWAYS])
        two = uniform_categorical([HIGH_VOLATILITY, LOW_VOLATILITY])
        @test entropy(three) ≈ log(3)
        @test normalised_entropy(three) ≈ 1.0
        @test normalised_entropy(two) ≈ 1.0
        certain = LabelledCategorical(BULL => 1.0, BEAR => 0.0)
        @test entropy(certain) ≈ 0.0
        @test normalised_entropy(certain) ≈ 0.0
        @test normalised_entropy(uniform_categorical([BULL])) == 0.0
    end

    @testset "normalise" begin
        @test normalise([2.0, 2.0]) ≈ [0.5, 0.5]
        @test sum(normalise([1.0, 2.0, 7.0])) ≈ 1.0
        @test_throws ArgumentError normalise([0.0, 0.0])
    end

    @testset "model version" begin
        version = ModelVersion(
            name = VOLATILITY, version = v"1.2.3", params_hash = "a1b2c3d4e5f6",
        )
        @test identifier(version) == "volatility@1.2.3+a1b2c3d4"
        @test identifier(ModelVersion(name = MOMENTUM, version = v"0.1.0")) ==
            "momentum@0.1.0"
        @test_throws ArgumentError ModelVersion(
            name = MOMENTUM, version = v"0.1.0",
            train_start = DateTime(2026, 1, 2), train_end = DateTime(2026, 1, 1),
        )
    end

    @testset "probabilistic result carries its own provenance" begin
        version = ModelVersion(name = MOMENTUM, version = v"0.1.0")
        result = ProbabilisticResult(
            model = version, symbol = "RELIANCE", as_of = DateTime(2026, 1, 2),
            horizon_bars = 5, distribution = Normal(0.02, 0.05),
            n_observations = 250, epistemic_variance = 0.0005,
        )
        @test mean(result) ≈ 0.02
        @test var(result) ≈ 0.0025
        @test probability_positive(result) > 0.6
        @test 0.02 in credible_interval(result)
        @test epistemic_share(result) ≈ 0.2
    end

    @testset "result rejects what cannot be true" begin
        version = ModelVersion(name = MOMENTUM, version = v"0.1.0")
        make(; kwargs...) = ProbabilisticResult(;
            model = version, symbol = "R", as_of = DateTime(2026, 1, 2),
            horizon_bars = 1, distribution = Normal(), kwargs...,
        )
        @test_throws ArgumentError make(n_observations = -1)
        @test_throws ArgumentError make(epistemic_variance = -1.0)
        @test_throws ArgumentError ProbabilisticResult(
            model = version, symbol = "R", as_of = DateTime(2026, 1, 2),
            horizon_bars = 0, distribution = Normal(),
        )
        @test_throws ArgumentError ProbabilisticResult(
            model = version, symbol = "", as_of = DateTime(2026, 1, 2),
            horizon_bars = 1, distribution = Normal(),
        )
    end

    @testset "epistemic share is bounded and safe on degenerate spreads" begin
        version = ModelVersion(name = MOMENTUM, version = v"0.1.0")
        degenerate = ProbabilisticResult(
            model = version, symbol = "R", as_of = DateTime(2026, 1, 2),
            horizon_bars = 1, distribution = Normal(0.0, 0.0),
            epistemic_variance = 1.0,
        )
        @test epistemic_share(degenerate) == 0.0

        heavy = ProbabilisticResult(
            model = version, symbol = "R", as_of = DateTime(2026, 1, 2),
            horizon_bars = 1, distribution = student_t(0.0, 1.0, 1.5),
            epistemic_variance = 1.0,
        )
        @test epistemic_share(heavy) == 0.0
    end

    @testset "the result type follows its distribution" begin
        version = ModelVersion(name = REGIME, version = v"0.1.0")
        mixture = MixtureModel([Normal(0.02, 0.02), Normal(-0.03, 0.05)], [0.7, 0.3])
        result = ProbabilisticResult(
            model = version, symbol = "R", as_of = DateTime(2026, 1, 2),
            horizon_bars = 1, distribution = mixture,
        )
        @test result isa ProbabilisticResult{typeof(mixture)}
        @test mean(result) ≈ mean(mixture)
    end
end
