# The regime filter, checked against a brute-force oracle rather than against itself.
#
# The forward recursion has a closed-form answer that can be computed a second way: enumerate
# every state path, weight each by its own probability, and marginalise. That is exponential
# and useless in production, which is exactly what makes it a good oracle over seven bars.

const RF_TRUE_PERSISTENCE = 0.94732

rf_prior(; kwargs...) = RegimePrior(; kwargs...)

function rf_returns(; n_bars = 6_000, seed = 11, process = RegimeSwitchingReturns())
    series = generate_series(
        process; symbol = "RELIANCE", n_bars = n_bars, seed = seed, start = Date(2005, 1, 1),
    )
    return convert(Vector{Float64}, true_log_returns(series)), series
end

function rf_fitted(; n_bars = 6_000, seed = 11, process = RegimeSwitchingReturns())
    returns, series = rf_returns(; n_bars = n_bars, seed = seed, process = process)
    filter = RegimeFilter(estimate_regime_parameters(returns))
    fit_filter!(filter, returns)
    return filter, returns, series
end

"""
Brute force: every path through the chain, weighted by its own probability.
"""
function rf_enumerate(parameters::RegimeParameters, observations::Vector{Float64})
    transitions = transition_matrix(parameters)
    stationary = parameters.prior.stationary
    n = length(observations)
    posterior = zeros(Float64, N_REGIMES)
    evidence = 0.0

    for code in 0:(N_REGIMES^n - 1)
        path = Vector{Int}(undef, n)
        rest = code
        for position in 1:n
            path[position] = rest % N_REGIMES + 1
            rest ÷= N_REGIMES
        end

        weight = stationary[path[1]]
        for position in 2:n
            weight *= transitions[path[position - 1], path[position]]
        end
        for position in 1:n
            weight *= pdf(emission(parameters, path[position]), observations[position])
        end
        evidence += weight
        posterior[path[n]] += weight
    end
    return posterior ./ evidence, evidence
end

@testset "regime filter" begin
    @testset "the prior fixes what the states mean" begin
        prior = rf_prior()
        @test length(prior.stationary) == N_REGIMES
        @test sum(prior.stationary) ≈ 1.0
        @test REGIME_STATES == (BULL, BEAR, SIDEWAYS)

        # Mean zero and unit variance under the stationary distribution, both load-bearing:
        # the first makes the state drifts average to the sample mean, the second makes the
        # estimated spread mean what it says.
        for shape in (prior.drift_shape, prior.variance_shape)
            @test sum(prior.stationary .* shape) ≈ 0.0 atol = 1.0e-12
            @test sum(prior.stationary .* shape .^ 2) ≈ 1.0
        end

        # The sign pattern is what pins state identity and rules out label switching.
        @test prior.drift_shape[1] > 0
        @test prior.drift_shape[2] < 0
        @test prior.variance_shape[2] == maximum(prior.variance_shape)

        @test_throws ArgumentError rf_prior(stationary = (0.5, 0.5, 0.5))
        @test_throws ArgumentError rf_prior(stationary = (0.5, -0.1, 0.6))
        @test_throws ArgumentError rf_prior(persistence = 1.0)
        @test_throws ArgumentError rf_prior(shape = 0.5)
        @test_throws ArgumentError rf_prior(drift_values = (1.0, 1.0, 1.0))
        @test_throws ArgumentError rf_prior(drift_values = (-1.0, 1.0, 0.0))
        @test rf_prior(expected_duration = 20).persistence ≈ 0.95
    end

    @testset "the transition matrix needs no eigenproblem" begin
        stationary = [0.35, 0.25, 0.4]
        for persistence in (0.0, 0.5, 0.9, 0.99)
            matrix = regime_transition(persistence, stationary)
            for row in 1:N_REGIMES
                @test sum(matrix[row, :]) ≈ 1.0
                @test all(matrix[row, :] .>= 0)
            end
            # Checked against the package's own LU solve, written independently for the
            # generator and knowing nothing about this family.
            @test stationary_distribution(matrix) ≈ stationary
            @test sort(abs.(eigvals(matrix)); rev = true)[2] ≈ persistence atol = 1.0e-12
        end

        # A^h in closed form, which is what makes horizon propagation O(K) and exact.
        persistence = 0.9
        matrix = regime_transition(persistence, stationary)
        for horizon in (1, 3, 25, 250)
            powered = matrix^horizon
            held = persistence^horizon
            expected = held * Matrix{Float64}(I, N_REGIMES, N_REGIMES) +
                (1 - held) * ones(N_REGIMES) * stationary'
            @test powered ≈ expected atol = 1.0e-12
        end
    end

    @testset "the forward recursion is the exact posterior" begin
        # Against every path through the chain, marginalised by brute force. Seven bars is
        # 2187 paths, which is affordable exactly once.
        returns, _ = rf_returns(; n_bars = 400)
        parameters = estimate_regime_parameters(returns)
        observations = returns[1:7]

        filter = RegimeFilter(parameters)
        for value in observations
            observe_return!(filter, value)
        end
        expected, evidence = rf_enumerate(parameters, observations)

        @test regime_probabilities(filter) ≈ expected atol = 1.0e-14
        @test filter.log_likelihood ≈ log(evidence) rtol = 1.0e-12
        @test sum(regime_probabilities(filter)) ≈ 1.0
    end

    @testset "it stays a probability distribution for twenty thousand bars" begin
        # The unnormalised recursion underflows within a few hundred bars and then reports
        # the prior forever, silently. This is the test that would catch that.
        returns, _ = rf_returns(; n_bars = 20_000, seed = 3)
        filter = RegimeFilter(estimate_regime_parameters(returns))
        violations = 0
        for value in returns
            observe_return!(filter, value)
            belief = regime_probabilities(filter)
            (isapprox(sum(belief), 1.0; atol = 1.0e-12) && all(isfinite, belief)) ||
                (violations += 1)
        end
        @test violations == 0
        @test isfinite(filter.log_likelihood)
        @test n_absorbed(filter) == 20_000
    end

    @testset "estimation is closed form and lands near the truth" begin
        # Every scalar is checked against a parameter the generator actually knows.
        process = RegimeSwitchingReturns()
        truth = deannualise.(process.annual_volatilities) .^ 2

        persistences = Float64[]
        for seed in 1:6
            returns, _ = rf_returns(; n_bars = 10_000, seed = seed)
            parameters = estimate_regime_parameters(returns)
            push!(persistences, parameters.persistence)
            # Bear is the state a risk system must not get wrong.
            @test parameters.variances[2] ≈ truth[2] rtol = 0.35
            @test parameters.variances[2] == maximum(parameters.variances)
            @test parameters.drifts[1] > parameters.drifts[2]
        end
        @test mean(persistences) ≈ RF_TRUE_PERSISTENCE atol = 0.03
        @test maximum(persistences) < 0.99

        @test_throws ArgumentError estimate_regime_parameters([0.01])
        @test_throws ArgumentError estimate_regime_parameters([0.01, NaN, 0.02])
    end

    @testset "a driftless market is not read as a regime market" begin
        # The null. Confidence must stay near the floor rather than inventing structure.
        returns, _ = rf_returns(;
            n_bars = 6_000, seed = 5,
            process = GaussianReturns(annual_drift = 0.0, annual_volatility = 0.25),
        )
        filter = RegimeFilter(estimate_regime_parameters(returns))
        confidences = Float64[]
        for value in returns
            observe_return!(filter, value)
            push!(confidences, regime_confidence(filter))
        end
        @test mean(confidences) < 0.35
        @test estimate_regime_parameters(returns).dispersion < 0.5
    end

    @testset "shifting every return shifts every state drift" begin
        returns, _ = rf_returns(; n_bars = 2_000)
        shift = 0.01
        plain = estimate_regime_parameters(returns)
        shifted = estimate_regime_parameters(returns .+ shift)
        @test shifted.centre ≈ plain.centre + shift
        @test shifted.drifts ≈ plain.drifts .+ shift
        @test shifted.persistence ≈ plain.persistence
        @test shifted.variances ≈ plain.variances rtol = 1.0e-9
    end

    @testset "a bar it cannot read is marginalised, not invented" begin
        # Substituting a zero return would be evidence for the quiet state. Marginalising is
        # the exact answer: propagate, correct nothing.
        returns, _ = rf_returns(; n_bars = 2_000)
        parameters = estimate_regime_parameters(returns)

        skipped = RegimeFilter(parameters)
        fit_filter!(skipped, returns[1:500])
        before = regime_probabilities(skipped)
        likelihood = skipped.log_likelihood
        skip_observation!(skipped)

        @test regime_probabilities(skipped) ≈ propagate(before, transition_matrix(parameters))
        @test skipped.log_likelihood == likelihood
        @test n_skipped(skipped) == 1
        @test n_absorbed(skipped) == 500

        # Enough skipped bars and the belief returns to the long-run distribution.
        for _ in 1:500
            skip_observation!(skipped)
        end
        @test regime_probabilities(skipped) ≈ parameters.prior.stationary atol = 1.0e-6

        zeroed = RegimeFilter(parameters)
        fit_filter!(zeroed, returns[1:500])
        observe_return!(zeroed, 0.0)
        @test regime_probabilities(zeroed) != regime_probabilities(skipped)
    end

    @testset "the horizon decays the belief toward the long run" begin
        filter, _, _ = rf_fitted(; n_bars = 3_000)
        stationary = filter.parameters.prior.stationary
        near = horizon_weights(filter; horizon_bars = 1)
        far = horizon_weights(filter; horizon_bars = 500)

        @test sum(near) ≈ 1.0
        @test sum(far) ≈ 1.0
        @test far ≈ stationary atol = 1.0e-6
        # Exactly one step of the chain, and exactly the matrix power at any horizon.
        @test near ≈ propagate(regime_probabilities(filter), transition_matrix(filter.parameters))
        for horizon in (1, 4, 30)
            powered = transition_matrix(filter.parameters)^horizon
            @test horizon_weights(filter; horizon_bars = horizon) ≈
                vec(regime_probabilities(filter)' * powered) atol = 1.0e-12
        end
        @test_throws ArgumentError horizon_weights(filter; horizon_bars = 0)
    end

    @testset "the variance splits three ways, exactly" begin
        filter, _, _ = rf_fitted(; n_bars = 3_000)
        for horizon in (1, 5, 20)
            split = variance_decomposition(filter; horizon_bars = horizon)
            total = var(predict_return(filter; horizon_bars = horizon))
            @test split.aleatoric + split.state + split.parameter ≈ total rtol = 1.0e-12
            @test split.aleatoric > 0
            @test split.state >= 0
            @test split.parameter > 0
        end

        # Two corners where the state term must be exactly zero, not merely small: a belief
        # that is certain, and a chain that forgets immediately.
        certain = RegimeFilter(filter.parameters)
        certain.belief = [1.0, 0.0, 0.0]
        @test variance_decomposition(certain).state == 0.0

        forgetful = RegimeFilter(
            RegimeParameters(
                rf_prior(); centre = 0.0, drift_spread = 0.001, persistence = 0.0,
                mean_variance = 1.0e-4, dispersion = 0.5, n_rows = 1_000,
            ),
        )
        forgetful.belief = [0.6, 0.3, 0.1]
        # Not exactly zero, but zero to the precision the arithmetic allows: with no
        # persistence every conditional forecast is the same number and the differences are
        # rounding.
        @test variance_decomposition(forgetful).state < 1.0e-60

        _, epistemic = predict(filter)
        @test epistemic <= var(predict_return(filter))
        @test epistemic ≈ variance_decomposition(filter).state +
            variance_decomposition(filter).parameter
    end

    @testset "the predictive is over a number the market prints" begin
        filter, returns, _ = rf_fitted(; n_bars = 3_000)
        predictive = predict_return(filter)

        @test predictive isa UnivariateDistribution
        @test isfinite(mean(predictive))
        @test isfinite(var(predictive))
        @test isfinite(logpdf(predictive, returns[end]))
        @test 0 < cdf(predictive, returns[end]) < 1
        @test isfinite(quantile(predictive, 0.99))

        # Non-centred, so the mean moves with the belief, but not by enough to make the
        # directional half of a calibration report informative. That is pinned elsewhere.
        certain_bull = RegimeFilter(filter.parameters)
        certain_bull.belief = [1.0, 0.0, 0.0]
        certain_bear = RegimeFilter(filter.parameters)
        certain_bear.belief = [0.0, 1.0, 0.0]
        @test mean(predict_return(certain_bull)) > mean(predict_return(certain_bear))
        @test var(predict_return(certain_bear)) > var(predict_return(certain_bull))
        @test std(predict_return(filter; horizon_bars = 4)) > std(predictive)
    end

    @testset "it forecasts better than an iid alternative" begin
        # Out of sample, one step ahead, against a Student-t fitted on the warm-up window.
        returns, _ = rf_returns(; n_bars = 6_000, seed = 3)
        warmup = 1_000
        filter = RegimeFilter(estimate_regime_parameters(returns[1:warmup]))
        fit_filter!(filter, returns[1:warmup])

        predictives = []
        outcomes = Float64[]
        for index in (warmup + 1):length(returns)
            push!(predictives, predict_return(filter))
            push!(outcomes, returns[index])
            observe_return!(filter, returns[index])
        end

        report = assess(predictives, outcomes)
        centre = mean(returns[1:warmup])
        spread = std(returns[1:warmup])
        flat = [student_t(centre, spread * sqrt(3 / 5), 5.0) for _ in outcomes]
        baseline = assess(flat, outcomes)

        @test report.mean_log_score > baseline.mean_log_score + 0.02
        @test interval_calibration_error(report) < 0.05
        @test report.pit_ks_statistic < 0.05

        # The directional half carries nothing, and says so in CI so a future reader cannot
        # mistake a Brier score of a quarter for a finding.
        @test report.brier_score ≈ 0.25 atol = 0.01
    end

    @testset "it says which regime it thinks this is" begin
        filter, returns, series = rf_fitted(; n_bars = 6_000, seed = 11)
        belief = regime_belief(filter)
        @test belief isa LabelledCategorical
        @test belief.labels == collect(REGIME_STATES)
        @test sum(belief.probabilities) ≈ 1.0
        @test most_likely_regime(filter) in REGIME_STATES
        @test 0 <= regime_confidence(filter) <= 1
        @test regime_confidence(filter) ≈ 1 - normalised_entropy(belief)

        # Identification against the recorded truth, using the belief held BEFORE each bar
        # is absorbed, so nothing here is scored on data it has already seen.
        truth = true_states(series)
        fresh = RegimeFilter(estimate_regime_parameters(returns[1:3_000]))
        called = Int[]
        actual = Int[]
        for index in 1:length(returns)
            if index > 3_000
                push!(called, argmax(regime_probabilities(fresh)))
                push!(actual, truth[index])
            end
            observe_return!(fresh, returns[index])
        end
        majority = maximum([mean(actual .== state) for state in 1:N_REGIMES])
        @test mean(called .== actual) > majority + 0.1
    end

    @testset "state" begin
        filter, returns, _ = rf_fitted(; n_bars = 2_000)
        restored = RegimeFilter(filter.parameters)
        load_state!(restored, state(filter))
        @test regime_probabilities(restored) == regime_probabilities(filter)
        @test restored.log_likelihood == filter.log_likelihood
        @test n_absorbed(restored) == n_absorbed(filter)

        saved = state(filter)
        saved["belief"][1] = -999.0
        @test regime_probabilities(restored) != saved["belief"]

        for broken in (
                Dict{String, Any}(state(filter)..., "belief" => [0.5, 0.5]),
                Dict{String, Any}(state(filter)..., "belief" => [0.5, 0.6, 0.2]),
                Dict{String, Any}(state(filter)..., "belief" => [-0.1, 0.6, 0.5]),
                Dict{String, Any}(state(filter)..., "n_seen" => -1),
                Dict{String, Any}(state(filter)..., "log_likelihood" => NaN),
            )
            before = regime_probabilities(restored)
            @test_throws ArgumentError load_state!(restored, broken)
            @test regime_probabilities(restored) == before
        end

        reset!(filter)
        @test regime_probabilities(filter) == filter.parameters.prior.stationary
        @test filter.log_likelihood == 0.0
        @test n_absorbed(filter) == 0
        @test n_skipped(filter) == 0
    end

    @testset "parameters name everything that moves a prediction" begin
        filter, _, _ = rf_fitted(; n_bars = 2_000)
        recorded = parameters(filter)
        @test recorded["persistence"] ≈ filter.parameters.persistence
        @test recorded["belief"] ≈ regime_probabilities(filter)
        @test length(recorded["variances"]) == N_REGIMES
        @test stable_hash(recorded) != stable_hash(parameters(RegimeFilter(filter.parameters)))
        @test occursin("RegimeFilter", sprint(show, filter))
    end

    @testset "type stability" begin
        filter, _, _ = rf_fitted(; n_bars = 500)
        @test @inferred(observe_return!(filter, 0.001)) === filter
        @test @inferred(skip_observation!(filter)) === filter
        @test @inferred(regime_confidence(filter)) isa Float64
        @test @inferred(horizon_weights(filter)) isa Vector{Float64}
        @inferred variance_decomposition(filter)
        @inferred predict_return(filter)
        @inferred predict(filter)
        @test_throws ArgumentError observe_return!(filter, NaN)
    end
end
