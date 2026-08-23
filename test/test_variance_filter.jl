# The conjugate variance filter, checked against oracles rather than against itself.
#
# Where a claim has an analytic answer it is asserted to floating-point precision. Where it
# does not, it is checked against Monte Carlo from the generative model, or against a
# simulated market whose true volatility path is recorded.

const VF_BAR = 0.02
const VF_PRIOR = 0.3

vf_prior(scale = VF_PRIOR; shape = 2.0) =
    variance_prior(volatility_scale = deannualise(scale), shape = shape)

vf_filter(; scale = VF_PRIOR, shape = 2.0, kwargs...) =
    DiscountedVarianceFilter(vf_prior(scale; shape = shape); kwargs...)

vf_returns(n = 4_000; seed = 1, volatility = VF_BAR) =
    volatility .* randn(Xoshiro(seed), n)

function vf_stochastic(n = 6_000; seed = 5, persistence = 0.97, vol_of_vol = 0.15)
    series = generate_series(
        StochasticVolatilityReturns(
            annual_drift = 0.0, annual_volatility = VF_PRIOR,
            persistence = persistence, volatility_of_volatility = vol_of_vol,
        );
        symbol = "SYNTH", n_bars = n, seed = seed, start = Date(2010, 1, 1),
    )
    return true_log_returns(series), true_volatility(series)
end

"""
Mean one-step log score: every bar scored under the state that existed before it.
"""
function vf_log_score(filter::DiscountedVarianceFilter, returns::AbstractVector{Float64})
    reset!(filter)
    total = 0.0
    for value in returns
        total += logpdf(return_predictive(filter), value)
        update!(filter, value)
    end
    return total / length(returns)
end

@testset "variance filter" begin
    @testset "prior" begin
        @test expected_noise_variance(InverseGammaPrior(2.0, 4.0)) ≈ 4.0
        @test expected_noise_variance(vf_prior(0.3)) ≈ deannualise(0.3)^2
        @test variance_prior(volatility_scale = 0.01, shape = 5.0).rate ≈ 1.0e-4 * 4

        @test_throws ArgumentError InverseGammaPrior(1.0, 1.0)
        @test_throws ArgumentError InverseGammaPrior(0.5, 1.0)
        @test_throws ArgumentError InverseGammaPrior(2.0, 0.0)
        @test_throws ArgumentError variance_prior(volatility_scale = 0.0)
        @test_throws ArgumentError variance_prior(volatility_scale = 0.1, shape = 1.0)
    end

    @testset "construction" begin
        filter = vf_filter()
        @test n_components(filter) == length(DEFAULT_DISCOUNTS)
        @test discount_grid(filter) == collect(DEFAULT_DISCOUNTS)
        @test sum(discount_weights(filter)) ≈ 1.0
        @test allequal(discount_weights(filter))
        @test discount_entropy(filter) ≈ log(n_components(filter))
        @test n_absorbed(filter) == 0
        @test n_skipped(filter) == 0

        @test_throws ArgumentError vf_filter(discounts = Float64[])
        @test_throws ArgumentError vf_filter(discounts = (0.9, 1.1))
        @test_throws ArgumentError vf_filter(discounts = (0.0, 0.9))
        @test_throws ArgumentError vf_filter(discounts = (0.95, 0.9))
        @test_throws ArgumentError vf_filter(discounts = (0.9, 0.9))
        @test_throws ArgumentError vf_filter(weight_forgetting = 0.0)
        @test_throws ArgumentError vf_filter(weight_forgetting = 1.5)
        @test_throws ArgumentError vf_filter(centre = NaN)
        @test_throws ArgumentError observe_variance!(vf_filter(), -1.0)
        @test_throws ArgumentError observe_variance!(vf_filter(), NaN)
        @test_throws ArgumentError observe_variance!(vf_filter(), 1.0; weight = 0.0)
        @test_throws ArgumentError return_predictive(vf_filter(); horizon_bars = 0)
        @test_throws ArgumentError predict_realised_variance(vf_filter(); horizon_bars = 0)
        @test_throws ArgumentError plugin_variance(vf_filter(); horizon_bars = 0)
        @test_throws ArgumentError variance_inflation(vf_filter(); horizon_bars = 0)
        @test_throws ArgumentError fit!(vf_filter(), [0.01, NaN])
        @test_throws ArgumentError fit!(vf_filter(), [0.01, 0.02], [1.0])
    end

    @testset "exactness" begin
        @testset "the return predictive is the marginal, not an approximation" begin
            # Draws from the generative model itself: sigma^2 from the posterior, then a
            # normal at that scale. A scale built as sqrt(b/(a-1)) instead of sqrt(b/a) is
            # the classic slip and no mean-only check would catch it.
            rng = Xoshiro(20260824)
            for (shape, rate, horizon) in ((7.3, 0.0021, 5), (2.5, 1.0e-4, 1))
                draws = Float64[]
                for _ in 1:2_000_000
                    variance = rand(rng, InverseGamma(shape, rate))
                    push!(draws, sqrt(horizon * variance) * randn(rng))
                end
                sort!(draws)
                predictive = student_t(0.0, sqrt(horizon * rate / shape), 2 * shape)
                for level in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
                    cut = quantile(predictive, level)
                    empirical = searchsortedlast(draws, cut) / length(draws)
                    @test empirical ≈ level atol = 0.004
                end
            end
        end

        @testset "realised variance follows the beta prime law" begin
            rng = Xoshiro(20260824)
            shape, rate, horizon = 7.3, 0.0021, 5
            draws = Float64[]
            for _ in 1:2_000_000
                variance = rand(rng, InverseGamma(shape, rate))
                push!(draws, variance * rand(rng, Chisq(horizon)))
            end
            sort!(draws)
            law = 2 * rate * BetaPrime(horizon / 2, shape)
            for level in (0.1, 0.5, 0.9, 0.99)
                cut = quantile(law, level)
                @test searchsortedlast(draws, cut) / length(draws) ≈ level atol = 0.004
            end
            @test cdf(law, 0.0) == 0.0
        end

        @testset "two independently derived families agree on the same quantity" begin
            # E[realised variance over h bars] must equal var(return over h bars). Nothing
            # in the source ties these together, and neither scale nor degrees of freedom
            # can be wrong on its own without breaking it.
            filter = vf_filter()
            fit!(filter, vf_returns())
            for horizon in (1, 2, 5, 20)
                @test var(return_predictive(filter; horizon_bars = horizon)) ≈
                    mean(predict_realised_variance(filter; horizon_bars = horizon)) rtol = 1.0e-12
            end
        end

        @testset "the shape follows its closed form" begin
            # Fails if the discount is applied to the prior as well as the data, if the
            # increment is 1 rather than 1/2, or if the decay lands after absorbing.
            for (discount, n) in ((0.9, 10), (0.94, 500), (0.97, 50), (1.0, 500))
                filter = vf_filter(discounts = (discount,))
                fit!(filter, vf_returns(n))
                expected = filter.prior.shape + 0.5 * (1 - discount^n) / (1 - discount)
                discount >= 1 && (expected = filter.prior.shape + n / 2)
                @test posterior_shape(filter, 1) ≈ expected atol = 1.0e-12
            end
        end

        @testset "forgetting bounds the effective sample size" begin
            for discount in (0.9, 0.94, 0.97)
                filter = vf_filter(discounts = (discount,))
                fit!(filter, vf_returns(20_000))
                @test effective_sample_size(filter) ≈ 1 / (1 - discount) rtol = 1.0e-6
                @test posterior_shape(filter, 1) ≈
                    steady_state_shape(filter.prior.shape, discount) rtol = 1.0e-6
            end
            @test steady_state_shape(2.0, 1.0) == Inf
        end

        @testset "a batch fit is a sequence of updates" begin
            returns = vf_returns(2_000)
            batch = vf_filter()
            fit!(batch, returns)
            stepwise = vf_filter()
            for value in returns
                update!(stepwise, value)
            end
            @test batch.weights ≈ stepwise.weights rtol = 1.0e-12
            @test batch.squares ≈ stepwise.squares rtol = 1.0e-12
            @test batch.log_weights ≈ stepwise.log_weights rtol = 1.0e-12
            @test n_absorbed(batch) == n_absorbed(stepwise) == length(returns)
        end

        @testset "order matters when forgetting, and only then" begin
            trending = Float64[
                0.005 * (1 + 3 * index / 1_000) * randn(Xoshiro(index)) for index in 1:1_000
            ]
            for discount in (0.97, 1.0)
                forward = vf_filter(discounts = (discount,))
                reversed = vf_filter(discounts = (discount,))
                fit!(forward, trending)
                fit!(reversed, reverse(trending))
                ratio = posterior_rate(forward, 1) / posterior_rate(reversed, 1)
                if discount < 1
                    @test abs(ratio - 1) > 0.01
                else
                    @test ratio ≈ 1.0 rtol = 1.0e-12
                end
            end
        end

        @testset "the epistemic split is an identity on the mixture" begin
            filter = vf_filter()
            fit!(filter, vf_returns())
            for horizon in (1, 4, 10)
                total = var(return_predictive(filter; horizon_bars = horizon))
                @test plugin_variance(filter; horizon_bars = horizon) +
                    variance_inflation(filter; horizon_bars = horizon) ≈ total rtol = 1.0e-12
            end
            for index in 1:n_components(filter)
                shape = evolved_shape(filter, index)
                plugin = evolved_rate(filter, index) / shape
                inflation = evolved_rate(filter, index) / (shape * (shape - 1))
                @test inflation / (plugin + inflation) ≈ 1 / shape rtol = 1.0e-12
            end
        end

        @testset "the dropped evidence terms really are the same for every component" begin
            # The reduced form is exact, not an approximation, so the difference between
            # two components must match the difference computed from the full return-form
            # log density.
            filter = vf_filter()
            fit!(filter, vf_returns(500))
            observation = 0.013
            estimate = abs2(observation)
            reduced = Float64[
                BayesTrade.log_variance_evidence(
                        evolved_shape(filter, index), evolved_rate(filter, index), estimate, 1.0,
                    ) for index in 1:n_components(filter)
            ]
            full = Float64[
                logpdf(
                        student_t(
                            0.0, sqrt(evolved_rate(filter, index) / evolved_shape(filter, index)),
                            2 * evolved_shape(filter, index),
                        ), observation,
                    ) for index in 1:n_components(filter)
            ]
            for index in 2:n_components(filter)
                @test reduced[index] - reduced[1] ≈ full[index] - full[1] atol = 1.0e-12
            end
        end

        @testset "scoring the squared return equals scoring the return" begin
            # The reduced variance-form evidence and the Student-t log density of the
            # return itself differ by a single constant, and by nothing that depends on
            # the observation. That constant is what the change of variable leaves behind
            # once the dropped terms are accounted for, so this pins both at once.
            for (shape, rate, error) in
                ((3.7, 4.2e-4, 0.013), (50.0, 0.02, -0.004), (1.2, 1.0e-6, 9.0e-4))
                variance_form = BayesTrade.log_variance_evidence(
                    shape, rate, abs2(error), 1.0,
                )
                return_form = logpdf(student_t(0.0, sqrt(rate / shape), 2 * shape), error)
                @test return_form - variance_form ≈ -0.5 * log(2 * pi) atol = 1.0e-10
            end
        end

        @testset "the horizon adds noise but no evidence" begin
            filter = vf_filter()
            fit!(filter, vf_returns())
            one_bar = return_predictive(filter; horizon_bars = 1)
            four_bar = return_predictive(filter; horizon_bars = 4)
            @test std(four_bar) / std(one_bar) ≈ 2.0 rtol = 1.0e-12
            for index in 1:n_components(filter)
                @test dof(one_bar.components[index].ρ) === dof(four_bar.components[index].ρ)
            end
        end
    end

    @testset "degenerate bars" begin
        @testset "a flat bar is evidence, not a special case" begin
            filter = vf_filter()
            fit!(filter, vf_returns(200))
            for _ in 1:500
                update!(filter, 0.0)
            end
            @test all(isfinite, filter.log_weights)
            @test sum(discount_weights(filter)) ≈ 1.0
            for index in 1:n_components(filter)
                @test posterior_rate(filter, index) >= filter.prior.rate
            end
            predictive = return_predictive(filter)
            @test isfinite(mean(predictive))
            @test isfinite(var(predictive))
            @test isfinite(logpdf(predictive, 0.0))
            @test isfinite(quantile(predictive, 0.99))
            @test 0 < cdf(predictive, 0.0) < 1
        end

        @testset "an unabsorbable bar is refused before it corrupts anything" begin
            # Both arguments can pass their own finiteness check while the step still
            # overflows, either in the product or in the accumulated sum of squares
            # several bars later. Either way every component's evidence goes to -Inf, the
            # softmax to NaN, and the grid posterior never recovers: the filter would go
            # on accepting bars and reporting NaN, and the first visible error would come
            # from inside the mixture constructor calls later.
            filter = vf_filter()
            fit!(filter, vf_returns(200))
            weights = copy(filter.weights)
            squares = copy(filter.squares)
            log_weights = copy(filter.log_weights)

            @test_throws ArgumentError observe_variance!(filter, 1.0e308; weight = 10.0)
            @test filter.weights == weights
            @test filter.squares == squares
            @test filter.log_weights == log_weights
            @test n_absorbed(filter) == 200

            # Reached by the ordinary single-return path, with no weight involved: one such
            # bar is representable, the next is not.
            creeping = vf_filter()
            update!(creeping, 1.0e154)
            @test all(isfinite, creeping.log_weights)
            @test_throws ArgumentError update!(creeping, 1.0e154)
            @test all(isfinite, creeping.log_weights)
            @test n_absorbed(creeping) == 1

            # And it survives to keep working on ordinary bars afterwards.
            for value in vf_returns(100; seed = 71)
                update!(creeping, value)
            end
            @test all(isfinite, creeping.log_weights)
            @test isfinite(noise_variance(creeping))
        end

        @testset "the rate floor is unreachable rather than defensive" begin
            filter = vf_filter()
            for _ in 1:500
                update!(filter, 0.0)
            end
            for index in 1:n_components(filter)
                @test posterior_rate(filter, index) == filter.prior.rate
                @test posterior_shape(filter, index) > filter.prior.shape
            end
        end

        @testset "a skipped bar ages the state and teaches it nothing" begin
            filter = vf_filter()
            fit!(filter, vf_returns(400; volatility = 2 * VF_BAR))
            weights = copy(filter.weights)
            squares = copy(filter.squares)
            trail = Float64[residual_scale(filter)]
            for _ in 1:1_000
                skip_observation!(filter)
                push!(trail, residual_scale(filter))
            end

            @test issorted(trail; rev = true)
            @test n_skipped(filter) == 1_000
            @test n_absorbed(filter) == 400
            for index in 1:n_components(filter)
                decay = filter.discounts[index]^1_000
                @test filter.weights[index] ≈ decay * weights[index] rtol = 1.0e-9
                @test filter.squares[index] ≈ decay * squares[index] rtol = 1.0e-9
            end
        end

        @testset "a filter that forgets returns all the way to the prior" begin
            # The component that never forgets is excluded on purpose: it is supposed to
            # hold its estimate through a thousand missing bars, and the test asserts that
            # rather than pretending the whole grid reverts.
            filter = vf_filter(discounts = (0.9, 0.95))
            fit!(filter, vf_returns(400; volatility = 2 * VF_BAR))
            for _ in 1:1_000
                skip_observation!(filter)
            end
            for index in 1:n_components(filter)
                @test posterior_shape(filter, index) ≈ filter.prior.shape atol = 1.0e-8
                @test posterior_rate(filter, index) ≈ filter.prior.rate atol = 1.0e-8
            end
            @test residual_scale(filter) ≈
                sqrt(expected_noise_variance(filter.prior)) atol = 1.0e-8

            patient = vf_filter(discounts = (1.0,))
            fit!(patient, vf_returns(400; volatility = 2 * VF_BAR))
            held = residual_scale(patient)
            for _ in 1:1_000
                skip_observation!(patient)
            end
            @test residual_scale(patient) == held
        end

        @testset "absorbing a flat feed is not the same as skipping it" begin
            # A halted feed routed through the absorb path errs narrow, which is the one
            # direction a risk system must not err. This is that failure, priced in CI.
            absorbed = vf_filter()
            skipped = vf_filter()
            returns = vf_returns(400; volatility = 2 * VF_BAR)
            fit!(absorbed, returns)
            fit!(skipped, returns)
            for _ in 1:500
                update!(absorbed, 0.0)
                skip_observation!(skipped)
            end
            @test annualise(residual_scale(absorbed)) < 0.1
            @test annualise(residual_scale(skipped)) > 0.35
        end
    end

    @testset "the discount posterior" begin
        @testset "it works out which market it is in, unaided" begin
            steady = vf_filter()
            fit!(steady, vf_returns(4_000; seed = 3))
            moving = vf_filter()
            fit!(moving, first(vf_stochastic(4_000; seed = 3)))

            @test expected_discount(steady) > expected_discount(moving) + 0.015
            @test expected_discount(moving) < 0.95
            @test argmax(discount_weights(moving)) == 1
        end

        @testset "not knowing the discount costs almost nothing" begin
            # The whole justification for carrying a grid: near the best single choice when
            # it matters, and far above the worst.
            returns = convert(Vector{Float64}, first(vf_stochastic(6_000; seed = 7)))
            mixture = vf_log_score(vf_filter(), returns)
            singles = Float64[
                vf_log_score(vf_filter(discounts = (discount,)), returns)
                    for discount in DEFAULT_DISCOUNTS
            ]
            @test maximum(singles) - mixture < 0.03
            @test mixture - minimum(singles) > 0.15
        end

        @testset "and it costs almost nothing when there is nothing to adapt to" begin
            returns = vf_returns(6_000; seed = 9)
            mixture = vf_log_score(vf_filter(), returns)
            singles = Float64[
                vf_log_score(vf_filter(discounts = (discount,)), returns)
                    for discount in DEFAULT_DISCOUNTS
            ]
            @test maximum(singles) - mixture < 0.03
        end

        @testset "a component beaten for a thousand bars can still come back" begin
            # Weights live in log space precisely so this is possible. In linear space one
            # underflows to zero, alpha * log(0) is -Inf, and the model stops being
            # adaptive with no error and no symptom.
            # A market that keeps changing is what crushes the component that never
            # forgets, so the crush is built from alternating regimes rather than from one
            # loud stretch, which it would simply learn.
            filter = vf_filter(discounts = (0.9, 1.0))
            rng = Xoshiro(3)
            for block in 1:20, _ in 1:100
                update!(filter, (isodd(block) ? 0.01 : 0.04) * randn(rng))
            end
            crushed = last(discount_weights(filter))
            @test crushed < 0.01

            settled = sqrt((0.01^2 + 0.04^2) / 2)
            for _ in 1:1_500
                update!(filter, settled * randn(rng))
            end
            @test all(isfinite, filter.log_weights)
            @test last(discount_weights(filter)) > 100 * crushed
        end

        @testset "static averaging is a supported configuration" begin
            filter = vf_filter(weight_forgetting = 1.0)
            fit!(filter, first(vf_stochastic(2_000)))
            @test sum(discount_weights(filter)) ≈ 1.0
            @test all(isfinite, filter.log_weights)
        end
    end

    @testset "tracking a market" begin
        @testset "it recovers a known volatility, time-averaged" begin
            # Deliberately not asserted at the endpoint: the shortest grid member carries
            # an effective sample of ten bars, so a single final reading is a 45 per cent
            # relative standard deviation in variance and would flake.
            filter = vf_filter()
            returns = vf_returns(4_000; seed = 11)
            trail = Float64[]
            for (index, value) in enumerate(returns)
                update!(filter, value)
                index > 500 && push!(trail, residual_scale(filter))
            end
            @test annualise(mean(trail)) ≈ annualise(VF_BAR) rtol = 0.1
        end

        @testset "it tracks a true path it never sees" begin
            returns, truth = vf_stochastic(6_000; seed = 13)
            filter = vf_filter()
            before = Float64[]
            after = Float64[]
            for value in returns
                push!(before, residual_scale(filter))
                update!(filter, value)
                push!(after, residual_scale(filter))
            end
            logs = log.(convert(Vector{Float64}, truth))
            @test cor(log.(before), logs) > 0.6
            # A one-bar lookahead would show up here as a suspiciously good score, so the
            # honest number is pinned below the peeking one rather than left unbounded.
            @test cor(log.(after), logs) > cor(log.(before), logs) + 0.02
        end

        @testset "a regime break moves it, but not instantly" begin
            quiet = vf_returns(1_200; seed = 17, volatility = deannualise(0.1))
            loud = vf_returns(300; seed = 19, volatility = deannualise(0.6))
            filter = vf_filter(scale = 0.1)
            before_weight = 0.0
            disagreement_before = Float64[]
            for value in quiet
                update!(filter, value)
                push!(disagreement_before, discount_disagreement(filter))
            end
            @test annualise(residual_scale(filter)) < 0.3
            before_weight = last(discount_weights(filter))

            crossed = 0
            disagreement_after = Float64[]
            for (index, value) in enumerate(loud)
                update!(filter, value)
                push!(disagreement_after, discount_disagreement(filter))
                if crossed == 0 && annualise(residual_scale(filter)) > 0.3
                    crossed = index
                end
            end
            @test crossed > 3
            @test crossed <= 30
            @test before_weight > 0.1
            @test last(discount_weights(filter)) < 0.05
            @test maximum(disagreement_after[1:100]) >
                5 * mean(disagreement_before[(end - 99):end])
        end
    end

    @testset "invariants that never bend" begin
        @testset "the moments are finite whatever the market does" begin
            # calibration.jl calls mean() unguarded. This is what makes that safe, and it
            # is a property of the parameterisation rather than of a clamp.
            filter = vf_filter(discounts = (0.5, 0.9, 1.0))
            rng = Xoshiro(23)
            for index in 1:2_000
                value = if index % 500 == 0
                    20 * VF_BAR
                elseif index % 97 == 0
                    0.0
                else
                    exp(-13 + 5 * rand(rng)) * randn(rng)
                end
                update!(filter, value)
                for component in 1:n_components(filter)
                    @test posterior_shape(filter, component) >= filter.prior.shape
                    @test posterior_rate(filter, component) >= filter.prior.rate
                    @test 2 * evolved_shape(filter, component) > 2
                end
                predictive = return_predictive(filter)
                @test isfinite(mean(predictive))
                @test isfinite(var(predictive))
            end
        end

        @testset "expected volatility sits below the root of expected variance" begin
            # Jensen, and not a technicality: a sizer that squares an expected volatility
            # to get a variance systematically under-reserves.
            filter = vf_filter()
            for value in vf_returns(500; seed = 29)
                update!(filter, value)
                @test expected_volatility(filter) < residual_scale(filter)
            end
            # Checked against draws from the posterior rather than against the source
            # formula re-derived inline, which could only ever restate it.
            rng = Xoshiro(43)
            draws = sqrt.(rand(rng, variance_posterior(filter), 400_000))
            @test mean(draws) ≈ expected_volatility(filter) rtol = 0.01
            @test mean(log.(draws)) ≈ expected_log_volatility(filter) rtol = 0.01
            @test std(log.(draws)) ≈ volatility_uncertainty(filter) rtol = 0.05
            @test sqrt(mean(abs2, draws)) ≈ residual_scale(filter) rtol = 0.01
        end

        @testset "expected volatility survives an enormous shape" begin
            # The two forms must agree where both are accurate, and the answer must stay
            # right where the direct one stops being: at a shape of 1e17 it returns
            # exactly 1 for a quantity near 3e-9.
            for shape in (2.0, 10.0, 1.0e3, 1.0e6, 1.0e7)
                @test BayesTrade.gamma_half_ratio(shape) ≈
                    exp(loggamma(shape - 0.5) - loggamma(shape)) rtol = 1.0e-6
            end
            for shape in (1.0e9, 1.0e13, 1.0e17)
                @test BayesTrade.gamma_half_ratio(shape) ≈ 1 / sqrt(shape) rtol = 1.0e-8
            end

            filter = vf_filter(scale = 0.3, shape = 1.0e12)
            fit!(filter, vf_returns(500; seed = 73))
            @test isfinite(expected_volatility(filter))
            @test expected_volatility(filter) < residual_scale(filter)
            @test expected_volatility(filter) ≈ residual_scale(filter) rtol = 1.0e-6
        end

        @testset "the prior keeps a bounded, computable share" begin
            filter = vf_filter()
            fit!(filter, vf_returns(2_000; seed = 31))
            expected = 0.0
            for index in 1:n_components(filter)
                weight = discount_weights(filter)[index]
                shape, rate = filter.prior.shape, filter.prior.rate
                expected += weight * (rate + filter.squares[index] / 2) /
                    ((shape - 1) + filter.weights[index] / 2)
            end
            @test noise_variance(filter) ≈ expected atol = 1.0e-12

            vague = vf_filter(scale = 0.1)
            strong = vf_filter(scale = 0.9)
            returns = vf_returns(5_000; seed = 37)
            fit!(vague, returns)
            fit!(strong, returns)
            @test residual_scale(strong) / residual_scale(vague) < 1.15
        end

        @testset "uncertainty is reported on the log scale and stays positive" begin
            filter = vf_filter()
            @test volatility_uncertainty(filter) > 0
            fit!(filter, vf_returns(2_000))
            @test volatility_uncertainty(filter) > 0
            @test discount_disagreement(filter) >= 0
            @test 0 <= discount_entropy(filter) <= log(n_components(filter)) + 1.0e-12
            @test expected_log_volatility(filter) < log(expected_volatility(filter))
        end

        @testset "the volatility interval is the root of the variance interval" begin
            filter = vf_filter()
            fit!(filter, vf_returns(1_000))
            variance = credible_interval(variance_posterior(filter))
            volatility = volatility_interval(filter)
            @test volatility.lower ≈ sqrt(variance.lower)
            @test volatility.upper ≈ sqrt(variance.upper)
            @test expected_volatility(filter) in volatility
        end
    end

    @testset "every knob does something" begin
        # Each of these was reachable by a mutation that the suite did not notice: the
        # centring could be deleted, the weight discount ignored, and the reweighting
        # dropped from a skipped bar, with every other assertion still passing.
        @testset "the centre is subtracted before squaring" begin
            drifting = fill(0.05, 400)
            centred = vf_filter(scale = 1.0, centre = 0.05)
            uncentred = vf_filter(scale = 1.0)
            fit!(centred, drifting)
            fit!(uncentred, drifting)

            # The centred filter sees a series with no variation at all, so what is left
            # is the prior; the uncentred one sees a constant five per cent move every bar.
            @test residual_scale(centred) < residual_scale(uncentred) / 5
            @test residual_scale(uncentred) > 0.04
            @test mean(return_predictive(centred)) ≈ 0.05
            @test mean(return_predictive(uncentred)) == 0.0
            @test mean(return_predictive(centred; horizon_bars = 4)) ≈ 0.2
        end

        @testset "the weight discount bounds how sure the grid may become" begin
            # This is what alpha is for, and it is not adaptation speed: both settings
            # eventually reach the right component. Below one, the evidence a bar
            # contributes decays, so the posterior cannot accumulate unbounded confidence
            # in a memory length from a stretch of market that has already passed.
            quiet = vf_returns(1_500; seed = 47, volatility = deannualise(0.12))
            adaptive = vf_filter(scale = 0.12, weight_forgetting = 0.98)
            static = vf_filter(scale = 0.12, weight_forgetting = 1.0)
            fit!(adaptive, quiet)
            fit!(static, quiet)

            @test discount_entropy(adaptive) > discount_entropy(static) + 0.5
            @test maximum(discount_weights(static)) > 0.9
            @test maximum(discount_weights(adaptive)) < 0.6
        end

        @testset "a skipped bar ages the grid posterior too" begin
            filter = vf_filter()
            fit!(filter, first(vf_stochastic(1_500; seed = 51)))
            before = discount_entropy(filter)
            for _ in 1:400
                skip_observation!(filter)
            end
            # No evidence arrives, so the discount of the weights pulls the posterior back
            # toward uniform. A skip that left the weights alone would leave the filter
            # certain about a memory length it has had no evidence for in 400 bars.
            @test discount_entropy(filter) > before
            @test discount_entropy(filter) ≈ log(n_components(filter)) atol = 0.05
        end

        @testset "the floor binds where the recursion cannot save it" begin
            # Below one, the weight recursion is contractive and the floor never binds. At
            # exactly one it is not, so this is the configuration where the floor is what
            # keeps a beaten component alive.
            filter = vf_filter(discounts = (0.9, 1.0), weight_forgetting = 1.0)
            rng = Xoshiro(3)
            lowest = 0.0
            for block in 1:60, _ in 1:100
                update!(filter, (isodd(block) ? 0.002 : 0.08) * randn(rng))
                lowest = min(lowest, minimum(filter.log_weights))
            end
            @test lowest ≈ BayesTrade.MIN_LOG_WEIGHT atol = 1.0e-9
            @test all(isfinite, filter.log_weights)
            @test minimum(filter.log_weights) > BayesTrade.MIN_LOG_WEIGHT + 1.0
        end

        @testset "predict reports the epistemic share, not the other one" begin
            filter = vf_filter()
            fit!(filter, vf_returns(600; seed = 53))
            distribution, epistemic = predict(filter; horizon_bars = 3)
            @test epistemic ≈ variance_inflation(filter; horizon_bars = 3)
            @test epistemic < plugin_variance(filter; horizon_bars = 3)
            @test epistemic + plugin_variance(filter; horizon_bars = 3) ≈
                var(distribution) rtol = 1.0e-12
        end
    end

    @testset "now and next are different questions" begin
        # The forecasting paths read the aged state and the volatility posterior reads the
        # filtered one. Each of those could be swapped for the other without any other
        # assertion noticing, so they are pinned here.
        filter = vf_filter(discounts = (0.9,))
        fit!(filter, vf_returns(800; seed = 59))

        @test evolved_shape(filter, 1) < posterior_shape(filter, 1)
        @test evolved_rate(filter, 1) < posterior_rate(filter, 1)

        predictive = return_predictive(filter)
        @test std(predictive)^2 ≈
            evolved_rate(filter, 1) / (evolved_shape(filter, 1) - 1) rtol = 1.0e-12
        @test mean(variance_posterior(filter)) ≈
            posterior_rate(filter, 1) / (posterior_shape(filter, 1) - 1) rtol = 1.0e-12
        @test mean(predict_realised_variance(filter)) ≈
            evolved_rate(filter, 1) / (evolved_shape(filter, 1) - 1) rtol = 1.0e-12

        @testset "the grid is scored on a forecast, not on a fit" begin
            graded = vf_filter()
            fit!(graded, vf_returns(300; seed = 61))
            expected = Float64[
                graded.weight_forgetting * graded.log_weights[index] +
                    BayesTrade.log_variance_evidence(
                        evolved_shape(graded, index), evolved_rate(graded, index), 4.0e-4, 1.0,
                    ) for index in 1:n_components(graded)
            ]
            expected .-= log(sum(exp, expected))
            update!(graded, 0.02)
            @test graded.log_weights ≈ expected rtol = 1.0e-12
        end
    end

    @testset "the horizon rule holds up where it is used" begin
        # The square-root rule assumes the variance holds over the horizon, which is what
        # this filter exists to say is false, so the multi-bar predictive is an
        # approximation and is scored here rather than asserted to be one. Measured across
        # seeds, the ten-bar predictive is not reliably worse calibrated than the one-bar:
        # summing ten bars averages over the very fluctuations that the rule ignores. What
        # is pinned is that both are calibrated, on outcomes the filter never saw.
        returns = convert(Vector{Float64}, first(vf_stochastic(6_000; seed = 67)))
        horizon = 10
        filter = vf_filter()
        near, near_outcomes = [], Float64[]
        far, far_outcomes = [], Float64[]
        for index in 1:(length(returns) - horizon)
            if index > 500
                push!(near, return_predictive(filter; horizon_bars = 1))
                push!(near_outcomes, returns[index])
                push!(far, return_predictive(filter; horizon_bars = horizon))
                push!(far_outcomes, sum(view(returns, index:(index + horizon - 1))))
            end
            update!(filter, returns[index])
        end
        near_report = assess(near, near_outcomes)
        far_report = assess(far, far_outcomes)
        @test interval_calibration_error(near_report) < 0.05
        @test interval_calibration_error(far_report) < 0.05
        @test near_report.pit_ks_statistic < 0.05
        @test far_report.pit_ks_statistic < 0.05
    end

    @testset "state" begin
        @testset "reset returns it to the prior exactly" begin
            filter = vf_filter()
            fit!(filter, vf_returns(1_000))
            skip_observation!(filter)
            reset!(filter)
            for index in 1:n_components(filter)
                @test posterior_shape(filter, index) === filter.prior.shape
                @test posterior_rate(filter, index) === filter.prior.rate
            end
            @test allequal(filter.log_weights)
            @test effective_sample_size(filter) == 0.0
            @test n_absorbed(filter) == 0
            @test n_skipped(filter) == 0
        end

        @testset "a restored filter is the same filter" begin
            original = vf_filter()
            fit!(original, vf_returns(1_000))
            restored = vf_filter()
            load_state!(restored, state(original))

            @test restored.weights == original.weights
            @test restored.squares == original.squares
            @test restored.log_weights == original.log_weights
            @test n_absorbed(restored) == n_absorbed(original)

            for value in vf_returns(100; seed = 41)
                update!(original, value)
                update!(restored, value)
            end
            @test restored.squares == original.squares
            @test restored.log_weights == original.log_weights
        end

        @testset "a restored filter does not share memory with what restored it" begin
            # `convert` is a no-op on a vector that already has the right type, so this is
            # the difference between loading a state and adopting the caller's arrays.
            original = vf_filter()
            fit!(original, vf_returns(500))
            saved = state(original)
            restored = vf_filter()
            load_state!(restored, saved)

            saved["weights"][1] = -999.0
            saved["squares"][1] = -999.0
            saved["log_weights"][1] = -999.0
            @test restored.weights[1] == original.weights[1]
            @test restored.squares[1] == original.squares[1]
            @test restored.log_weights[1] == original.log_weights[1]

            handed_out = state(restored)
            restored.weights[1] = -1.0
            @test handed_out["weights"][1] != -1.0
        end

        @testset "a bad state is refused and changes nothing" begin
            filter = vf_filter()
            fit!(filter, vf_returns(500))
            saved = state(filter)
            before = copy(filter.weights)

            for broken in (
                    Dict{String, Any}(saved..., "weights" => [1.0, 2.0]),
                    Dict{String, Any}(saved..., "squares" => fill(NaN, n_components(filter))),
                    Dict{String, Any}(
                        saved..., "log_weights" => zeros(n_components(filter)),
                    ),
                    Dict{String, Any}(saved..., "n_seen" => -1),
                    Dict{String, Any}(
                        saved..., "weights" => fill(-1.0, n_components(filter)),
                    ),
                )
                @test_throws ArgumentError load_state!(filter, broken)
                @test filter.weights == before
            end
            @test_throws KeyError load_state!(filter, Dict{String, Any}())
        end

        @testset "parameters name everything that moves a prediction" begin
            filter = vf_filter()
            fit!(filter, vf_returns(500))
            recorded = parameters(filter)
            @test recorded["discounts"] == collect(DEFAULT_DISCOUNTS)
            @test sum(recorded["discount_weights"]) ≈ 1.0
            @test recorded["expected_volatility"] ≈ expected_volatility(filter)
            @test length(recorded["shapes"]) == n_components(filter)
            @test stable_hash(recorded) != stable_hash(parameters(vf_filter()))
            @test occursin("DiscountedVarianceFilter", sprint(show, filter))
        end
    end

    @testset "type stability" begin
        filter = vf_filter()
        fit!(filter, vf_returns(200))
        @test @inferred(update!(filter, 0.01)) === filter
        @test @inferred(observe_variance!(filter, 1.0e-4; weight = 2.0)) === filter
        @test @inferred(skip_observation!(filter)) === filter
        @test @inferred(BayesTrade.log_variance_evidence(3.0, 1.0e-4, 1.0e-4, 1.0)) isa Float64
        @test @inferred(noise_variance(filter)) isa Float64
        @test @inferred(expected_discount(filter)) isa Float64
        @test @inferred(volatility_uncertainty(filter)) isa Float64
        @inferred return_predictive(filter)
        @inferred variance_posterior(filter)
        @inferred predict_realised_variance(filter)
        @inferred predict(filter)

        # probability_above dispatches on UnivariateDistribution, so a predictive that only
        # quacks like one would fail inside assess rather than here.
        @test return_predictive(filter) isa UnivariateDistribution
        @test variance_posterior(filter) isa UnivariateDistribution
        @test 0 < probability_positive(return_predictive(filter)) < 1
    end
end
