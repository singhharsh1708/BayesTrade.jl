# Each process is checked against the parameters it was constructed with. Sample sizes are
# large and seeds fixed, so the tolerances are deterministic statements about a specific
# path rather than probabilistic hopes.

const LONG = 200_000
rng() = Xoshiro(11)

@testset "return processes" begin
    @testset "gaussian returns" begin
        @testset "the path recovers its own moments" begin
            process = GaussianReturns(annual_drift = 0.12, annual_volatility = 0.3)
            path = simulate(process, LONG, rng())
            expected_mu = 0.12 / BARS_PER_YEAR
            expected_sigma = deannualise(0.3)
            standard_error = expected_sigma / sqrt(LONG)
            @test mean(path.log_returns) ≈ expected_mu atol = 3 * standard_error
            @test std(path.log_returns) ≈ expected_sigma rtol = 0.01
        end

        @testset "returns are serially independent" begin
            path = simulate(GaussianReturns(), LONG, rng())
            @test abs(cor(path.log_returns[1:(end - 1)], path.log_returns[2:end])) < 0.01
        end

        @testset "conditional volatility is constant and no state is recorded" begin
            path = simulate(GaussianReturns(annual_volatility = 0.25), 500, rng())
            @test allequal(path.volatility)
            @test path.states === nothing
            @test length(path) == 500
        end

        @testset "a non-positive volatility is rejected" begin
            @test_throws ArgumentError GaussianReturns(annual_volatility = 0.0)
        end

        @testset "parameters are reported in both units" begin
            parameters = process_parameters(GaussianReturns(annual_volatility = 0.252))
            @test parameters["process"] == "gaussian"
            @test parameters["bar_volatility"] ≈ 0.252 / sqrt(252)
        end
    end

    @testset "autoregressive returns" begin
        @testset "the sample autocorrelation recovers phi" begin
            for phi in (0.3, -0.25)
                path = simulate(
                    AR1Returns(phi = phi, annual_drift = 0.0), LONG, rng(),
                )
                estimate = cor(path.log_returns[1:(end - 1)], path.log_returns[2:end])
                @test estimate ≈ phi atol = 0.01
            end
        end

        @testset "stationary volatility matches the requested level" begin
            process = AR1Returns(phi = 0.5, annual_volatility = 0.3)
            path = simulate(process, LONG, rng())
            @test std(path.log_returns) ≈ deannualise(0.3) rtol = 0.02
        end

        @testset "the innovation scale shrinks as persistence rises" begin
            mild = AR1Returns(phi = 0.1, annual_volatility = 0.25)
            strong = AR1Returns(phi = 0.8, annual_volatility = 0.25)
            @test innovation_scale(strong) < innovation_scale(mild)
        end

        @testset "a unit root is rejected" begin
            @test_throws ArgumentError AR1Returns(phi = 1.0)
            @test_throws ArgumentError AR1Returns(phi = -1.0)
        end
    end

    @testset "regime switching returns" begin
        process = RegimeSwitchingReturns()

        @testset "empirical transitions recover the matrix" begin
            path = simulate(process, LONG, rng())
            states = path.states
            @test states !== nothing
            counts = zeros(3, 3)
            for (current, following) in zip(states[1:(end - 1)], states[2:end])
                counts[current, following] += 1
            end
            empirical = counts ./ sum(counts, dims = 2)
            @test all(abs.(empirical .- process.transitions) .< 0.01)
        end

        @testset "occupancy matches the stationary distribution" begin
            path = simulate(process, LONG, rng())
            states = path.states
            occupancy = [count(==(k), states) / length(states) for k in 1:3]
            @test all(abs.(occupancy .- stationary_distribution(process)) .< 0.02)
        end

        @testset "the stationary distribution is a probability vector" begin
            distribution = stationary_distribution(process)
            @test sum(distribution) ≈ 1.0
            @test all(>(0), distribution)
            @test n_regimes(process) == 3
        end

        @testset "the bear regime is the most volatile and the only negative one" begin
            path = simulate(process, LONG, rng())
            states = path.states
            means = [mean(path.log_returns[states .== k]) for k in 1:3]
            volatilities = [std(path.log_returns[states .== k]) for k in 1:3]
            @test means[2] < 0 < means[1]
            @test volatilities[2] == maximum(volatilities)
        end

        @testset "conditional volatility tracks the latent state" begin
            path = simulate(process, 1_000, rng())
            expected = deannualise.(process.annual_volatilities)
            @test path.volatility ≈ expected[path.states]
        end

        @testset "an inconsistent specification is rejected" begin
            @test_throws ArgumentError RegimeSwitchingReturns(
                labels = ["bull", "bear"],
                annual_drifts = [0.1, -0.1],
                annual_volatilities = [0.2, 0.4],
                transitions = [0.9 0.2; 0.1 0.9],
            )
            @test_throws ArgumentError RegimeSwitchingReturns(
                labels = ["bull", "bear"],
                annual_drifts = [0.1, -0.1],
                annual_volatilities = [0.2],
                transitions = [0.9 0.1; 0.1 0.9],
            )
        end
    end

    @testset "stochastic volatility returns" begin
        @testset "log volatility persistence is recovered" begin
            process = StochasticVolatilityReturns(persistence = 0.92)
            path = simulate(process, LONG, rng())
            logs = log.(path.volatility)
            @test cor(logs[1:(end - 1)], logs[2:end]) ≈ 0.92 atol = 0.01
        end

        @testset "median volatility matches the requested level" begin
            process = StochasticVolatilityReturns(
                annual_volatility = 0.3, persistence = 0.9,
            )
            path = simulate(process, LONG, rng())
            @test median(path.volatility) ≈ deannualise(0.3) rtol = 0.03
        end

        @testset "absolute returns cluster" begin
            path = simulate(
                StochasticVolatilityReturns(persistence = 0.95), LONG, rng(),
            )
            absolute = abs.(path.log_returns)
            @test cor(absolute[1:(end - 1)], absolute[2:end]) > 0.15
        end

        @testset "unconditional returns are fatter tailed than normal" begin
            path = simulate(
                StochasticVolatilityReturns(persistence = 0.95), LONG, rng(),
            )
            standardised = path.log_returns ./ std(path.log_returns)
            @test mean(standardised .^ 4) > 3.5
        end

        @testset "conditioning on the true volatility restores normality" begin
            # The fat tails come from the mixture, not from a fat-tailed innovation, which
            # is the property a volatility model is supposed to exploit.
            path = simulate(
                StochasticVolatilityReturns(persistence = 0.95, annual_drift = 0.0),
                LONG, rng(),
            )
            standardised = path.log_returns ./ path.volatility
            @test mean(standardised .^ 4) ≈ 3.0 atol = 0.1
            @test std(standardised) ≈ 1.0 atol = 0.01
        end

        @testset "an explosive volatility process is rejected" begin
            @test_throws ArgumentError StochasticVolatilityReturns(persistence = 1.0)
            @test_throws ArgumentError StochasticVolatilityReturns(
                volatility_of_volatility = 0.0,
            )
        end
    end

    @testset "a mismatched path cannot be constructed" begin
        @test_throws ArgumentError ProcessPath([1.0, 2.0], [1.0])
        @test_throws ArgumentError ProcessPath([1.0, 2.0], [1.0, 1.0], [1])
    end

    @testset "an incomplete process says what it is missing" begin
        struct BareProcess <: ReturnProcess end
        @test_throws ArgumentError simulate(BareProcess(), 10, rng())
        @test_throws ArgumentError process_parameters(BareProcess())
    end
end
