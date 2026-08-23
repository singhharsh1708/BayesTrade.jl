# Calibration machinery, checked against distributions whose calibration is known.
#
# Outcomes are drawn from the very distribution being scored, so a correct implementation must
# report calibration; then the predictives are deliberately narrowed and widened, and the
# report must say which way it went wrong.

const CAL_N = 8_000
const CAL_SPREAD = 0.02

function honest(n = CAL_N; seed = 7, centre = 0.0, scale_factor = 1.0)
    rng = Xoshiro(seed)
    draws = centre .+ CAL_SPREAD .* randn(rng, n)
    predictives = [Normal(centre, CAL_SPREAD * scale_factor) for _ in 1:n]
    return predictives, draws
end

@testset "calibration" begin
    @testset "probability integral transform" begin
        @testset "a correct model produces a uniform transform" begin
            pit = probability_integral_transform(honest()...)
            @test mean(pit) ≈ 0.5 atol = 0.01
            @test std(pit) ≈ 1 / sqrt(12) atol = 0.01
            @test kolmogorov_smirnov_uniform(pit) < 0.02
        end

        @testset "too narrow a predictive pushes the transform to the edges" begin
            pit = probability_integral_transform(honest(; scale_factor = 0.5)...)
            @test count(value -> value < 0.05 || value > 0.95, pit) / length(pit) > 0.2
            @test kolmogorov_smirnov_uniform(pit) > 0.05
        end

        @testset "too wide a predictive bunches it in the middle" begin
            pit = probability_integral_transform(honest(; scale_factor = 2.0)...)
            @test count(value -> 0.3 < value < 0.7, pit) / length(pit) > 0.6
        end

        @testset "a biased predictive shifts the transform" begin
            rng = Xoshiro(3)
            draws = 0.01 .+ CAL_SPREAD .* randn(rng, CAL_N)
            predictives = [Normal(0.0, CAL_SPREAD) for _ in 1:CAL_N]
            @test mean(probability_integral_transform(predictives, draws)) > 0.6
        end

        @testset "a length mismatch is rejected" begin
            predictives, draws = honest(10)
            @test_throws ArgumentError probability_integral_transform(predictives, draws[1:5])
        end
    end

    @testset "kolmogorov smirnov" begin
        @testset "a perfectly uniform sample has a small statistic" begin
            grid = ((1:1_000) .- 0.5) ./ 1_000
            @test kolmogorov_smirnov_uniform(collect(grid)) < 0.001
        end

        @testset "a concentrated sample has a large one" begin
            @test kolmogorov_smirnov_uniform(fill(0.5, 1_000)) > 0.45
            @test kolmogorov_smirnov_uniform(Float64[]) == 0.0
        end
    end

    @testset "coverage" begin
        @testset "a correct model covers at its nominal rate" begin
            pit = probability_integral_transform(honest()...)
            for point in coverage_curve(pit)
                @test point.empirical ≈ point.level atol = 0.02
            end
        end

        @testset "a narrow model covers less than it claims" begin
            pit = probability_integral_transform(honest(; scale_factor = 0.5)...)
            for point in coverage_curve(pit, (0.9, 0.95))
                @test coverage_error(point) < -0.05
            end
        end

        @testset "a wide model covers more than it claims" begin
            pit = probability_integral_transform(honest(; scale_factor = 2.0)...)
            @test coverage_error(first(coverage_curve(pit, (0.5,)))) > 0.15
        end

        @testset "an invalid level is rejected" begin
            @test_throws ArgumentError coverage_curve(zeros(10), (1.0,))
        end
    end

    @testset "reliability" begin
        @testset "a calibrated forecaster sits on the diagonal" begin
            rng = Xoshiro(11)
            probabilities = rand(rng, CAL_N)
            realised = rand(rng, CAL_N) .< probabilities
            curve = reliability_curve(probabilities, realised)
            @test expected_calibration_error(curve) < 0.02
            for bin in curve
                @test bin.observed_frequency ≈ bin.mean_predicted atol = 0.05
            end
        end

        @testset "an overconfident forecaster is caught" begin
            rng = Xoshiro(13)
            truth = 0.3 .+ 0.4 .* rand(rng, CAL_N)
            stated = clamp.((truth .- 0.5) .* 3 .+ 0.5, 0.01, 0.99)
            realised = rand(rng, CAL_N) .< truth
            @test expected_calibration_error(reliability_curve(stated, realised)) > 0.08
        end

        @testset "empty buckets are dropped rather than counted as misses" begin
            curve = reliability_curve(fill(0.5, 100), vcat(trues(50), falses(50)))
            @test length(curve) == 1
            @test first(curve).n == 100
        end

        @testset "the top bucket includes a probability of one" begin
            curve = reliability_curve(fill(1.0, 10), trues(10))
            @test sum(bin.n for bin in curve) == 10
        end

        @testset "a length mismatch is rejected" begin
            @test_throws ArgumentError reliability_curve([0.5, 0.5], [true])
            @test_throws ArgumentError reliability_curve([0.5], [true]; bins = 0)
        end
    end

    @testset "brier score" begin
        @test brier_score([1.0, 0.0, 1.0], [true, false, true]) ≈ 0.0
        @test brier_score(fill(0.5, 4), [true, false, true, false]) ≈ 0.25
        @test brier_score([0.0, 1.0], [false, true]) ≈ 0.0
        @test brier_score([0.0, 1.0], [true, false]) ≈ 1.0
        @test brier_score(Float64[], Bool[]) == 0.0
        @test_throws ArgumentError brier_score([0.5], [true, false])
    end

    @testset "assess" begin
        @testset "a correct model passes every check" begin
            report = assess(honest()...)
            @test report.n == CAL_N
            @test interval_calibration_error(report) < 0.02
            @test report.expected_calibration_error < 0.03
            @test report.pit_ks_statistic < 0.02
            @test abs(report.bias) < 0.001
            @test report.sharpness ≈ CAL_SPREAD rtol = 0.01
        end

        @testset "an overconfident model is named as such" begin
            report = assess(honest(; scale_factor = 0.5)...)
            @test is_overconfident(report)
            @test interval_calibration_error(report) > 0.05
        end

        @testset "an underconfident model is not called overconfident" begin
            report = assess(honest(; scale_factor = 2.0)...)
            @test !is_overconfident(report)
            @test interval_calibration_error(report) > 0.05
        end

        @testset "a biased model is reported as biased" begin
            rng = Xoshiro(5)
            draws = 0.01 .+ CAL_SPREAD .* randn(rng, CAL_N)
            predictives = [Normal(0.0, CAL_SPREAD) for _ in 1:CAL_N]
            @test assess(predictives, draws).bias ≈ 0.01 atol = 0.001
        end

        @testset "sharpness separates a useless calibrated model from a useful one" begin
            # Both are calibrated. Only one says anything.
            rng = Xoshiro(19)
            signal = CAL_SPREAD .* randn(rng, CAL_N)
            noise = CAL_SPREAD .* randn(rng, CAL_N)
            realised = signal .+ noise

            informed = assess([Normal(s, CAL_SPREAD) for s in signal], realised)
            vague = assess(
                [Normal(0.0, sqrt(2) * CAL_SPREAD) for _ in 1:CAL_N], realised,
            )
            @test interval_calibration_error(informed) < 0.02
            @test interval_calibration_error(vague) < 0.02
            @test informed.sharpness < vague.sharpness
            @test informed.mean_log_score > vague.mean_log_score
        end

        @testset "a student-t predictive is scored on its own terms" begin
            rng = Xoshiro(23)
            degrees = 6.0
            draws = CAL_SPREAD .* rand(rng, TDist(degrees), CAL_N)
            predictives = [student_t(0.0, CAL_SPREAD, degrees) for _ in 1:CAL_N]
            @test interval_calibration_error(assess(predictives, draws)) < 0.02
        end

        @testset "scoring t outcomes with a normal looks overconfident in the tails" begin
            rng = Xoshiro(29)
            draws = CAL_SPREAD .* rand(rng, TDist(4.0), CAL_N)
            matched = std(draws)
            report = assess([Normal(0.0, matched) for _ in 1:CAL_N], draws)
            @test coverage_error(first(coverage_curve(report.pit, (0.99,)))) < -0.005
        end

        @testset "the summary reports the numbers that matter" begin
            summary = summarise(assess(honest(2_000)...))
            @test occursin("interval error", summary)
            @test occursin("Brier score", summary)
            @test occursin("sharpness", summary)
        end

        @testset "an empty or mismatched sample is refused" begin
            @test_throws ArgumentError assess(Normal[], Float64[])
            predictives, draws = honest(10)
            @test_throws ArgumentError assess(predictives, draws[1:5])
        end
    end
end
