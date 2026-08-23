# Momentum features, checked against closed forms on constructed series.

@testset "momentum features" begin
    @testset "moving average" begin
        @testset "the simple average is the mean of the window" begin
            @test evaluate(MovingAverage(5), featurewindow(Float64.(1:20))) ≈ 18.0
            @test evaluate(
                MovingAverage(3), featurewindow([1.0, 2.0, 3.0, 100.0, 100.0, 100.0]),
            ) ≈ 100.0
        end

        @testset "the exponential average matches the recursion" begin
            closes = Float64.(1:30)
            feature = MovingAverage(5; exponential = true)
            used = closes[(end - required_bars(feature) + 1):end]
            alpha = 2 / 6
            expected = mean(used[1:5])
            for value in used[6:end]
                expected += alpha * (value - expected)
            end
            @test evaluate(feature, featurewindow(closes)) ≈ expected
        end

        @testset "it needs more history than its window" begin
            feature = MovingAverage(10; exponential = true)
            @test lookback(feature) == 29
            @test evaluate(feature, featurewindow(fill(100.0, 20))) === nothing
            @test evaluate(feature, featurewindow(fill(100.0, 30))) ≈ 100.0
        end

        @testset "the seed has decayed out of the reported value" begin
            # Two very different prehistories converge by the time it is reported.
            tail_values = fill(100.0, 30)
            low = evaluate(
                MovingAverage(10; exponential = true),
                featurewindow(vcat(fill(1.0, 30), tail_values)),
            )
            high = evaluate(
                MovingAverage(10; exponential = true),
                featurewindow(vcat(fill(500.0, 30), tail_values)),
            )
            @test low ≈ high rtol = 1.0e-9
        end

        @testset "a degenerate window is rejected" begin
            @test_throws ArgumentError MovingAverage(1)
        end
    end

    @testset "price to moving average" begin
        @testset "a flat series sits on its average" begin
            @test evaluate(PriceToMovingAverage(5), featurewindow(fill(100.0, 10))) ≈ 0.0
        end

        @testset "trading above the average is positive" begin
            @test evaluate(
                PriceToMovingAverage(5), featurewindow(vcat(fill(100.0, 9), 120.0)),
            ) > 0
        end

        @testset "the log form is symmetric where a percentage is not" begin
            feature = PriceToMovingAverage(50)
            above = evaluate(feature, featurewindow(vcat(fill(100.0, 49), 120.0)))
            below = evaluate(feature, featurewindow(vcat(fill(100.0, 49), 100.0 / 1.2)))
            @test above ≈ -below rtol = 0.01
        end

        @testset "it is unchanged by the price level" begin
            cheap = evaluate(PriceToMovingAverage(5), featurewindow(vcat(fill(10.0, 9), 12.0)))
            dear = evaluate(
                PriceToMovingAverage(5), featurewindow(vcat(fill(10_000.0, 9), 12_000.0)),
            )
            @test cheap ≈ dear
        end
    end

    @testset "moving average spread" begin
        @testset "the sign follows the trend and the magnitude its steepness" begin
            @test evaluate(MovingAverageSpread(5, 20), featurewindow(exponential(120, 0.01))) > 0
            @test evaluate(MovingAverageSpread(5, 20), featurewindow(exponential(120, -0.01))) < 0
            @test evaluate(MovingAverageSpread(5, 20), featurewindow(fill(100.0, 100))) ≈ 0.0
            gentle = evaluate(MovingAverageSpread(5, 20), featurewindow(exponential(120, 0.002)))
            steep = evaluate(MovingAverageSpread(5, 20), featurewindow(exponential(120, 0.02)))
            @test steep > gentle > 0
        end

        @testset "the fast window must be shorter" begin
            @test_throws ArgumentError MovingAverageSpread(26, 12)
        end
    end

    @testset "momentum" begin
        @testset "without a skip it is the window log return" begin
            @test evaluate(Momentum(20, 0), featurewindow(exponential(50, 0.01))) ≈ 0.2
        end

        @testset "the skip excludes the most recent bars" begin
            closes = exponential(50, 0.01)
            closes[end] = closes[end - 1] * 0.5
            @test evaluate(Momentum(20, 5), featurewindow(closes)) ≈ 0.2
            @test evaluate(Momentum(20, 0), featurewindow(closes)) < 0
        end

        @testset "the name records the skip and the warm-up covers both" begin
            @test feature_name(Momentum(60, 5)) === :momentum_60_5
            @test feature_name(Momentum(60, 0)) === :momentum_60
            @test lookback(Momentum(20, 5)) == 25
            @test evaluate(Momentum(20, 5), featurewindow(fill(100.0, 25))) === nothing
            @test evaluate(Momentum(20, 5), featurewindow(fill(100.0, 26))) ≈ 0.0
        end

        @testset "invalid parameters are rejected" begin
            @test_throws ArgumentError Momentum(20, -1)
            @test_throws ArgumentError Momentum(0, 0)
        end
    end

    @testset "relative strength index" begin
        @testset "unbroken moves pin it to its ends" begin
            @test evaluate(RelativeStrengthIndex(14), featurewindow(exponential(30, 0.01))) ≈ 100.0
            @test evaluate(RelativeStrengthIndex(14), featurewindow(exponential(30, -0.01))) ≈ 0.0
        end

        @testset "a flat series is neutral rather than undefined" begin
            @test evaluate(RelativeStrengthIndex(14), featurewindow(fill(100.0, 30))) ≈ 50.0
        end

        @testset "it matches the ratio it is defined as" begin
            closes = [100.0, 102.0, 101.0, 104.0, 103.0, 107.0, 105.0, 110.0]
            changes = diff(closes)[(end - 3):end]
            gains = mean(max.(changes, 0.0))
            losses = mean(max.(-changes, 0.0))
            @test evaluate(RelativeStrengthIndex(4), featurewindow(closes)) ≈
                100 - 100 / (1 + gains / losses)
        end

        @testset "it stays within its scale" begin
            rng = Xoshiro(4)
            closes = 100.0 .* exp.(cumsum(0.02 .* randn(rng, 200)))
            for stop in 20:200
                value = evaluate(RelativeStrengthIndex(14), featurewindow(closes[1:stop]))
                @test 0.0 <= value <= 100.0
            end
        end
    end

    @testset "price z-score" begin
        @testset "a flat window reports nothing rather than dividing by zero" begin
            @test evaluate(PriceZScore(10), featurewindow(fill(100.0, 20))) === nothing
        end

        @testset "a price at its mean scores zero and a spike scores high" begin
            @test evaluate(
                PriceZScore(6), featurewindow([100.0, 110.0, 90.0, 110.0, 90.0, 100.0]),
            ) ≈ 0.0 atol = 0.05
            @test evaluate(PriceZScore(10), featurewindow(vcat(fill(100.0, 19), 130.0))) > 1
        end

        @testset "it matches the standardised log price" begin
            closes = [100.0, 105.0, 98.0, 110.0, 103.0, 112.0]
            logs = log.(closes)
            @test evaluate(PriceZScore(6), featurewindow(closes)) ≈
                (logs[end] - mean(logs)) / std(logs)
        end
    end

    @testset "drawdown from high" begin
        @testset "a new high is zero drawdown and it is never positive" begin
            @test evaluate(DrawdownFromHigh(20), featurewindow(exponential(40, 0.01))) ≈ 0.0
            rng = Xoshiro(7)
            closes = 100.0 .* exp.(cumsum(0.02 .* randn(rng, 200)))
            for stop in 21:200
                @test evaluate(DrawdownFromHigh(20), featurewindow(closes[1:stop])) <= 1.0e-12
            end
        end

        @testset "it measures against the intraday high, not the close" begin
            # A stop is hit intraday, and a close-to-close drawdown does not know that.
            closes = fill(100.0, 10)
            highs = vcat(fill(100.0, 9), 150.0)
            @test evaluate(DrawdownFromHigh(10), featurewindow(closes; highs = highs)) ≈
                log(100 / 150)
            @test evaluate(DrawdownFromHigh(11), featurewindow(vcat(fill(100.0, 10), 80.0))) ≈
                log(0.8)
        end
    end

    @testset "trend" begin
        @testset "a perfect exponential is fully explained" begin
            for rate in (0.004, -0.004)
                closes = exponential(60, rate)
                @test evaluate(TrendSlope(30), featurewindow(closes)) ≈ rate
                @test evaluate(TrendQuality(30), featurewindow(closes)) ≈ 1.0
            end
        end

        @testset "noise lowers the quality without moving the slope" begin
            rng = Xoshiro(3)
            clean = exponential(400, 0.002)
            noisy = clean .* exp.(0.02 .* randn(rng, 400))
            @test evaluate(TrendSlope(200), featurewindow(noisy)) ≈ 0.002 atol = 0.001
            quality = evaluate(TrendQuality(200), featurewindow(noisy))
            @test 0.0 < quality < 0.99
        end

        @testset "a flat series has no trend to explain" begin
            @test evaluate(TrendSlope(20), featurewindow(fill(100.0, 30))) ≈ 0.0
            @test evaluate(TrendQuality(20), featurewindow(fill(100.0, 30))) === nothing
        end

        @testset "quality stays within zero and one" begin
            rng = Xoshiro(11)
            closes = 100.0 .* exp.(cumsum(0.02 .* randn(rng, 300)))
            for stop in 21:7:300
                value = evaluate(TrendQuality(20), featurewindow(closes[1:stop]))
                @test 0.0 <= value <= 1.0
            end
        end

        @testset "too short a window is rejected" begin
            @test_throws ArgumentError TrendSlope(2)
            @test_throws ArgumentError TrendQuality(2)
        end
    end

    @testset "flatness is detected against a relative tolerance" begin
        # The mean of twenty identical logs is not always exactly that log, so an exact
        # check reads a residual around 1e-31 as real variance.
        @test is_flat(log.(fill(100.0, 20)))
        @test !is_flat(log.(exponential(20, 0.001)))
    end
end
