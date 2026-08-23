# Volatility features, checked against data whose true volatility is known.
#
# Recovery tests run with the opening gap switched off, so the whole of a bar's move is
# travelled by the intraday path and every estimator measures the same quantity. The gap case
# is tested separately, because the disagreement it produces is a real property, not an error.

const TRUE_VOL = 0.3

volseries(volatility = TRUE_VOL; gap = 0.0, n_bars = 4_000, seed = 19) = generate_series(
    GaussianReturns(annual_drift = 0.0, annual_volatility = volatility);
    symbol = "SYNTH", n_bars = n_bars, seed = seed, start = Date(2010, 1, 1),
    shape = BarShape(gap_fraction = gap),
)

volwindow(volatility = TRUE_VOL; kwargs...) = BarWindow(volseries(volatility; kwargs...).bars)

flatvolwindow(n = 60, close = 100.0) = BarWindow(
    Bar[
        Bar("SYNTH", DateTime(2026, 1, 5, 10, 0) + Day(i), close, close, close, close, 1000.0)
            for i in 0:(n - 1)
    ],
)

trendwindow(n = 60, rate = 0.01) = BarWindow(
    Bar[
        (
                c = 100.0 * exp(rate * i);
                Bar("SYNTH", DateTime(2026, 1, 5, 10, 0) + Day(i), c, c, c, c, 1.0)
            ) for i in 0:(n - 1)
    ],
)

@testset "volatility features" begin
    @testset "every estimator recovers the generating volatility" begin
        window = volwindow()
        for feature in (
                RealisedVolatility(2_000), ParkinsonVolatility(2_000),
                GarmanKlassVolatility(2_000), EwmaVolatility(500),
            )
            @test evaluate(feature, window) ≈ TRUE_VOL rtol = 0.05
        end
    end

    @testset "a quieter market reads quieter" begin
        calm = evaluate(RealisedVolatility(2_000), volwindow(0.1))
        wild = evaluate(RealisedVolatility(2_000), volwindow(0.6))
        @test calm < 0.15 < 0.55 < wild
    end

    @testset "range estimators are more efficient than close-to-close" begin
        # The claim that justifies computing them at all, measured rather than asserted.
        bars = volseries().bars
        cc = Float64[]
        pk = Float64[]
        for stop in 100:20:length(bars)
            window = BarWindow(bars[(stop - 59):stop])
            push!(cc, evaluate(RealisedVolatility(20), window))
            push!(pk, evaluate(ParkinsonVolatility(20), window))
        end
        @test std(pk) < 0.75 * std(cc)
    end

    @testset "a range estimator reads low when risk arrives overnight" begin
        gappy = volwindow(TRUE_VOL; gap = 0.5)
        @test evaluate(RealisedVolatility(2_000), gappy) ≈ TRUE_VOL rtol = 0.05
        @test evaluate(ParkinsonVolatility(2_000), gappy) ≈ TRUE_VOL * 0.5 rtol = 0.1
    end

    @testset "realised volatility" begin
        @testset "it matches the sample standard deviation" begin
            window = volwindow()
            returns = window.log_returns[(end - 49):end]
            @test evaluate(RealisedVolatility(50), window) ≈ std(returns) * sqrt(252)
        end

        @testset "a steady climb is not charged as volatility" begin
            # Measured about the mean, so a pure trend contributes nothing.
            @test evaluate(RealisedVolatility(50), trendwindow()) ≈ 0.0 atol = 1.0e-9
            @test evaluate(RealisedVolatility(50), flatvolwindow()) ≈ 0.0
        end

        @testset "annualisation can be switched off" begin
            window = volwindow()
            daily = evaluate(RealisedVolatility(50; annualised = false), window)
            @test evaluate(RealisedVolatility(50), window) ≈ daily * sqrt(252)
        end
    end

    @testset "ewma volatility" begin
        @testset "it matches the recursion it is defined by" begin
            window = volwindow()
            feature = EwmaVolatility(20)
            squared = window.log_returns[(end - lookback(feature) + 1):end] .^ 2
            alpha = 2 / 21
            level = mean(squared[1:20])
            for value in squared[21:end]
                level += alpha * (value - level)
            end
            @test evaluate(feature, window) ≈ sqrt(level) * sqrt(252)
        end

        @testset "it reacts faster than a long flat average" begin
            calm = volseries(0.1; n_bars = 400).bars
            shocked = volseries(0.9; n_bars = 40, seed = 23).bars
            shifted = Bar[
                Bar(
                        bar.symbol, last(calm).timestamp + Day(i), bar.open, bar.high,
                        bar.low, bar.close, bar.volume,
                    ) for (i, bar) in enumerate(shocked)
            ]
            window = BarWindow(vcat(calm, shifted))
            @test evaluate(EwmaVolatility(10), window) >
                evaluate(RealisedVolatility(400), window)
        end

        @testset "a flat market reports nothing rather than zero volatility" begin
            @test evaluate(EwmaVolatility(10), flatvolwindow(120)) === nothing
        end
    end

    @testset "range estimators" begin
        @testset "parkinson matches its closed form" begin
            window = volwindow()
            ranges = log.(window.highs[(end - 49):end] ./ window.lows[(end - 49):end])
            @test evaluate(ParkinsonVolatility(50), window) ≈
                sqrt(mean(ranges .^ 2) / (4 * log(2))) * sqrt(252)
        end

        @testset "garman-klass matches its closed form" begin
            window = volwindow()
            ranges = log.(window.highs[(end - 49):end] ./ window.lows[(end - 49):end])
            bodies = log.(window.closes[(end - 49):end] ./ window.opens[(end - 49):end])
            variance = mean(0.5 .* ranges .^ 2 .- (2 * log(2) - 1) .* bodies .^ 2)
            @test evaluate(GarmanKlassVolatility(50), window) ≈ sqrt(variance) * sqrt(252)
        end

        @testset "both report nothing on bars with no range" begin
            @test evaluate(ParkinsonVolatility(20), flatvolwindow()) === nothing
            @test evaluate(GarmanKlassVolatility(20), flatvolwindow()) === nothing
        end

        @testset "garman-klass stays non-negative when bodies dominate" begin
            # The worst case for the estimator: every bar is pure body, no wick at all.
            bars = Bar[]
            for index in 0:29
                base = 100.0
                close = base * (index % 2 == 0 ? 1.02 : 0.98)
                push!(
                    bars, Bar(
                        "SYNTH", DateTime(2026, 1, 5, 10, 0) + Day(index), base,
                        max(base, close), min(base, close), close, 1.0,
                    ),
                )
            end
            @test evaluate(GarmanKlassVolatility(20), BarWindow(bars)) > 0
        end
    end

    @testset "downside volatility" begin
        @testset "a market that only rises has no downside" begin
            @test evaluate(DownsideVolatility(50), trendwindow()) ≈ 0.0
        end

        @testset "it sits below the symmetric estimate on a symmetric market" begin
            window = volwindow()
            downside = evaluate(DownsideVolatility(2_000), window)
            total = evaluate(RealisedVolatility(2_000), window)
            @test downside ≈ total / sqrt(2) rtol = 0.05
        end
    end

    @testset "volatility ratio" begin
        @testset "a calm stretch after a wild one reads negative" begin
            wild = volseries(0.9; n_bars = 200).bars
            calm = Bar[
                Bar(
                        bar.symbol, last(wild).timestamp + Day(i), bar.open, bar.high,
                        bar.low, bar.close, bar.volume,
                    ) for (i, bar) in enumerate(volseries(0.1; n_bars = 40, seed = 29).bars)
            ]
            @test evaluate(VolatilityRatio(20, 200), BarWindow(vcat(wild, calm))) < -0.5
        end

        @testset "a wild stretch after a calm one reads positive" begin
            calm = volseries(0.1; n_bars = 200).bars
            wild = Bar[
                Bar(
                        bar.symbol, last(calm).timestamp + Day(i), bar.open, bar.high,
                        bar.low, bar.close, bar.volume,
                    ) for (i, bar) in enumerate(volseries(0.9; n_bars = 40, seed = 31).bars)
            ]
            @test evaluate(VolatilityRatio(20, 200), BarWindow(vcat(calm, wild))) > 0.5
        end

        @testset "a steady market sits near zero" begin
            @test abs(evaluate(VolatilityRatio(60, 1_000), volwindow())) < 0.3
        end

        @testset "a flat market reports nothing" begin
            @test evaluate(VolatilityRatio(5, 30), flatvolwindow()) === nothing
        end

        @testset "the fast window must be shorter" begin
            @test_throws ArgumentError VolatilityRatio(60, 5)
        end
    end

    @testset "average true range" begin
        @testset "it is reported as a fraction of price" begin
            value = evaluate(AverageTrueRange(14), volwindow())
            @test 0.0 < value < 0.2
        end

        @testset "it matches the definition of true range" begin
            window = volwindow()
            n = length(window)
            total = 0.0
            for index in (n - 13):n
                previous = window.closes[index - 1]
                total += max(
                    window.highs[index] - window.lows[index],
                    abs(window.highs[index] - previous),
                    abs(window.lows[index] - previous),
                )
            end
            @test evaluate(AverageTrueRange(14), window) ≈ total / 14 / window.closes[end]
        end

        @testset "unlike the range estimators it counts the overnight gap" begin
            # On the same gappy series, true range exceeds the high-low range it contains.
            window = volwindow(TRUE_VOL; gap = 0.6)
            high_low = mean(window.highs[(end - 199):end] .- window.lows[(end - 199):end]) /
                window.closes[end]
            @test evaluate(AverageTrueRange(200), window) > high_low * 1.2
        end

        @testset "a flat market has no range" begin
            @test evaluate(AverageTrueRange(14), flatvolwindow()) ≈ 0.0
        end
    end

    @testset "degenerate windows are rejected at construction" begin
        @test_throws ArgumentError RealisedVolatility(2)
        @test_throws ArgumentError EwmaVolatility(1)
        @test_throws ArgumentError ParkinsonVolatility(1)
        @test_throws ArgumentError GarmanKlassVolatility(1)
        @test_throws ArgumentError DownsideVolatility(2)
        @test_throws ArgumentError AverageTrueRange(1)
    end
end
