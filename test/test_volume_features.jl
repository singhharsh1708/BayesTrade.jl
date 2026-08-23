volbar(index, volume; close = 100.0) = Bar(
    "SYNTH", DateTime(2026, 1, 5, 10, 0) + Day(index),
    close, close, close, close, volume,
)

volumewindow(volumes; closes = fill(100.0, length(volumes))) = BarWindow(
    Bar[volbar(i - 1, volumes[i]; close = closes[i]) for i in eachindex(volumes)],
)

generatedwindow(n = 500) = BarWindow(
    generate_series(
        GaussianReturns(); symbol = "SYNTH", n_bars = n, seed = 31,
        start = Date(2022, 1, 3), shape = BarShape(base_volume = 1.0e6),
    ).bars,
)

@testset "volume and liquidity features" begin
    @testset "relative volume" begin
        @testset "a typical day reads zero and a busy one reads positive" begin
            @test evaluate(RelativeVolume(11), volumewindow(fill(1000.0, 11))) ≈ 0.0
            @test evaluate(
                RelativeVolume(11), volumewindow(vcat(fill(1000.0, 10), 3000.0)),
            ) ≈ log(3)
        end

        @testset "twice and half are equal and opposite" begin
            busy = evaluate(RelativeVolume(11), volumewindow(vcat(fill(1000.0, 10), 2000.0)))
            quiet = evaluate(RelativeVolume(11), volumewindow(vcat(fill(1000.0, 10), 500.0)))
            @test busy ≈ -quiet
        end

        @testset "the median resists a single spike" begin
            # One results-day spike must not make ordinary days afterwards look quiet.
            volumes = vcat(fill(1000.0, 19), 500_000.0, fill(1000.0, 5))
            @test evaluate(RelativeVolume(25), volumewindow(volumes)) ≈ 0.0
        end

        @testset "a bar with no volume reports nothing" begin
            @test evaluate(
                RelativeVolume(11), volumewindow(vcat(fill(1000.0, 10), 0.0)),
            ) === nothing
        end
    end

    @testset "volume z-score" begin
        @testset "a flat volume history reports nothing" begin
            @test evaluate(VolumeZScore(20), volumewindow(fill(1000.0, 25))) === nothing
        end

        @testset "a spike scores high" begin
            rng = Xoshiro(2)
            volumes = 1000.0 .* exp.(0.3 .* randn(rng, 60))
            volumes[end] = 20_000.0
            @test evaluate(VolumeZScore(60), volumewindow(volumes)) > 2
        end

        @testset "it standardises log volume" begin
            rng = Xoshiro(5)
            volumes = 1000.0 .* exp.(0.4 .* randn(rng, 40))
            logs = log.(volumes)
            @test evaluate(VolumeZScore(40), volumewindow(volumes)) ≈
                (logs[end] - mean(logs)) / std(logs)
        end

        @testset "a zero volume bar in the window reports nothing" begin
            volumes = vcat(fill(1000.0, 9), 0.0, fill(1000.0, 5))
            @test evaluate(VolumeZScore(10), volumewindow(volumes)) === nothing
        end
    end

    @testset "median turnover" begin
        @testset "it reports rupees, not a ratio" begin
            @test evaluate(
                MedianTurnover(20), volumewindow(fill(1000.0, 25); closes = fill(500.0, 25)),
            ) ≈ 500_000.0
        end

        @testset "it is the median of traded value" begin
            window = generatedwindow()
            expected = median(turnover(bar) for bar in window.bars[(end - 19):end])
            @test evaluate(MedianTurnover(20), window) ≈ expected
        end

        @testset "the price level changes it, which is the point" begin
            cheap = evaluate(
                MedianTurnover(20), volumewindow(fill(1000.0, 25); closes = fill(10.0, 25)),
            )
            dear = evaluate(
                MedianTurnover(20),
                volumewindow(fill(1000.0, 25); closes = fill(10_000.0, 25)),
            )
            @test dear ≈ cheap * 1000
        end
    end

    @testset "amihud illiquidity" begin
        @testset "a market that moves on no volume is illiquid" begin
            closes = [100.0 * (index % 2 == 0 ? 1.03 : 0.97) for index in 1:30]
            thin = evaluate(AmihudIlliquidity(20), volumewindow(fill(100.0, 30); closes = closes))
            thick = evaluate(
                AmihudIlliquidity(20), volumewindow(fill(1.0e7, 30); closes = closes),
            )
            @test thin > thick
        end

        @testset "it matches its definition" begin
            window = generatedwindow()
            n = length(window)
            expected = mean(
                abs(window.log_returns[index - 1]) / turnover(window.bars[index])
                    for index in (n - 19):n
            ) * 1.0e6
            @test evaluate(AmihudIlliquidity(20), window) ≈ expected
        end

        @testset "halted days are skipped, not treated as infinite" begin
            volumes = vcat(fill(1000.0, 15), fill(0.0, 5), fill(1000.0, 10))
            closes = [100.0 * exp(0.001 * index) for index in 1:30]
            value = evaluate(AmihudIlliquidity(20), volumewindow(volumes; closes = closes))
            @test value !== nothing
            @test isfinite(value)
        end

        @testset "a window with no trading at all reports nothing" begin
            @test evaluate(AmihudIlliquidity(20), volumewindow(fill(0.0, 30))) === nothing
        end

        @testset "it is never negative" begin
            window = generatedwindow()
            for stop in 50:37:length(window)
                value = evaluate(AmihudIlliquidity(20), BarWindow(window.bars[1:stop]))
                @test value >= 0
            end
        end
    end

    @testset "degenerate windows are rejected at construction" begin
        @test_throws ArgumentError RelativeVolume(1)
        @test_throws ArgumentError VolumeZScore(2)
        @test_throws ArgumentError MedianTurnover(1)
        @test_throws ArgumentError AmihudIlliquidity(1)
    end
end
