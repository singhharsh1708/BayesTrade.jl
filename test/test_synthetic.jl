series(; n_bars = 500, seed = 7, kwargs...) = generate_series(
    GaussianReturns(annual_drift = 0.1, annual_volatility = 0.25);
    symbol = "SYNTH", n_bars = n_bars, seed = seed, kwargs...,
)

@testset "synthetic series" begin
    @testset "determinism" begin
        @testset "the same seed reproduces the series exactly" begin
            @test closes(series()) == closes(series())
            first, second = series(), series()
            @test all(
                getfield(a, f) == getfield(b, f)
                    for (a, b) in zip(first.bars, second.bars)
                    for f in fieldnames(Bar)
            )
        end

        @testset "a different seed produces a different series" begin
            @test closes(series(seed = 7)) != closes(series(seed = 8))
        end

        @testset "the parameters that produced the series are recorded" begin
            s = series(n_bars = 100)
            @test s.seed == 7
            @test s.parameters["process"] == "gaussian"
            @test s.parameters["n_bars"] == 100
            @test s.parameters["annual_volatility"] ≈ 0.25
        end
    end

    @testset "bar validity" begin
        @testset "every bar satisfies the OHLC ordering" begin
            for bar in series(n_bars = 2_000).bars
                @test bar.low <= bar.open <= bar.high
                @test bar.low <= bar.close <= bar.high
                @test bar.volume >= 0
            end
        end

        @testset "prices are reported in paise" begin
            for bar in series(n_bars = 200).bars
                for price in (bar.open, bar.high, bar.low, bar.close)
                    @test round(price, digits = 2) == price
                end
            end
        end

        @testset "a violent process still produces valid bars" begin
            s = generate_series(
                GaussianReturns(annual_drift = 0.0, annual_volatility = 3.0);
                n_bars = 2_000, seed = 3,
                shape = BarShape(intraday_volatility_multiple = 4.0, gap_fraction = 1.0),
            )
            @test length(s) == 2_000
        end

        @testset "timestamps are increasing weekday session closes" begin
            s = series(n_bars = 60, start = Date(2026, 1, 1))
            stamps = [bar.timestamp for bar in s.bars]
            @test stamps[1] == DateTime(2026, 1, 1, 10, 0)
            @test issorted(stamps) && allunique(stamps)
            @test all(stamp -> dayofweek(stamp) <= 5, stamps)
        end

        @testset "degenerate arguments are rejected" begin
            @test_throws ArgumentError series(n_bars = 0)
            @test_throws ArgumentError generate_series(
                GaussianReturns(); initial_price = 0.0,
            )
        end
    end

    @testset "recoverability" begin
        @testset "returns recovered from closes match the latent path" begin
            # Elementwise: `isapprox` on arrays compares norms, so a per-element bound is
            # what this actually means, and rounding to paise is a per-element effect.
            s = series(n_bars = 5_000)
            @test maximum(abs.(realised_log_returns(s) .- true_log_returns(s)[2:end])) <
                1.0e-4
        end

        @testset "realised volatility recovers the process parameter" begin
            s = generate_series(
                GaussianReturns(annual_drift = 0.0, annual_volatility = 0.28);
                n_bars = 20_000, seed = 4,
            )
            @test annualise(std(realised_log_returns(s))) ≈ 0.28 rtol = 0.03
        end

        @testset "momentum survives the trip through rounded bars" begin
            s = generate_series(
                AR1Returns(phi = 0.35, annual_drift = 0.0); n_bars = 20_000, seed = 4,
            )
            returns = realised_log_returns(s)
            @test cor(returns[1:(end - 1)], returns[2:end]) ≈ 0.35 atol = 0.03
        end

        @testset "the latent regime path is kept" begin
            s = generate_series(RegimeSwitchingReturns(); n_bars = 1_000, seed = 4)
            states = true_states(s)
            @test states !== nothing
            @test length(states) == 1_000
            @test Set(unique(states)) == Set(1:3)
        end

        @testset "an independent process keeps no state path" begin
            @test true_states(series(n_bars = 100)) === nothing
        end
    end

    @testset "intraday extremes" begin
        # Range estimators are derived from the extremes of a Brownian path, so the wicks
        # must be those extremes rather than noise around the body.
        parkinson(s) = annualise(
            sqrt(
                mean(
                    log.([b.high for b in s.bars] ./ [b.low for b in s.bars]) .^ 2,
                ) / (4 * log(2)),
            ),
        )

        @testset "parkinson recovers the generating volatility without gaps" begin
            s = generate_series(
                GaussianReturns(annual_drift = 0.0, annual_volatility = 0.3);
                n_bars = 20_000, seed = 4, shape = BarShape(gap_fraction = 0.0),
            )
            @test parkinson(s) ≈ 0.3 rtol = 0.02
            @test annualise(std(realised_log_returns(s))) ≈ 0.3 rtol = 0.02
        end

        @testset "a range estimator sees only the part the price walked through" begin
            # A gap is a jump, not something the path travelled, so a range estimator reads
            # low on gappy data. That is a property of the world, not an error.
            for gap in (0.25, 0.5)
                s = generate_series(
                    GaussianReturns(annual_drift = 0.0, annual_volatility = 0.3);
                    n_bars = 20_000, seed = 4, shape = BarShape(gap_fraction = gap),
                )
                @test parkinson(s) ≈ 0.3 * (1 - gap) rtol = 0.05
                @test annualise(std(realised_log_returns(s))) ≈ 0.3 rtol = 0.02
            end
        end

        @testset "fewer steps still recover the volatility thanks to the correction" begin
            coarse = generate_series(
                GaussianReturns(annual_drift = 0.0, annual_volatility = 0.3);
                n_bars = 20_000, seed = 4,
                shape = BarShape(gap_fraction = 0.0, intraday_steps = 26),
            )
            fine = generate_series(
                GaussianReturns(annual_drift = 0.0, annual_volatility = 0.3);
                n_bars = 20_000, seed = 4,
                shape = BarShape(gap_fraction = 0.0, intraday_steps = 400),
            )
            @test parkinson(coarse) ≈ parkinson(fine) rtol = 0.03
        end
    end

    @testset "volume" begin
        @testset "volume rises with the size of the move" begin
            s = series(n_bars = 20_000, seed = 9)
            moves = abs.(true_log_returns(s))
            volumes = [bar.volume for bar in s.bars]
            @test cor(moves, log.(volumes)) > 0.3
        end

        @testset "median volume sits near the configured base" begin
            s = series(n_bars = 20_000, seed = 9, shape = BarShape(base_volume = 2.0e6))
            @test median([bar.volume for bar in s.bars]) ≈ 2.0e6 rtol = 0.1
        end

        @testset "the coupling can be switched off" begin
            s = series(
                n_bars = 20_000, seed = 9,
                shape = BarShape(volume_volatility_beta = 0.0),
            )
            moves = abs.(true_log_returns(s))
            volumes = [bar.volume for bar in s.bars]
            @test abs(cor(moves, log.(volumes))) < 0.05
        end
    end

    @testset "point-in-time slicing" begin
        s = series(n_bars = 50, start = Date(2026, 1, 1))
        cutoff = s.bars[20].timestamp
        visible = bars_until(s, cutoff)
        @test length(visible) == 20
        @test last(visible).timestamp == cutoff
        @test all(bar -> bar.timestamp <= cutoff, visible)
        @test isempty(bars_until(s, DateTime(2025, 12, 31)))
        @test length(bars_until(s, DateTime(2027, 1, 1))) == 50
    end

    @testset "bar shape validates its own arguments" begin
        @test_throws ArgumentError BarShape(gap_fraction = 1.5)
        @test_throws ArgumentError BarShape(intraday_steps = 0)
        @test_throws ArgumentError BarShape(base_volume = 0.0)
    end
end
