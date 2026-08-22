const NOW = DateTime(2026, 1, 2, 15, 30)

bar(close = 100.0; kwargs...) = Bar(
    get(kwargs, :symbol, "RELIANCE"),
    get(kwargs, :timestamp, NOW),
    get(kwargs, :open, close),
    get(kwargs, :high, close),
    get(kwargs, :low, close),
    close,
    get(kwargs, :volume, 1000.0),
)

@testset "market observations" begin
    @testset "instrument" begin
        @test key(Instrument("TCS"; exchange = BSE)) == "BSE:TCS"
        instrument = Instrument("RELIANCE"; tick_size = 0.05)
        @test round_to_tick(instrument, 1234.567) ≈ 1234.55
        @test round_to_tick(instrument, 1234.58) ≈ 1234.6
        @test_throws ArgumentError Instrument("RELIANCE"; lot_size = 0)
        @test_throws ArgumentError Instrument("")
    end

    @testset "bar validates its own shape" begin
        @test_throws ArgumentError Bar("R", NOW, 100.0, 99.0, 101.0, 100.0, 1.0)
        @test_throws ArgumentError Bar("R", NOW, 95.0, 101.0, 99.0, 100.0, 1.0)
        @test_throws ArgumentError Bar("R", NOW, 100.0, 101.0, 99.0, 105.0, 1.0)
        @test_throws ArgumentError Bar("R", NOW, 100.0, 101.0, 99.0, 100.0, -1.0)
        @test_throws ArgumentError Bar("R", NOW, 0.0, 101.0, 99.0, 100.0, 1.0)
        @test Bar("R", NOW, 100.0, 101.0, 99.0, 100.0, 0.0).volume == 0.0
    end

    @testset "derived quantities" begin
        b = Bar("R", NOW, 100.0, 102.0, 98.0, 101.0, 500.0)
        @test typical_price(b) ≈ (102 + 98 + 101) / 3
        @test turnover(b) ≈ typical_price(b) * 500
        @test true_range(b) ≈ 4.0
    end

    @testset "log returns compose additively" begin
        first, second, third = bar(100.0), bar(110.0), bar(121.0)
        step = log_return(first, second) + log_return(second, third)
        @test step ≈ log_return(first, third)
        @test step ≈ log(1.21)
    end

    @testset "quote" begin
        @test_throws ArgumentError Quote("R", NOW, 100.0; bid = 101.0, ask = 100.0)
        q = Quote("R", NOW, 100.0; bid = 99.9, ask = 100.1)
        @test mid(q) ≈ 100.0
        @test spread(q) ≈ 0.2
        @test spread_bps(q) ≈ 20.0
        bare = Quote("R", NOW, 100.0)
        @test mid(bare) === nothing
        @test spread(bare) === nothing
        @test spread_bps(bare) === nothing
    end

    @testset "results cannot be reported before the period ends" begin
        @test_throws ArgumentError FundamentalSnapshot(
            symbol = "R",
            period_end = Date(2026, 3, 31),
            reported_at = DateTime(2026, 3, 1),
        )
        @test_throws ArgumentError FundamentalSnapshot(
            symbol = "R",
            period_end = Date(2026, 3, 31),
            reported_at = DateTime(2026, 4, 25),
            debt_to_equity = -0.5,
        )
    end

    @testset "knowability follows the reporting date" begin
        snapshot = FundamentalSnapshot(
            symbol = "R",
            period_end = Date(2026, 3, 31),
            reported_at = DateTime(2026, 4, 25, 18),
            roe = 0.18,
        )
        @test !is_known_at(snapshot, DateTime(2026, 4, 1))
        @test !is_known_at(snapshot, DateTime(2026, 4, 25, 17, 59))
        @test is_known_at(snapshot, DateTime(2026, 4, 25, 18))
    end

    @testset "news is invisible before publication" begin
        item = NewsItem(
            symbol = "R", published_at = NOW, headline = "Results beat estimates",
            sentiment = POSITIVE, sentiment_confidence = 0.8,
        )
        @test is_known_at(item, NOW)
        @test !is_known_at(item, NOW - Second(1))
        @test_throws ArgumentError NewsItem(
            symbol = "R", published_at = NOW, headline = "x", sentiment_confidence = 0.8,
        )
        @test_throws ArgumentError NewsItem(symbol = "R", published_at = NOW, headline = "")
    end

    @testset "a bar is knowable at its own close" begin
        b = bar()
        @test is_known_at(b, NOW)
        @test !is_known_at(b, NOW - Second(1))
    end
end
