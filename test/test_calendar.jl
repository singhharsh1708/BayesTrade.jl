@testset "trading calendar" begin
    @testset "weekends are skipped" begin
        @test trading_days(Date(2026, 1, 2), 3) ==
            [Date(2026, 1, 2), Date(2026, 1, 5), Date(2026, 1, 6)]
    end

    @testset "a weekend start rolls forward" begin
        @test first(trading_days(Date(2026, 1, 3), 1)) == Date(2026, 1, 5)
    end

    @testset "days are strictly increasing weekdays" begin
        days = trading_days(Date(2026, 1, 1), 60)
        @test length(days) == 60
        @test issorted(days) && allunique(days)
        @test all(is_trading_day, days)
    end

    @testset "degenerate counts" begin
        @test isempty(trading_days(Date(2026, 1, 1), 0))
        @test_throws ArgumentError trading_days(Date(2026, 1, 1), -1)
    end

    @testset "a daily bar is stamped at the session close" begin
        @test session_close(Date(2026, 1, 2)) == DateTime(2026, 1, 2, 10, 0)
    end

    @testset "the annualisation convention is fixed" begin
        @test BARS_PER_YEAR == 252
        @test ANNUALISER ≈ sqrt(252)
        @test annualise(deannualise(0.3)) ≈ 0.3
        @test deannualise(0.252) ≈ 0.252 / sqrt(252)
    end
end
