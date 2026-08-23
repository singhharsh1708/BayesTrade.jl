@testset "risk limits" begin
    @testset "defaults are internally consistent" begin
        limits = RiskLimits()
        @test limits.max_position_weight <= limits.max_sector_exposure
        @test limits.max_sector_exposure <= limits.max_portfolio_exposure
        @test limits.risk_budget_per_trade <= limits.max_daily_loss
        @test limits.max_daily_loss <= limits.max_drawdown
    end

    @testset "limits are checked against each other, not only against a range" begin
        # Each of these is a plausible typo that would otherwise be discovered by a trade.
        @test_throws ArgumentError RiskLimits(
            max_position_weight = 0.3, max_sector_exposure = 0.25,
        )
        @test_throws ArgumentError RiskLimits(
            max_sector_exposure = 0.7, max_portfolio_exposure = 0.6,
        )
        @test_throws ArgumentError RiskLimits(max_daily_loss = 0.2, max_drawdown = 0.15)
        @test_throws ArgumentError RiskLimits(
            risk_budget_per_trade = 0.05, max_daily_loss = 0.02,
        )
    end

    @testset "a gate at or below a coin flip is not a gate" begin
        @test_throws ArgumentError RiskLimits(min_probability_positive = 0.45)
        @test_throws ArgumentError RiskLimits(min_probability_positive = 1.0)
        @test RiskLimits(min_probability_positive = 0.5) isa RiskLimits
    end

    @testset "fractions must be fractions" begin
        @test_throws ArgumentError RiskLimits(max_position_weight = 0.0)
        @test_throws ArgumentError RiskLimits(max_drawdown = 1.5)
        @test_throws ArgumentError RiskLimits(max_open_positions = 0)
        @test_throws ArgumentError RiskLimits(stop_loss_atr_multiple = 0.0)
    end

    @testset "the binding constraint is reported" begin
        loose = RiskLimits(
            max_position_weight = 0.1, max_sector_exposure = 0.3,
            max_portfolio_exposure = 0.6, max_open_positions = 10,
        )
        @test max_concurrent_position_weight(loose) ≈ 1.0
        @test is_position_cap_binding(loose)

        tight = RiskLimits(
            max_position_weight = 0.02, max_sector_exposure = 0.25,
            max_portfolio_exposure = 0.6, max_open_positions = 5,
        )
        @test max_concurrent_position_weight(tight) ≈ 0.1
        @test !is_position_cap_binding(tight)
    end

    @testset "the conservative preset is tighter everywhere it matters" begin
        default = RiskLimits()
        @test CONSERVATIVE.max_position_weight < default.max_position_weight
        @test CONSERVATIVE.max_portfolio_exposure < default.max_portfolio_exposure
        @test CONSERVATIVE.max_daily_loss < default.max_daily_loss
        @test CONSERVATIVE.max_drawdown < default.max_drawdown
        @test CONSERVATIVE.min_probability_positive > default.min_probability_positive
        @test CONSERVATIVE.max_probability_large_loss < default.max_probability_large_loss
    end
end
