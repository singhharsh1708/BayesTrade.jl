@testset "enums" begin
    @testset "trading mode" begin
        @test is_simulated(BACKTEST)
        @test is_simulated(PAPER)
        @test !is_simulated(LIVE)
        @test slug(PAPER) == "paper"
    end

    @testset "no trade is not a hold" begin
        @test NO_TRADE !== HOLD
        @test is_actionable(BUY)
        @test is_actionable(SELL)
        @test !is_actionable(HOLD)
        @test !is_actionable(NO_TRADE)
    end

    @testset "terminal order statuses" begin
        for status in (FILLED, CANCELLED, REJECTED)
            @test is_terminal(status)
        end
        for status in (PENDING, OPEN, PARTIALLY_FILLED)
            @test !is_terminal(status)
        end
    end

    @testset "slugs are stable serialised forms" begin
        @test slug(BUY_SIDE) == "buy"
        @test slug(SELL_SIDE) == "sell"
        @test slug(NSE) == "NSE"
        @test slug(HIGH_VOLATILITY) == "high_volatility"
        @test slug(EDGE_TOO_SMALL) == "edge_too_small"
    end

    @testset "every regime and model has a distinct slug" begin
        @test length(unique(slug.(instances(Regime)))) == length(instances(Regime))
        @test length(unique(slug.(instances(ModelName)))) == length(instances(ModelName))
    end
end
