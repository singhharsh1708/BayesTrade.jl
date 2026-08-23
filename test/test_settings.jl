env(pairs...) = Dict{String, String}("BAYESTRADE_" * k => v for (k, v) in pairs)

session() = BrokerSettings(
    kite_api_key = Secret("key"),
    kite_api_secret = Secret("secret"),
    kite_access_token = Secret("token"),
)

@testset "settings" begin
    @testset "the default is simulated" begin
        settings = load_settings(env = Dict{String, String}())
        @test settings.trading_mode === PAPER
        @test is_simulated(settings)
        @test !is_live(settings)
        @test !settings.allow_live_trading
    end

    @testset "live trading takes two independent keys" begin
        # One variable is too easy to inherit from a stale shell, a copied compose file or
        # a CI secret, and the failure mode is real money.
        @test_throws ArgumentError load_settings(env = env("TRADING_MODE" => "live"))
        @test_throws ArgumentError Settings(trading_mode = LIVE)
        @test is_simulated(Settings(allow_live_trading = true))
        @test_throws ArgumentError Settings(
            trading_mode = LIVE, allow_live_trading = true,
        )
    end

    @testset "live trading also needs a broker session" begin
        credentials_only = BrokerSettings(
            kite_api_key = Secret("key"), kite_api_secret = Secret("secret"),
        )
        @test has_credentials(credentials_only)
        @test !has_session(credentials_only)
        @test_throws ArgumentError Settings(
            trading_mode = LIVE, allow_live_trading = true, broker = credentials_only,
        )
        live = Settings(trading_mode = LIVE, allow_live_trading = true, broker = session())
        @test is_live(live)
        @test !is_simulated(live)
    end

    @testset "scalars come from the environment" begin
        settings = load_settings(env = env("STARTING_EQUITY" => "250000", "SEED" => "42"))
        @test settings.starting_equity ≈ 250_000
        @test settings.seed == 42
    end

    @testset "nested settings use the double underscore" begin
        settings = load_settings(env = env("RISK__MAX_POSITION_WEIGHT" => "0.01"))
        @test settings.risk.max_position_weight ≈ 0.01
    end

    @testset "an inconsistent limit set fails at load" begin
        @test_throws ArgumentError load_settings(env = env("RISK__MAX_DAILY_LOSS" => "0.5"))
    end

    @testset "an unparsable value is an error, not a silent default" begin
        # Ignoring MAX_DAILY_LOSS=2% would run the system on limits nobody chose.
        @test_throws ArgumentError load_settings(env = env("RISK__MAX_DAILY_LOSS" => "2%"))
        @test_throws ArgumentError load_settings(env = env("SEED" => "many"))
        @test_throws ArgumentError load_settings(env = env("TRADING_MODE" => "yolo"))
    end

    @testset "booleans accept the spellings people actually use" begin
        for raw in ("true", "TRUE", "1", "yes", "on")
            @test load_settings(env = env("ALLOW_LIVE_TRADING" => raw)).allow_live_trading
        end
        for raw in ("false", "0", "no", "off")
            @test !load_settings(env = env("ALLOW_LIVE_TRADING" => raw)).allow_live_trading
        end
    end

    @testset "the universe is a comma separated list" begin
        settings = load_settings(env = env("UNIVERSE" => "RELIANCE, TCS ,INFY"))
        @test settings.universe == ["RELIANCE", "TCS", "INFY"]
        @test isempty(load_settings(env = env("UNIVERSE" => " , ")).universe)
    end

    @testset "secrets do not print" begin
        secret = Secret("supersecret")
        @test !occursin("supersecret", sprint(show, secret))
        @test !occursin("supersecret", sprint(show, MIME"text/plain"(), secret))
        @test !occursin("supersecret", string(secret))
        @test reveal(secret) == "supersecret"
        @test reveal(nothing) === nothing
    end

    @testset "the description never contains a secret value" begin
        settings = Settings(allow_live_trading = true, broker = session())
        rendered = join(last.(describe(settings)), " ")
        @test !occursin("secret", rendered)
        @test !occursin("token", rendered)
        @test Dict(describe(settings))["broker_session"] == "present"
    end

    @testset "doctor states plainly whether live trading is armed" begin
        simulated = sprint(io -> doctor(Settings(); io = io))
        @test occursin("Live trading is not armed", simulated)

        armed = sprint(io -> doctor(Settings(allow_live_trading = true); io = io))
        @test occursin("permitted but not selected", armed)

        live = Settings(trading_mode = LIVE, allow_live_trading = true, broker = session())
        @test occursin("LIVE TRADING IS ARMED", sprint(io -> doctor(live; io = io)))
    end

    @testset "doctor covers every limit that can reject a trade" begin
        rendered = sprint(io -> doctor(Settings(); io = io))
        for name in (
                "max_position_weight", "max_portfolio_exposure", "max_sector_exposure",
                "max_daily_loss", "max_drawdown", "max_open_positions",
                "min_probability_positive", "max_probability_large_loss",
            )
            @test occursin(name, rendered)
        end
    end

    @testset "execution costs are charged on both legs and default non-zero" begin
        costs = ExecutionSettings(slippage_bps = 5.0, brokerage_bps = 3.0, taxes_bps = 12.0)
        @test total_cost_bps(costs) ≈ 20.0
        @test round_trip_cost(costs) ≈ 0.004
        @test round_trip_cost(ExecutionSettings()) > 0
        @test_throws ArgumentError ExecutionSettings(slippage_bps = -1.0)
    end

    @testset "a non-positive starting equity is rejected" begin
        @test_throws ArgumentError Settings(starting_equity = 0.0)
    end
end
