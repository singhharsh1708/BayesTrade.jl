# Phase E: what stands between running this and sending a real order.
#
# These tests exist to fail loudly if a future change quietly removes a barrier. The strongest
# one is not a check at all: there is no live broker implementation, so there is no code path
# to a venue rather than a guarded one.

using InteractiveUtils: subtypes

@testset "safety gates" begin
    @testset "no broker exists that can reach a venue" begin
        # The first and strongest gate is absence. Adding a live broker is a deliberate
        # act that will make this test fail, which is the point: it should not be possible
        # to do it by accident or as a side effect of something else.
        @test subtypes(Broker) == [PaperBroker]
        @test broker_mode(PaperBroker()) === PAPER
        @test !is_live(PaperBroker())

        # The generic fallback refuses rather than doing something plausible.
        struct _NotABroker <: Broker end
        @test_throws ArgumentError broker_mode(_NotABroker())
        @test_throws ArgumentError place_order!(
            _NotABroker(),
            Order(
                id = "x", symbol = "X", side = BUY_SIDE, quantity = 1.0,
                placed_at = DateTime(2026, 1, 5),
            ),
            Quote("X", DateTime(2026, 1, 5), 100.0),
        )
    end

    @testset "a paper session cannot hold anything but a paper broker" begin
        # Typed rather than checked. A live broker could not be put here even if one existed.
        @test fieldtype(PaperTradingSession, :broker) === PaperBroker
    end

    @testset "a live Kite session takes two independent keys" begin
        credentials = KiteCredentials(api_key = "k", api_secret = "s")
        stub(_) = KiteResponse(200, """{"status":"success","data":{}}""")

        withenv("BAYESTRADE_ALLOW_LIVE_TRADING" => nothing) do
            @test_throws ArgumentError KiteSession(credentials, stub; mode = LIVE)
        end
        withenv("BAYESTRADE_ALLOW_LIVE_TRADING" => "false") do
            @test_throws ArgumentError KiteSession(credentials, stub; mode = LIVE)
        end
        withenv("BAYESTRADE_ALLOW_LIVE_TRADING" => "0") do
            @test_throws ArgumentError KiteSession(credentials, stub; mode = LIVE)
        end
        # Paper needs no ceremony, which is the default everything else runs on.
        @test broker_mode(KiteSession(credentials, stub)) === PAPER
    end

    @testset "settings refuse a half-set live configuration" begin
        withenv(
            "BAYESTRADE_TRADING_MODE" => "live",
            "BAYESTRADE_ALLOW_LIVE_TRADING" => nothing,
        ) do
            @test_throws ArgumentError load_settings()
        end
        withenv(
            "BAYESTRADE_TRADING_MODE" => "live",
            "BAYESTRADE_ALLOW_LIVE_TRADING" => "false",
        ) do
            @test_throws ArgumentError load_settings()
        end
        withenv(
            "BAYESTRADE_TRADING_MODE" => nothing,
            "BAYESTRADE_ALLOW_LIVE_TRADING" => nothing,
        ) do
            @test load_settings().trading_mode === PAPER
        end
    end

    @testset "a session with no credentials is the normal case" begin
        withenv("KITE_API_KEY" => nothing, "KITE_API_SECRET" => nothing) do
            @test credentials_from_env() === nothing
        end
    end
end
