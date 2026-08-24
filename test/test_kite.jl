# The Kite client. No network, no credentials: the binary protocol is checked against frames
# laid out byte by byte, and the REST client against a stub transport.

kt_be32(v) = UInt8[(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]
kt_be16(v) = UInt8[(v >> 8) & 0xFF, v & 0xFF]

const KT_TOKEN = 408065          # low byte 1, so NSE_CM
const KT_CDS_TOKEN = 3           # low byte 3, so currency

kt_ltp(token = KT_TOKEN, paise = 254_550) = vcat(kt_be32(token), kt_be32(paise))

kt_quote_packet(token = KT_TOKEN) = vcat(
    kt_be32(token), kt_be32(254_550), kt_be32(12), kt_be32(254_000), kt_be32(1_250_000),
    kt_be32(9_000), kt_be32(8_000), kt_be32(253_000), kt_be32(256_000), kt_be32(252_000),
    kt_be32(253_500),
)

function kt_full_packet(token = KT_TOKEN; stamp = 1_767_240_000)
    depth = UInt8[]
    for rung in 1:10
        depth = vcat(depth, kt_be32(100 * rung), kt_be32(254_000 + 10 * rung), kt_be16(rung), UInt8[0, 0])
    end
    return vcat(
        kt_quote_packet(token), kt_be32(1_767_239_000), kt_be32(4_200), kt_be32(4_300),
        kt_be32(4_100), kt_be32(stamp), depth,
    )
end

kt_frame(packets...) = vcat(
    kt_be16(length(packets)),
    reduce(vcat, [vcat(kt_be16(length(p)), p) for p in packets]; init = UInt8[]),
)

@testset "kite binary protocol" begin
    @testset "the segment decides the divisor" begin
        # A currency future read at the equity divisor is out by a factor of a hundred
        # thousand, which is the kind of error that looks like a price.
        @test segment_of(KT_TOKEN) === NSE_CM
        @test segment_of(KT_CDS_TOKEN) === CDS
        @test segment_of(2) === NSE_FO
        @test segment_of(4) === BSE_CM
        @test price_divisor(NSE_CM) == 100.0
        @test price_divisor(CDS) == 10_000_000.0
        @test price_divisor(BSE_CDS) == 10_000.0
        @test parse_packet(kt_ltp(KT_CDS_TOKEN, 875_000_000)).last_price ≈ 87.5
    end

    @testset "each mode is read at its own offsets" begin
        # A price read four bytes late is still a price, and wrong in a way nothing
        # downstream can detect. Every field is asserted against a byte laid by hand.
        ltp = parse_packet(kt_ltp())
        @test ltp.mode === :ltp
        @test ltp.token == KT_TOKEN
        @test ltp.last_price ≈ 2_545.50
        @test ltp.volume === nothing
        @test ltp.open === nothing
        @test isempty(ltp.bids)

        quoted = parse_packet(kt_quote_packet())
        @test quoted.mode === :quote
        @test quoted.last_price ≈ 2_545.50
        @test quoted.last_quantity == 12
        @test quoted.average_price ≈ 2_540.00
        @test quoted.volume == 1_250_000
        @test quoted.buy_quantity == 9_000
        @test quoted.sell_quantity == 8_000
        @test quoted.open ≈ 2_530.00
        @test quoted.high ≈ 2_560.00
        @test quoted.low ≈ 2_520.00
        @test quoted.close ≈ 2_535.00
        @test quoted.exchange_timestamp === nothing
        @test isempty(quoted.asks)

        full = parse_packet(kt_full_packet())
        @test full.mode === :full
        @test full.last_price ≈ 2_545.50
        @test full.open ≈ 2_530.00
        # Offset 44 is the last traded timestamp and 48 is open interest, so the value
        # laid at 48 is the one that must come back here.
        @test full.open_interest == 4_200
        @test full.exchange_timestamp == unix2datetime(1_767_240_000)
        @test length(full.bids) == 5
        @test length(full.asks) == 5
        @test first(full.bids).quantity == 100
        @test first(full.bids).price ≈ 2_540.10
        @test first(full.bids).orders == 1
        @test first(full.asks).quantity == 600
    end

    @testset "a length it does not recognise is an error, not a guess" begin
        # Reading a 44-byte quote as a 184-byte full packet would build depth rungs out of
        # whatever followed it in the buffer.
        @test_throws KiteProtocolError parse_packet(UInt8[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        @test_throws KiteProtocolError parse_packet(UInt8[1, 2, 3])
        @test_throws KiteProtocolError parse_packet(zeros(UInt8, 100))
        @test occursin(
            "KiteProtocolError", sprint(showerror, KiteProtocolError("x")),
        )
    end

    @testset "a frame carries however many packets it says" begin
        ticks = parse_frame(kt_frame(kt_ltp(), kt_quote_packet(), kt_full_packet()))
        @test length(ticks) == 3
        @test [t.mode for t in ticks] == [:ltp, :quote, :full]
        @test all(t -> t.token == KT_TOKEN, ticks)

        mixed = parse_frame(kt_frame(kt_ltp(KT_TOKEN, 100), kt_ltp(KT_CDS_TOKEN, 875_000_000)))
        @test mixed[1].last_price ≈ 1.0
        @test mixed[2].last_price ≈ 87.5
    end

    @testset "a heartbeat is not an error and a truncated frame is" begin
        # An empty frame is a normal thing to receive. A frame whose lengths run past its
        # end is not, and reading it as far as it goes would assemble a tick from memory.
        @test isempty(parse_frame(UInt8[]))
        @test isempty(parse_frame(kt_be16(0)))
        @test isempty(parse_frame(UInt8[0x00]))
        @test_throws KiteProtocolError parse_frame(vcat(kt_be16(1), kt_be16(200), UInt8[1, 2, 3]))
        @test_throws KiteProtocolError parse_frame(vcat(kt_be16(2), kt_be16(8), kt_ltp()))
    end

    @testset "a tick becomes a quote the rest of the system can read" begin
        full = parse_packet(kt_full_packet())
        price = to_quote(full, "RELIANCE", DateTime(2026, 1, 2, 10))
        @test price isa Quote
        @test price.symbol == "RELIANCE"
        @test price.last_price ≈ 2_545.50
        @test price.timestamp == unix2datetime(1_767_240_000)
        @test price.bid ≈ 2_540.10
        @test price.volume ≈ 1_250_000.0

        # A mode that carried no timestamp falls back to the moment it arrived.
        bare = to_quote(parse_packet(kt_ltp()), "RELIANCE", DateTime(2026, 1, 2, 10))
        @test bare.timestamp == DateTime(2026, 1, 2, 10)
        @test bare.bid === nothing
    end
end

@testset "kite session" begin
    credentials = KiteCredentials(api_key = "abc123", api_secret = "s3cret", user_id = "AB1234")
    ok(body) = _ -> KiteResponse(200, body)

    @testset "a secret that reaches a log has already leaked" begin
        rendered = sprint(show, credentials)
        @test occursin("abc123", rendered)
        @test !occursin("s3cret", rendered)
        @test occursin("redacted", rendered)
        @test_throws ArgumentError KiteCredentials(api_key = "", api_secret = "x")
        @test_throws ArgumentError KiteCredentials(api_key = "x", api_secret = "")
    end

    @testset "the login flow is a human with a browser" begin
        @test occursin("api_key=abc123", login_url(credentials))
        @test startswith(login_url(credentials), "https://kite.zerodha.com/connect/login")
        # sha256(api_key + request_token + api_secret)
        @test session_checksum(credentials, "reqtok") ==
            bytes2hex(BayesTrade.sha256("abc123reqtok" * "s3cret"))
        @test length(session_checksum(credentials, "reqtok")) == 64
    end

    @testset "a request is assembled before it is sent" begin
        # The part with the mistakes in it, asserted with no network and no account.
        session = KiteSession(credentials, ok("{}"); access_token = "tok999")
        @test is_authenticated(session)
        @test authorisation(session) == "token abc123:tok999"

        request = build_request(session, :GET, "/quote/ltp"; query = Dict("i" => "NSE:X"))
        @test request.method === :GET
        @test request.path == "/quote/ltp"
        @test request.query["i"] == "NSE:X"
        @test request.headers["X-Kite-Version"] == "3"
        @test request.headers["Authorization"] == "token abc123:tok999"

        anonymous = build_request(session, :POST, "/session/token"; authenticated = false)
        @test !haskey(anonymous.headers, "Authorization")

        fresh = KiteSession(credentials, ok("{}"))
        @test !is_authenticated(fresh)
        @test_throws KiteError authorisation(fresh)
    end

    @testset "it exchanges a request token for an access token" begin
        session = KiteSession(
            credentials, ok("""{"status":"success","data":{"access_token":"tok999"}}"""),
        )
        authenticate!(session, "reqtok")
        @test is_authenticated(session)
        @test session.access_token == "tok999"

        empty = KiteSession(credentials, ok("""{"status":"success","data":{}}"""))
        @test_throws KiteError authenticate!(empty, "reqtok")
    end

    @testset "a refusal is raised, not returned" begin
        refused = KiteSession(
            credentials,
            _ -> KiteResponse(
                403,
                """{"status":"error","message":"Invalid api_key or access_token","error_type":"TokenException"}""",
            );
            access_token = "t",
        )
        error = try
            kite_call(refused, build_request(refused, :GET, "/x"))
        catch caught
            caught
        end
        @test error isa KiteError
        @test error.status == 403
        @test occursin("TokenException", error.message)
        @test occursin("KiteError(403)", sprint(showerror, error))

        garbled = KiteSession(credentials, ok("not json at all"); access_token = "t")
        @test_throws KiteError kite_call(garbled, build_request(garbled, :GET, "/x"))

        unsuccessful = KiteSession(credentials, ok("""{"status":"pending"}"""); access_token = "t")
        @test_throws KiteError kite_call(unsuccessful, build_request(unsuccessful, :GET, "/x"))
    end

    @testset "a quote comes back as the type everything else speaks" begin
        session = KiteSession(
            credentials,
            ok("""{"status":"success","data":{"NSE:RELIANCE":{"last_price":2545.5}}}""");
            access_token = "t",
        )
        price = kite_quote(session, "NSE", "RELIANCE", DateTime(2026, 1, 2, 10))
        @test price isa Quote
        @test price.last_price ≈ 2_545.5
        @test price.symbol == "RELIANCE"

        missing_key = KiteSession(
            credentials, ok("""{"status":"success","data":{}}"""); access_token = "t",
        )
        @test_throws KiteError kite_quote(missing_key, "NSE", "RELIANCE", DateTime(2026, 1, 2, 10))
    end

    @testset "an order becomes the fields Kite expects" begin
        order = Order(
            id = "x", symbol = "RELIANCE", side = BUY_SIDE, quantity = 12.4,
            placed_at = DateTime(2026, 1, 2, 10),
        )
        params = order_params(order)
        @test params["tradingsymbol"] == "RELIANCE"
        @test params["transaction_type"] == "BUY"
        @test params["order_type"] == "MARKET"
        @test params["quantity"] == "12"
        @test params["validity"] == "DAY"
        @test !haskey(params, "price")

        limit = Order(
            id = "y", symbol = "RELIANCE", side = SELL_SIDE, quantity = 5.0,
            order_type = LIMIT, limit_price = 2_600.456, placed_at = DateTime(2026, 1, 2, 10),
        )
        limit_params = order_params(limit)
        @test limit_params["transaction_type"] == "SELL"
        @test limit_params["order_type"] == "LIMIT"
        @test limit_params["price"] == "2600.46"

        # Fractional weights can round to nothing, and sending a zero-share order is worse
        # than refusing one.
        tiny = Order(
            id = "z", symbol = "RELIANCE", side = BUY_SIDE, quantity = 0.4,
            placed_at = DateTime(2026, 1, 2, 10),
        )
        @test_throws ArgumentError order_params(tiny)
    end

    @testset "live takes two keys, and one is not consent" begin
        @test broker_mode(KiteSession(credentials, ok("{}"))) === PAPER
        withenv("BAYESTRADE_ALLOW_LIVE_TRADING" => nothing) do
            @test_throws ArgumentError KiteSession(credentials, ok("{}"); mode = LIVE)
        end
        withenv("BAYESTRADE_ALLOW_LIVE_TRADING" => "false") do
            @test_throws ArgumentError KiteSession(credentials, ok("{}"); mode = LIVE)
        end
        withenv("BAYESTRADE_ALLOW_LIVE_TRADING" => "true") do
            session = KiteSession(credentials, ok("{}"); mode = LIVE)
            @test broker_mode(session) === LIVE
            @test occursin("live", sprint(show, session))
        end
    end

    @testset "no credentials is a normal state" begin
        # Every test here and every backtest runs without them.
        withenv("KITE_API_KEY" => nothing, "KITE_API_SECRET" => nothing) do
            @test credentials_from_env() === nothing
        end
        withenv("KITE_API_KEY" => "k", "KITE_API_SECRET" => nothing) do
            @test credentials_from_env() === nothing
        end
        withenv("KITE_API_KEY" => "k", "KITE_API_SECRET" => "s", "KITE_USER_ID" => "U1") do
            found = credentials_from_env()
            @test found isa KiteCredentials
            @test found.api_key == "k"
            @test found.user_id == "U1"
        end
    end
end
