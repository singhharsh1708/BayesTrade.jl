# The network edge. Everything here runs against a server on the loopback interface, so it
# exercises real sockets, real query encoding and real status codes without reaching Groww and
# without needing an account.

"""
Serve `handler` on a free loopback port and hand the block a root URL pointing at it. The port
is taken from the operating system rather than guessed, because a hardcoded one collides with
whatever else is running on the machine and fails in a way that looks like a bug in the client.
"""
function with_stub_groww(body, handler)
    socket = HTTP.Sockets.listenany(HTTP.Sockets.localhost, UInt16(0))
    port = Int(socket[1])
    close(socket[2])
    server = HTTP.serve!(handler, "127.0.0.1", port; verbose = -1)
    try
        return body("http://127.0.0.1:$port/v1")
    finally
        close(server)
    end
end

http_ok(payload) = HTTP.Response(
    200, JSON3.write(Dict("status" => "SUCCESS", "payload" => payload)),
)

http_candle(moment::DateTime, close::Real) =
    Any[string(moment), close - 0.5, close + 1.0, close - 1.0, close, 1000, nothing]

@testset "the http transport" begin
    @testset "it is only there once HTTP is loaded" begin
        # The core resolves, precompiles, backtests and paper trades with no network library.
        # Loading one is a decision the caller makes.
        @test !isempty(methods(groww_transport))
        @test groww_transport() isa Function
        extension = Base.get_extension(BayesTrade, :BayesTradeHTTPExt)
        @test extension !== nothing
    end

    @testset "a request reaches the wire intact" begin
        seen = Ref{Any}(nothing)
        with_stub_groww(
            function (request)
                seen[] = request
                return http_ok(Dict("candles" => Any[]))
            end,
        ) do root
            session = GrowwSession(
                GrowwCredentials(api_key = "key-abc", api_secret = "secret-xyz"),
                groww_transport(root = root); access_token = "tok",
            )
            response = BayesTrade.groww_call(
                session,
                BayesTrade.build_request(
                    session, :GET, "/historical/candles";
                    query = Dict{String, String}(
                        "groww_symbol" => "NSE-WIPRO",
                        "start_time" => "2026-01-05 00:00:00",
                    ),
                ),
            )
            @test response["candles"] == []

            request = seen[]
            @test request.method == "GET"
            @test startswith(request.target, "/v1/historical/candles?")
            # A space and a colon have to survive the URL, or the window silently changes.
            @test occursin("start_time=2026-01-05%2000%3A00%3A00", request.target)
            @test occursin("groww_symbol=NSE-WIPRO", request.target)
            @test HTTP.header(request, "Authorization") == "Bearer tok"
            @test HTTP.header(request, "x-api-version") == "1.0"
        end
    end

    @testset "a token is minted over a real socket" begin
        seen = Ref{Any}(nothing)
        with_stub_groww(
            function (request)
                seen[] = (request.target, String(request.body))
                return HTTP.Response(
                    200,
                    JSON3.write(
                        Dict("token" => "issued", "expiry" => "2026-08-26T06:00:00"),
                    ),
                )
            end,
        ) do root
            session = GrowwSession(
                GrowwCredentials(api_key = "key-abc", api_secret = "secret-xyz"),
                groww_transport(root = root),
            )
            authenticate!(session; timestamp = 1719830400)
            @test session.access_token == "issued"
            @test session.expiry == DateTime(2026, 8, 26, 6)

            target, body = seen[]
            @test target == "/v1/token/api/access"
            sent = JSON3.read(body, Dict{String, Any})
            @test sent["key_type"] == "approval"
            @test sent["timestamp"] == "1719830400"
            @test sent["checksum"] ==
                bytes2hex(BayesTrade.sha256("secret-xyz1719830400"))
        end
    end

    @testset "a refusal is data, not an exception" begin
        # A 429 is how Groww says slow down. Let HTTP raise on it and the classification that
        # turns it into a retry never happens.
        attempts = Ref(0)
        with_stub_groww(
            function (request)
                attempts[] += 1
                attempts[] < 3 && return HTTP.Response(
                    429,
                    JSON3.write(
                        Dict(
                            "status" => "FAILURE",
                            "error" => Dict("code" => "GA429", "message" => "slow down"),
                        ),
                    ),
                )
                return http_ok(
                    Dict("candles" => Any[http_candle(DateTime(2026, 1, 6, 9, 15), 100.0)]),
                )
            end,
        ) do root
            session = GrowwSession(
                GrowwCredentials(api_key = "k", api_secret = "s"),
                groww_transport(root = root); access_token = "tok",
            )
            source = RetryingSource(
                GrowwSource(
                    session; pause_seconds = 0, clock = () -> DateTime(2026, 6, 1),
                );
                attempts = 3, sleeper = _ -> nothing,
            )
            bars = fetch_bars(
                source, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
            )
            @test length(bars) == 1
            @test attempts[] == 3
        end
    end

    @testset "a socket that never opened is transient, not a crash" begin
        # A refused connection is not a refusal by Groww. It comes back as status zero, which
        # is what the retry wrapper reads as worth another attempt.
        transport = groww_transport(root = "http://127.0.0.1:1", timeout = 2)
        session = GrowwSession(
            GrowwCredentials(api_key = "k", api_secret = "s"), transport;
            access_token = "tok",
        )
        response = transport(
            BayesTrade.build_request(session, :GET, "/historical/candles"),
        )
        @test response isa GrowwResponse
        @test response.status == 0
        @test !isempty(response.body)

        attempts = Ref(0)
        source = RetryingSource(
            GrowwSource(session; pause_seconds = 0, clock = () -> DateTime(2026, 6, 1));
            attempts = 2, sleeper = _ -> (attempts[] += 1),
        )
        @test_throws TransientSourceError fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
        )
        @test attempts[] == 1      # it retried rather than giving up on the first refusal
    end

    @testset "a real fetch, over a real socket, end to end" begin
        with_stub_groww(
            function (request)
                query = HTTP.queryparams(HTTP.URI(request.target))
                from = Date(DateTime(query["start_time"], "yyyy-mm-dd HH:MM:SS"))
                to = Date(DateTime(query["end_time"], "yyyy-mm-dd HH:MM:SS"))
                return http_ok(
                    Dict(
                        "candles" => Any[
                            http_candle(DateTime(day) + Hour(9) + Minute(15), 100.0 + index)
                                for (index, day) in enumerate(from:Day(1):to)
                        ],
                    ),
                )
            end,
        ) do root
            session = GrowwSession(
                GrowwCredentials(api_key = "k", api_secret = "s"),
                groww_transport(root = root); access_token = "tok",
            )
            source = GrowwSource(
                session; pause_seconds = 0, clock = () -> DateTime(2026, 6, 1),
            )
            bars = fetch_bars(
                source, "WIPRO"; start = Date(2026, 1, 5), stop = Date(2026, 1, 9),
            )
            @test length(bars) == 5
            @test all(bar -> bar.symbol == "WIPRO", bars)
            @test issorted(bars, by = bar -> bar.timestamp)
            @test first(bars).interval == "1d"
        end
    end

    @testset "connect_groww will not run without credentials in the environment" begin
        # There is no keyword to pass a secret in. An argument is a thing that ends up in a
        # script, a shell history and a stack trace.
        withenv("GROWW_API_KEY" => nothing, "GROWW_API_SECRET" => nothing) do
            failure = try
                connect_groww()
                nothing
            catch error
                error
            end
            @test failure isa ArgumentError
            @test occursin("GROWW_API_KEY", failure.msg)
        end
        @test !any(
            name -> occursin("secret", lowercase(String(name))),
            Base.kwarg_decl(only(methods(connect_groww, Tuple{}))),
        )
    end
end
