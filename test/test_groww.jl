# Groww: history for backtests, and nothing else. The first testset is the one that matters,
# because it is the only thing standing between this adapter and an order.

gw_session(transport; token = "stub-token") = GrowwSession(
    GrowwCredentials(api_key = "key-abc", api_secret = "secret-xyz"), transport;
    access_token = token,
)

gw_candle(moment::DateTime, close::Real; volume = 1000) =
    Any[string(moment), close - 0.5, close + 1.0, close - 1.0, close, volume, nothing]

"""
A transport that answers with whatever `candles` returns for the request, and records what it
was asked. The recording is the point: every request this client would send is asserted on
without a network and without an account.
"""
function gw_transport(candles; status = 200)
    seen = GrowwRequest[]
    return seen, function (request::GrowwRequest)
            push!(seen, request)
            return GrowwResponse(
                status,
                JSON3.write(
                    Dict(
                        "status" => "SUCCESS",
                        "payload" => Dict("candles" => candles(request)),
                    ),
                ),
            )
    end
end

# Daily candles across the requested window, one per calendar day.
function gw_daily(request::GrowwRequest; price = 100.0)
    from = Date(DateTime(request.query["start_time"], "yyyy-mm-dd HH:MM:SS"))
    to = Date(DateTime(request.query["end_time"], "yyyy-mm-dd HH:MM:SS"))
    return Any[
        gw_candle(DateTime(day) + Hour(9) + Minute(15), price + Dates.value(day - Date(2026, 1, 1)))
            for day in from:Day(1):to
    ]
end

@testset "groww cannot place an order" begin
    @testset "the only paths it can build are reads" begin
        # A capability that does not exist beats one that is switched off. The whitelist is
        # checked when the request is built, so a refactor or a copied snippet cannot reach an
        # order endpoint by accident.
        @test GROWW_READ_PATHS == Set(["/token/api/access", "/historical/candles"])
        session = gw_session(request -> GrowwResponse(200, "{}"))
        for path in (
                "/order/create", "/order/modify", "/order/cancel", "/margins/detail",
                "/positions/user", "/holdings/user", "/live-data/quote",
            )
            @test_throws ArgumentError build_request(session, :POST, path)
        end
        for path in GROWW_READ_PATHS
            @test build_request(session, :GET, path) isa GrowwRequest
        end
    end

    @testset "a bar source is not a broker" begin
        source = GrowwSource(gw_session(request -> GrowwResponse(200, "{}")))
        @test source isa BarSource
        @test !(source isa Broker)
        # That PaperBroker is the only Broker is asserted in test_safety_gates.jl. Repeating
        # it here would only be a second statement of the same fact, and one that a fake
        # subtype defined by that file makes order-dependent.
    end

    @testset "a request carries the version header and a bearer token" begin
        session = gw_session(request -> GrowwResponse(200, "{}"))
        request = build_request(session, :GET, "/historical/candles")
        @test request.headers["Authorization"] == "Bearer stub-token"
        @test request.headers["x-api-version"] == "1.0"
        @test request.headers["Accept"] == "application/json"

        # Minting a token authenticates with the key itself, so it must not demand a token.
        unauthenticated = GrowwSession(
            GrowwCredentials(api_key = "key-abc", api_secret = "s"),
            request -> GrowwResponse(200, "{}"),
        )
        @test !is_authenticated(unauthenticated)
        @test_throws GrowwError build_request(unauthenticated, :GET, "/historical/candles")
        minted = build_request(
            unauthenticated, :POST, "/token/api/access"; authenticated = false,
        )
        @test minted.headers["Authorization"] == "Bearer key-abc"
    end
end

@testset "groww credentials" begin
    @testset "a secret is never printed" begin
        # A credential that reaches a log has already leaked, and care afterwards does not put
        # it back.
        credentials = GrowwCredentials(api_key = "key-abc", api_secret = "secret-xyz")
        rendered = sprint(show, credentials)
        @test !occursin("secret-xyz", rendered)
        @test !occursin("key-abc", rendered)
        @test occursin("redacted", rendered)
        @test !occursin("secret-xyz", sprint(show, gw_session(r -> r)))
    end

    @testset "absence is a normal state" begin
        withenv("GROWW_API_KEY" => nothing, "GROWW_API_SECRET" => nothing) do
            @test groww_credentials_from_env() === nothing
        end
        withenv("GROWW_API_KEY" => "from-env", "GROWW_API_SECRET" => nothing) do
            credentials = groww_credentials_from_env()
            @test credentials.api_key == "from-env"
            @test !has_secret(credentials)      # a TOTP key has no stored secret
        end
        @test_throws ArgumentError GrowwCredentials(api_key = "")
    end

    @testset "the checksum is sha256 of secret and timestamp" begin
        credentials = GrowwCredentials(api_key = "k", api_secret = "secret-xyz")
        @test access_checksum(credentials, 1719830400) ==
            bytes2hex(BayesTrade.sha256("secret-xyz1719830400"))
        # Passed in rather than read from the clock, so the signature is reproducible.
        @test access_checksum(credentials, 1) != access_checksum(credentials, 2)
    end
end

@testset "minting an access token" begin
    @testset "an approval key signs a checksum" begin
        seen = GrowwRequest[]
        session = GrowwSession(
            GrowwCredentials(api_key = "key-abc", api_secret = "secret-xyz"),
            function (request)
                push!(seen, request)
                return GrowwResponse(
                    200,
                    JSON3.write(
                        Dict("token" => "issued", "expiry" => "2026-08-26T06:00:00"),
                    ),
                )
            end,
        )
        authenticate!(session; timestamp = 1719830400)
        @test is_authenticated(session)
        @test session.access_token == "issued"
        @test session.expiry == DateTime(2026, 8, 26, 6, 0, 0)

        request = only(seen)
        @test request.path == "/token/api/access"
        @test request.body["key_type"] == "approval"
        @test request.body["timestamp"] == "1719830400"
        @test request.body["checksum"] ==
            bytes2hex(BayesTrade.sha256("secret-xyz1719830400"))
        @test !haskey(request.body, "totp")
    end

    @testset "a totp key sends a code and no secret" begin
        seen = GrowwRequest[]
        session = GrowwSession(
            GrowwCredentials(api_key = "key-abc"),
            function (request)
                push!(seen, request)
                return GrowwResponse(200, JSON3.write(Dict("token" => "issued")))
            end,
        )
        authenticate!(session; totp = " 123456 ")
        @test only(seen).body == Dict{String, Any}("key_type" => "totp", "totp" => "123456")
        @test session.expiry === nothing
    end

    @testset "an ambiguous credential is refused, not guessed" begin
        # Guessing which flow the caller meant is how a request goes out signed the wrong way
        # and comes back with an error that says nothing useful.
        with_secret = gw_session(request -> GrowwResponse(200, "{}"))
        @test_throws ArgumentError authenticate!(with_secret; totp = "123456")

        without = GrowwSession(
            GrowwCredentials(api_key = "key-abc"), request -> GrowwResponse(200, "{}"),
        )
        @test_throws ArgumentError authenticate!(without)
        @test_throws ArgumentError authenticate!(without; totp = "   ")
    end

    @testset "a reply with no token is not a success" begin
        session = GrowwSession(
            GrowwCredentials(api_key = "k", api_secret = "s"),
            request -> GrowwResponse(200, JSON3.write(Dict("tokenRefId" => "ref-1"))),
        )
        @test_throws GrowwError authenticate!(session; timestamp = 1)
        @test !is_authenticated(session)
    end

    @testset "a token that is known to be dead says so" begin
        session = gw_session(request -> GrowwResponse(200, "{}"))
        session.expiry = DateTime(2026, 8, 25, 6)
        @test token_expired(session, DateTime(2026, 8, 25, 6, 0, 1))
        @test !token_expired(session, DateTime(2026, 8, 25, 5, 59))
        # An unrecorded expiry is not an expired one: refusing a usable session because the
        # expiry was never returned would be a failure invented here.
        session.expiry = nothing
        @test !token_expired(session, DateTime(2030, 1, 1))
    end
end

@testset "reading a groww reply" begin
    @testset "a refusal wrapped in a 200 is still a refusal" begin
        # Read the code alone and this one passes for a success.
        session = gw_session(
            request -> GrowwResponse(
                200,
                JSON3.write(
                    Dict(
                        "status" => "FAILURE",
                        "error" => Dict("code" => "GA001", "message" => "Bad request."),
                    ),
                ),
            ),
        )
        failure = try
            groww_call(session, build_request(session, :GET, "/historical/candles"))
            nothing
        catch error
            error
        end
        @test failure isa GrowwError
        @test failure.code == "GA001"
        @test occursin("Bad request.", sprint(showerror, failure))
        @test occursin("GA001", sprint(showerror, failure))
    end

    @testset "an unreadable reply is an error, not a crash" begin
        for body in ("not json at all", "[1, 2, 3]", "42")
            session = gw_session(request -> GrowwResponse(200, body))
            @test_throws GrowwError groww_call(
                session, build_request(session, :GET, "/historical/candles"),
            )
        end
        session = gw_session(request -> "a String is not a GrowwResponse")
        @test_throws GrowwError groww_call(
            session, build_request(session, :GET, "/historical/candles"),
        )
    end

    @testset "a status code past 400 is an error even without an envelope" begin
        session = gw_session(request -> GrowwResponse(503, "{}"))
        @test_throws GrowwError groww_call(
            session, build_request(session, :GET, "/historical/candles"),
        )
    end
end

@testset "groww as a bar source" begin
    stub_source(; kwargs...) = GrowwSource(
        gw_session(last(gw_transport(gw_daily)));
        clock = () -> DateTime(2026, 6, 1), pause_seconds = 0, kwargs...,
    )

    @testset "it declares what it can serve" begin
        source = stub_source()
        @test source_name(source) == "groww"
        @test supports(source, "1d")
        @test supports(source, "5m")
        @test !supports(source, "3d")
        @test keys(GROWW_INTERVALS) == keys(GROWW_MAX_WINDOW_DAYS)
        @test_throws ArgumentError fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 1), stop = Date(2026, 2, 1),
            interval = "3d",
        )
        @test_throws ArgumentError fetch_bars(
            source, "WIPRO"; start = Date(2026, 2, 1), stop = Date(2026, 1, 1),
        )
    end

    @testset "a symbol is translated out and stamped back" begin
        # Bars carry the local canonical ticker, so nothing downstream has to know which
        # vendor was used or how it spells things.
        seen, transport = gw_transport(gw_daily)
        source = GrowwSource(
            gw_session(transport); clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
            symbols = Dict("NIFTY50" => "NSE-NIFTY"),
        )
        bars = fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 5), stop = Date(2026, 1, 9),
        )
        @test only(seen).query["groww_symbol"] == "NSE-WIPRO"
        @test all(bar -> bar.symbol == "WIPRO", bars)
        @test vendor_symbol(source, "NIFTY50") == "NSE-NIFTY"
        @test vendor_symbol(source, "WIPRO") == "NSE-WIPRO"
    end

    @testset "the request says what was actually asked for" begin
        seen, transport = gw_transport(gw_daily)
        source = GrowwSource(
            gw_session(transport); clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 5), stop = Date(2026, 1, 9),
            interval = "1d",
        )
        request = only(seen)
        @test request.method === :GET
        @test request.path == "/historical/candles"
        @test request.query["exchange"] == "NSE"
        @test request.query["segment"] == "CASH"
        @test request.query["candle_interval"] == "1day"   # not the local "1d"
        @test request.query["start_time"] == "2026-01-05 00:00:00"
        @test request.query["end_time"] == "2026-01-09 23:59:59"
    end

    @testset "a window wider than the vendor allows is split, not truncated" begin
        # A truncated series still looks like a series, which is the whole problem.
        @test window_chunks(Date(2026, 1, 1), Date(2026, 1, 10), 30) ==
            [(Date(2026, 1, 1), Date(2026, 1, 10))]
        chunks = window_chunks(Date(2026, 1, 1), Date(2026, 3, 31), 30)
        @test length(chunks) == 3
        @test first(first(chunks)) == Date(2026, 1, 1)
        @test last(last(chunks)) == Date(2026, 3, 31)
        for (from, to) in chunks
            @test from <= to
            @test Dates.value(to - from) < 30
        end
        # Contiguous and non-overlapping: every day is covered exactly once.
        for index in 2:length(chunks)
            @test first(chunks[index]) == last(chunks[index - 1]) + Day(1)
        end

        seen, transport = gw_transport(gw_daily)
        source = GrowwSource(
            gw_session(transport); clock = () -> DateTime(2028, 1, 1), pause_seconds = 0,
        )
        stop = Date(2026, 1, 1) + Day(399)
        bars = fetch_bars(source, "WIPRO"; start = Date(2026, 1, 1), stop = stop)
        @test length(seen) == 3      # four hundred days against a hundred and eighty day cap
        @test length(bars) == 400
        @test issorted(bars, by = bar -> bar.timestamp)
    end

    @testset "a candle returned twice is stored once" begin
        # The pieces meet at their edges, and two bars at one moment is a repeated
        # observation that moves every estimate that counts observations.
        repeated = DateTime(2026, 1, 5, 9, 15)
        seen, transport = gw_transport(
            request -> Any[
                gw_candle(repeated, 100.0), gw_candle(repeated, 100.0),
                gw_candle(DateTime(2026, 1, 6, 9, 15), 101.0),
            ],
        )
        source = GrowwSource(
            gw_session(transport); clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        bars = fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 5), stop = Date(2026, 1, 6),
        )
        @test length(bars) == 2
        @test length(unique(bar -> bar.timestamp, bars)) == 2
    end

    @testset "the candle still forming is not served" begin
        # Its close is not a close. A model trained on it learns from a number that never
        # existed at that time, which is a look-ahead bug dressed as a data problem.
        seen, transport = gw_transport(
            request -> Any[
                gw_candle(DateTime(2026, 1, 5, 9, 15), 100.0),
                gw_candle(DateTime(2026, 1, 6, 9, 15), 101.0),
            ],
        )
        # Mid-session on the sixth: the fifth has closed, the sixth has not.
        mid_session = GrowwSource(
            gw_session(transport); pause_seconds = 0,
            clock = () -> DateTime(2026, 1, 6, 12, 0),
        )
        bars = fetch_bars(
            mid_session, "WIPRO"; start = Date(2026, 1, 5), stop = Date(2026, 1, 6),
        )
        @test length(bars) == 1
        @test Date(only(bars).timestamp) == Date(2026, 1, 5)

        # After the close, the same day is usable. A daily candle ends at the session close,
        # not at midnight, so a finished day is not withheld for another eight hours.
        after_close = GrowwSource(
            gw_session(last(gw_transport(request -> Any[gw_candle(DateTime(2026, 1, 6, 9, 15), 101.0)])));
            pause_seconds = 0, clock = () -> DateTime(2026, 1, 6, 15, 31),
        )
        @test length(
            fetch_bars(
                after_close, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
            ),
        ) == 1
    end

    @testset "an intraday candle ends a fixed span after it opens" begin
        opened = DateTime(2026, 1, 6, 9, 15)
        source = stub_source()
        @test BayesTrade.period_end(source, "5m", opened) == opened + Minute(5)
        @test BayesTrade.period_end(source, "30m", opened) == opened + Minute(30)
        @test BayesTrade.period_end(source, "1h", opened) == opened + Hour(1)
        @test BayesTrade.period_end(source, "4h", opened) == opened + Hour(4)
        @test BayesTrade.period_end(source, "1d", opened) == DateTime(2026, 1, 6, 15, 30)
        @test BayesTrade.period_end(source, "1w", opened) == DateTime(2026, 1, 13)
        @test BayesTrade.period_end(source, "1mo", opened) == DateTime(2026, 2, 6)

        seen, transport = gw_transport(
            request -> Any[
                gw_candle(DateTime(2026, 1, 6, 9, 15), 100.0),
                gw_candle(DateTime(2026, 1, 6, 9, 20), 101.0),
            ],
        )
        source = GrowwSource(
            gw_session(transport); pause_seconds = 0,
            clock = () -> DateTime(2026, 1, 6, 9, 23),
        )
        bars = fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
            interval = "5m",
        )
        @test length(bars) == 1
        @test only(bars).interval == "5m"
    end

    @testset "epoch seconds are converted once, not reinterpreted" begin
        # unix2datetime yields UTC and this exchange trades at 09:15 local. Read the stamp as
        # though it were local and the open moves to 03:45.
        opened = DateTime(2026, 1, 6, 9, 15)
        epoch = Dates.datetime2unix(opened - IST_OFFSET)
        seen, transport = gw_transport(
            request -> Any[Any[epoch, 99.5, 101.0, 99.0, 100.0, 1000]],
        )
        source = GrowwSource(
            gw_session(transport); clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        bars = fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
        )
        @test only(bars).timestamp == opened
    end

    @testset "a candle with no volume is an index, not an error" begin
        seen, transport = gw_transport(
            request -> Any[Any["2026-01-06T09:15:00", 99.5, 101.0, 99.0, 100.0, nothing]],
        )
        source = GrowwSource(
            gw_session(transport); clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        @test only(
            fetch_bars(source, "NIFTY"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6)),
        ).volume == 0.0
    end

    @testset "a candle that cannot be a bar is reported" begin
        broken = Any[
            Any["2026-01-06T09:15:00", 99.5, 98.0, 99.0, 100.0, 10],   # high below low
            Any["2026-01-06T09:15:00", 99.5, 101.0, 99.0],             # too few fields
            Any["not a time", 99.5, 101.0, 99.0, 100.0, 10],
            Any["2026-01-06T09:15:00", nothing, 101.0, 99.0, 100.0, 10],
            Any["2026-01-06T09:15:00", -1.0, 101.0, -2.0, 100.0, 10],
        ]
        for candle in broken
            source = GrowwSource(
                gw_session(last(gw_transport(request -> Any[candle])));
                clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
            )
            @test_throws MalformedBarError fetch_bars(
                source, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
            )
        end

        # Not strict: the bad candle is dropped and the good ones still arrive.
        lenient = GrowwSource(
            gw_session(
                last(
                    gw_transport(
                        request -> Any[
                            Any["2026-01-06T09:15:00", 99.5, 98.0, 99.0, 100.0, 10],
                            gw_candle(DateTime(2026, 1, 7, 9, 15), 101.0),
                        ],
                    ),
                ),
            );
            clock = () -> DateTime(2026, 6, 1), pause_seconds = 0, strict = false,
        )
        bars = @test_logs (:warn,) match_mode = :any fetch_bars(
            lenient, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 7),
        )
        @test length(bars) == 1
        @test Date(only(bars).timestamp) == Date(2026, 1, 7)
    end

    @testset "an empty window is empty, not an error" begin
        source = GrowwSource(
            gw_session(last(gw_transport(request -> Any[])));
            clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        @test isempty(
            fetch_bars(source, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6)),
        )
        missing_key = GrowwSource(
            gw_session(
                request -> GrowwResponse(
                    200, JSON3.write(Dict("status" => "SUCCESS", "payload" => Dict())),
                ),
            );
            clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        @test isempty(
            fetch_bars(
                missing_key, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
            ),
        )
    end
end

@testset "groww failures are sorted by whether a retry helps" begin
    @testset "a rate limit is transient and an unknown symbol is not" begin
        # Three attempts do not make a symbol exist, and burning the retry budget on it delays
        # every symbol behind this one in the run.
        @test BayesTrade.translate_error(GrowwError(429, "slow down")) isa
            TransientSourceError
        @test BayesTrade.translate_error(GrowwError(504, "timeout")) isa
            TransientSourceError
        @test BayesTrade.translate_error(GrowwError(503, "unavailable")) isa
            TransientSourceError
        @test BayesTrade.translate_error(GrowwError(0, "transport died")) isa
            TransientSourceError
        @test BayesTrade.translate_error(GrowwError(404, "no such symbol")) isa
            SymbolNotFoundError
        # An authentication failure is not retried either: the token will still be rejected.
        @test BayesTrade.translate_error(GrowwError(401, "bad token")) isa GrowwError
    end

    @testset "the retry wrapper only spends attempts on the transient ones" begin
        attempts = Ref(0)
        flaky = GrowwSource(
            gw_session(
                function (request)
                    attempts[] += 1
                    attempts[] < 3 && return GrowwResponse(
                        429,
                        JSON3.write(
                            Dict(
                                "status" => "FAILURE",
                                "error" => Dict("code" => "GA429", "message" => "slow down"),
                            ),
                        ),
                    )
                    return GrowwResponse(
                        200,
                        JSON3.write(
                            Dict(
                                "status" => "SUCCESS",
                                "payload" => Dict(
                                    "candles" => Any[gw_candle(DateTime(2026, 1, 6, 9, 15), 100.0)],
                                ),
                            ),
                        ),
                    )
                end,
            );
            clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        retrying = RetryingSource(flaky; attempts = 3, sleeper = _ -> nothing)
        @test source_name(retrying) == "groww"
        @test supports(retrying, "1d")
        bars = fetch_bars(
            retrying, "WIPRO"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
        )
        @test length(bars) == 1
        @test attempts[] == 3

        spent = Ref(0)
        unknown = GrowwSource(
            gw_session(
                function (request)
                    spent[] += 1
                    return GrowwResponse(404, JSON3.write(Dict("status" => "SUCCESS")))
                end,
            );
            clock = () -> DateTime(2026, 6, 1), pause_seconds = 0,
        )
        @test_throws SymbolNotFoundError fetch_bars(
            RetryingSource(unknown; attempts = 3, sleeper = _ -> nothing),
            "NOSUCH"; start = Date(2026, 1, 6), stop = Date(2026, 1, 6),
        )
        @test spent[] == 1
    end

    @testset "requests are paced between chunks" begin
        # Groww rate limits, and a long fetch is many requests. The sleeper is injected so the
        # pacing is provable without a suite that waits.
        naps = Float64[]
        source = GrowwSource(
            gw_session(last(gw_transport(gw_daily)));
            clock = () -> DateTime(2028, 1, 1), pause_seconds = 0.25,
            sleeper = seconds -> push!(naps, seconds),
        )
        fetch_bars(
            source, "WIPRO"; start = Date(2026, 1, 1), stop = Date(2026, 1, 1) + Day(399),
        )
        @test naps == [0.25, 0.25]      # between the three chunks, not before the first
        @test_throws ArgumentError GrowwSource(
            gw_session(request -> GrowwResponse(200, "{}")); pause_seconds = -1,
        )
    end
end
