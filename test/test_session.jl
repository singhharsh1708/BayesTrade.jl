# The long-running paper session. These tests are about the failures that only appear over
# time: outcomes used before they exist, a feed that goes quiet, a day that rolls over, and a
# process that gets restarted.

ps_series(; n_bars = 1_200, seed = 7, process = AR1Returns(phi = 0.55, annual_drift = 0.0)) =
    generate_series(
    process; symbol = "SYNTH", n_bars = n_bars, seed = seed, start = Date(2026, 1, 1),
)

ps_features() = FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)])

function ps_session(; horizon = 1, warmup = 700, refit = 50, journal = nothing, models = :all)
    factories = models === :all ?
        (
            () -> BayesianReturnModel([:log_return_1]; horizon_bars = horizon),
            () -> BayesianVolatilityModel(; horizon_bars = horizon),
            () -> MarketRegimeModel(; horizon_bars = horizon),
        ) : (() -> BayesianVolatilityModel(; horizon_bars = horizon),)
    return PaperTradingSession(
        "SYNTH", factories, ps_features();
        horizon_bars = horizon, warmup = warmup, refit_every = refit,
        interval = Day(1), interval_label = "1d", max_silence = Day(3),
        sector = "energy", journal = journal,
    )
end

ps_quote(bar) = Quote("SYNTH", bar.timestamp, bar.close; volume = bar.volume)

function ps_run!(session, series; upto = nothing)
    bars = upto === nothing ? series.bars : series.bars[1:upto]
    for bar in bars
        on_tick!(session, ps_quote(bar))
    end
    close_bar!(session)
    return session
end

@testset "paper trading session" begin
    @testset "it runs the whole pipeline and counts what it did" begin
        session = ps_session()
        ps_run!(session, ps_series())
        report = session_report(session)

        @test report["schema"] == SESSION_SCHEMA_VERSION
        @test session.fitted
        @test report["bars"] == 1_200
        @test report["predictions"] > 0
        @test report["refits"] > 1
        @test report["fills"] > 0
        # Every prediction is either declined, vetoed or approved, and nothing else.
        @test report["declined"] + report["vetoed"] + report["approved"] ==
            report["predictions"]
        @test report["reduced"] <= report["approved"]
        @test occursin("PaperTradingSession", sprint(show, session))
    end

    @testset "an outcome is not used before it exists" begin
        # The queue carries two stamps: when the outcome exists, and which bar was forecast.
        # Using the due time as the bar would score a prediction against a moment that never
        # printed, and almost nothing would settle.
        session = ps_session(; horizon = 5)
        ps_run!(session, ps_series())
        report = session_report(session)
        @test report["predictions"] > 0
        @test report["settled"] > 0
        # Everything settles except what is still in flight at the end.
        @test report["settled"] + report["pending"] == report["predictions"]
        @test report["pending"] <= 5
        @test session.reliability.n_scored == report["settled"]
    end

    @testset "a feed that has gone quiet is skipped, not believed" begin
        # A halted feed looks exactly like a still market. Absorbing it as unchanged prices
        # drives the variance estimate down, which is the direction that must not be wrong.
        series = ps_series()
        session = ps_session()
        for bar in series.bars[1:900]
            on_tick!(session, ps_quote(bar))
        end
        @test !is_stale(session.feed.health, series.bars[900].timestamp)

        # Jump the clock far past the silence limit, then deliver a bar.
        late = series.bars[901]
        gap = Bar(
            "SYNTH", late.timestamp + Day(30), late.open, late.high, late.low, late.close,
            late.volume; interval = "1d",
        )
        before = session.counters.stale_bars
        on_bar!(session, gap)
        @test session.counters.stale_bars == before + 1
        @test is_stale(session.feed.health, gap.timestamp)
    end

    @testset "the day rolling over resets what the day resets" begin
        # The daily-loss limit is measured against the equity the day opened at, so somebody
        # has to notice the day changed.
        session = ps_session()
        series = ps_series()
        ps_run!(session, series; upto = 800)
        first_day = session.current_day
        @test first_day == Date(series.bars[800].timestamp)
        opening = session.day_start_equity

        on_bar!(session, series.bars[801])
        @test session.current_day == Date(series.bars[801].timestamp)
        @test session.current_day > first_day
        # A new day opens at whatever the account is worth now.
        @test session.day_start_equity != opening ||
            equity(session.broker) == opening
    end

    @testset "the journal is written as it happens" begin
        # Weeks of running will not be one process, and a crash must not take the record.
        mktempdir() do dir
            path = joinpath(dir, "nested", "session.jsonl")
            session = ps_session(; journal = path)
            ps_run!(session, ps_series(; n_bars = 900))

            @test isfile(path)
            lines = readlines(path)
            @test length(lines) > 10
            entries = [JSON3.read(line, Dict{String, Any}) for line in lines]
            @test any(e -> e["event"] == "refit", entries)
            @test any(e -> e["event"] == "bar", entries)

            bars = [e for e in entries if e["event"] == "bar"]
            for entry in bars
                @test haskey(entry, "action")
                @test entry["approved"] <= entry["requested"] + 1.0e-12
                if entry["action"] == "no_trade"
                    @test entry["reason"] !== nothing
                end
            end
            # Written line by line, so a truncated file still parses up to the cut.
            @test all(line -> startswith(line, "{"), lines)
        end
    end

    @testset "a session with no journal writes nothing and still runs" begin
        session = ps_session(; journal = nothing)
        ps_run!(session, ps_series(; n_bars = 900))
        @test session.counters.bars == 900
        @test session.journal === nothing
    end

    @testset "it is paper by construction" begin
        # There is no flag that turns this into a live session. That takes another broker.
        session = ps_session()
        @test broker_mode(session.broker) === PAPER
        @test !is_live(session.broker)
        @test session.broker isa PaperBroker
    end

    @testset "before the warm-up it predicts nothing" begin
        session = ps_session(; warmup = 700)
        ps_run!(session, ps_series(; n_bars = 300))
        @test !session.fitted
        @test session.counters.predictions == 0
        @test session.counters.fills == 0
        @test equity(session.broker) == 1.0e6
    end

    @testset "a market with no edge is declined, not traded" begin
        session = ps_session(; models = :one)
        ps_run!(session, ps_series(; process = GaussianReturns(annual_drift = 0.0)))
        report = session_report(session)
        @test report["predictions"] > 0
        # A centred predictive never clears the direction gate.
        @test report["fills"] == 0
        @test report["declined"] == report["predictions"]
        @test equity(session.broker) == 1.0e6
    end

    @testset "it is refused what it cannot run" begin
        @test_throws ArgumentError PaperTradingSession("SYNTH", (), ps_features())
        @test_throws ArgumentError PaperTradingSession(
            "SYNTH", (() -> BayesianVolatilityModel(; horizon_bars = 1),), ps_features();
            horizon_bars = 0,
        )
        @test_throws ArgumentError PaperTradingSession(
            "SYNTH", (() -> BayesianVolatilityModel(; horizon_bars = 1),), ps_features();
            warmup = 0,
        )
        @test_throws ArgumentError PaperTradingSession(
            "SYNTH", (() -> BayesianVolatilityModel(; horizon_bars = 1),), ps_features();
            refit_every = 0,
        )
    end
end
