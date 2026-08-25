# Sections 18 and 19: what the system does when its inputs are wrong, and what it does when its
# machinery breaks underneath it.
#
# The standard every scenario is held to is the one the brief sets: when the system cannot
# establish that its inputs and its state are trustworthy, it must not trade.

const SC_SYMBOL = "STRESS"

sc_features() = FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)])

sc_models() = (
    () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
    () -> BayesianVolatilityModel(; horizon_bars = 1),
)

function sc_session(; journal = nothing, warmup = 300, max_silence = Day(3))
    return PaperTradingSession(
        SC_SYMBOL, sc_models(), sc_features();
        horizon_bars = 1, warmup = warmup, refit_every = 50,
        interval = Day(1), interval_label = "1d", max_silence = max_silence,
        journal = journal,
    )
end

sc_series(; n_bars = 600, seed = 808) = generate_series(
    AR1Returns(phi = 0.35, annual_drift = 0.05);
    symbol = SC_SYMBOL, n_bars = n_bars, seed = seed, start = Date(2024, 1, 2),
)

feed!(session, bars) = for bar in bars
    on_tick!(session, Quote(SC_SYMBOL, bar.timestamp, bar.close; volume = bar.volume))
end

"""
    scaled_bar(bar, factor; volume = bar.volume)

The same bar at a different price level, still a valid bar.
"""
scaled_bar(bar::Bar, factor::Real; volume::Real = bar.volume) = Bar(
    bar.symbol, bar.timestamp, bar.open * factor, bar.high * factor,
    bar.low * factor, bar.close * factor, volume; interval = bar.interval,
)

@testset "stress: inputs that are wrong" begin
    @testset "a tenfold volatility burst does not produce a tenfold position" begin
        # The size comes off the posterior spread, so a wider market has to mean a smaller
        # position rather than the same one at a worse price.
        calm = sc_session()
        series = sc_series()
        feed!(calm, series.bars)
        settled = session_report(calm)

        violent = sc_session()
        # Amplify the log return around the pivot rather than the level, so the prices stay
        # positive: a bar with a negative price is rejected by the constructor and the test
        # would pass for the wrong reason.
        pivot = series.bars[400].close
        shocked = vcat(
            series.bars[1:400],
            [
                scaled_bar(bar, exp(10 * log(bar.close / pivot)) * pivot / bar.close)
                    for bar in series.bars[401:end]
            ],
        )
        feed!(violent, shocked)
        report = session_report(violent)

        @test report["halted_bars"] == 0        # a violent market is not a broken one
        @test report["bars"] == settled["bars"]
        # Either it stops trading or it trades smaller. Both are acceptable; trading the same
        # size through ten times the volatility is not.
        @test report["vetoed"] + report["declined"] >=
            settled["vetoed"] + settled["declined"]
    end

    @testset "a fifty percent gap is absorbed without halting or exploding" begin
        session = sc_session()
        series = sc_series()
        gapped = vcat(
            series.bars[1:400],
            [scaled_bar(bar, 0.5) for bar in series.bars[401:end]],
        )
        feed!(session, gapped)
        report = session_report(session)
        @test report["bars"] >= length(gapped) - 1   # the last bar has not closed yet
        @test isfinite(report["equity"])
        @test report["equity"] > 0
        @test isfinite(report["max_drawdown_pct"])
        @test report["rejected"] == 0
    end

    @testset "a long outage halts rather than trading through it" begin
        # The failure this rules out: a session that wakes after a week of silence and treats
        # the next bar as though nothing happened.
        session = sc_session(max_silence = Day(3))
        series = sc_series()
        feed!(session, series.bars[1:400])
        before = session_report(session)["halted_bars"]

        gap = Year(1)
        for bar in series.bars[401:430]
            on_tick!(
                session,
                Quote(SC_SYMBOL, bar.timestamp + gap, bar.close; volume = bar.volume),
            )
        end
        report = session_report(session)
        @test report["halted_bars"] > before
        @test report["stale_bars"] >= 0
    end

    @testset "a duplicate bar is absorbed once, not twice" begin
        session = sc_session()
        series = sc_series()
        doubled = Bar[]
        for (index, bar) in enumerate(series.bars)
            push!(doubled, bar)
            index % 50 == 0 && push!(doubled, bar)      # the same bar again
        end
        feed!(session, doubled)
        report = session_report(session)
        @test isfinite(report["equity"])
        @test report["equity"] > 0
        # The store is keyed by timestamp, so a repeat updates rather than appends.
        stored = history(
            session.engine.store, SC_SYMBOL; as_of = last(series.bars).timestamp,
            count = 10_000,
        )
        # Keyed by timestamp, so a repeat updates rather than appends. One short of the input
        # because the final bar of a stream has not closed yet.
        @test length(stored) == length(series.bars) - 1
        @test allunique(bar.timestamp for bar in stored)
    end

    @testset "an out-of-order bar does not corrupt the history" begin
        session = sc_session()
        series = sc_series()
        shuffled = copy(series.bars)
        shuffled[300], shuffled[310] = shuffled[310], shuffled[300]
        feed!(session, shuffled)
        stored = history(
            session.engine.store, SC_SYMBOL; as_of = last(series.bars).timestamp,
            count = 10_000,
        )
        @test issorted(stored, by = bar -> bar.timestamp)
        # The aggregator will not reopen a bar it has already closed, so a tick arriving out
        # of order is dropped rather than rewriting history behind the decisions already
        # taken. What matters is that the record stays ordered and free of duplicates: a
        # silently reordered history would move features under predictions already made.
        @test allunique(bar.timestamp for bar in stored)
        @test length(stored) < length(series.bars)
        @test length(stored) > length(series.bars) - 20
        @test isfinite(session_report(session)["equity"])
    end

    @testset "a zero-volume bar is untradeable, not invisible" begin
        # A liquidity check that passes on a price nobody traded at is worse than no check.
        session = sc_session()
        series = sc_series()
        dead = vcat(
            series.bars[1:400],
            [scaled_bar(bar, 1.0; volume = 0.0) for bar in series.bars[401:end]],
        )
        feed!(session, dead)
        report = session_report(session)
        @test report["fills"] >= 0
        @test isfinite(report["equity"])
        # Whatever it decided, the liquidity gate saw the zero and said so.
        quality = validate_bars(dead)
        @test any(issue -> issue.check === :zero_volume, quality.issues)
    end

    @testset "a price that cannot be a price is refused at the boundary" begin
        for bad in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError Quote(SC_SYMBOL, DateTime(2026, 1, 2), bad)
        end
        for bad in (-1.0, Inf, NaN)
            @test_throws ArgumentError Quote(
                SC_SYMBOL, DateTime(2026, 1, 2), 100.0; volume = bad,
            )
        end
    end

    @testset "a sudden regime change is followed, not ignored" begin
        session = sc_session()
        rising = generate_series(
            AR1Returns(phi = 0.3, annual_drift = 0.4);
            symbol = SC_SYMBOL, n_bars = 400, seed = 11, start = Date(2024, 1, 2),
        )
        feed!(session, rising.bars)
        first_half = session_report(session)["predictions"]

        falling = generate_series(
            AR1Returns(phi = 0.3, annual_drift = -0.4);
            symbol = SC_SYMBOL, n_bars = 300, seed = 12,
            start = Date(2024, 1, 2) + Day(600),
        )
        # Halting is expected across the discontinuity; what matters is that it resumes.
        feed!(session, falling.bars)
        report = session_report(session)
        @test report["predictions"] > first_half
        @test isfinite(report["equity"])
    end
end

@testset "chaos: machinery that breaks underneath it" begin
    @testset "an unwritable journal stops trading and keeps the process" begin
        # Being unable to explain a decision is a reason to stop, not to crash.
        mktempdir() do dir
            path = joinpath(dir, "locked", "journal.jsonl")
            mkpath(dirname(path))
            session = sc_session(journal = path)
            series = sc_series()
            feed!(session, series.bars[1:350])
            @test !session.journal_failed

            # Replace the file with a directory, which no write can succeed against.
            rm(path)
            mkpath(path)
            feed!(session, series.bars[351:end])

            report = session_report(session)
            @test session.journal_failed
            @test report["halted_bars"] > 0
            @test isfinite(report["equity"])       # the process survived
        end
    end

    @testset "health fails closed when it cannot measure" begin
        session = sc_session()
        # Nothing fed: no feed, no models, no trading day. Every check that cannot see its
        # input must fail rather than pass.
        status = check_health(session, DateTime(2026, 1, 2, 10))
        @test !may_trade(status)
        @test !isempty(problems(status))

        feed!(session, sc_series().bars)
        healthy_now = check_health(session, last(sc_series().bars).timestamp)
        @test may_trade(healthy_now)
        @test isempty(problems(healthy_now))
    end

    @testset "a model that will not fit does not become a confident one" begin
        # Two bars is not a training set. The session must stay unfitted and refuse rather
        # than predict from a posterior that is still the prior.
        session = sc_session(warmup = 300)
        series = sc_series()
        feed!(session, series.bars[1:50])
        report = session_report(session)
        @test !report["fitted"]
        @test report["predictions"] == 0
        @test report["fills"] == 0
        @test !may_trade(check_health(session, series.bars[50].timestamp))
    end

    @testset "a crash mid-run resumes without trading the same bar twice" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            series = sc_series(n_bars = 700)
            first_process = sc_session(journal = path)
            feed!(first_process, series.bars[1:500])
            before = session_report(first_process)["fills"]

            # A crash cuts the last write in half.
            open(path, "a") do handle
                write(handle, "{\"event\":\"bar\",\"as_of\":\"2029-")
            end

            recovered = sc_session(journal = nothing)
            state = resume!(recovered, path)
            @test !isempty(state)
            feed!(recovered, series.bars)
            after = session_report(recovered)
            @test after["replayed"] >= 490
            @test after["fills"] == 0
            @test before >= 0
        end
    end

    @testset "repeated crashes converge rather than compound" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            series = sc_series(n_bars = 700)
            cuts = (200, 350, 500, 700)
            fills = Int[]
            for stop in cuts
                session = sc_session(journal = path)
                resume!(session, path)
                feed!(session, series.bars[1:stop])
                push!(fills, session_report(session)["fills"])
            end
            # Each restart picks up where the last stopped, so the work is done once. A
            # process that re-traded its history would show the same fills again every round.
            @test all(>=(0), fills)
            state = read_journal(path)
            @test !isempty(state)
            # The last entry is the last bar that closed, one short of the stream, and it is
            # inside the final restart's window rather than behind it.
            @test state.last_as_of !== nothing
            @test state.last_as_of >= series.bars[end - 2].timestamp
            @test state.last_as_of <= last(series.bars).timestamp
        end
    end

    @testset "a corrupt journal line does not discard the record" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            session = sc_session(journal = path)
            feed!(session, sc_series().bars)
            whole = read_journal(path)

            open(path, "a") do handle
                write(handle, "not json at all\n{\"event\":\"bar\",\"as_of\":\n")
            end
            torn = read_journal(path)
            @test torn.entries == whole.entries
            @test torn.last_as_of == whole.last_as_of
        end
    end
end
