# Phases I and J: can the system prove its state is sound, and does it stop when it cannot.

ht_series(; n = 1_000) = generate_series(
    AR1Returns(phi = 0.55, annual_drift = 0.0);
    symbol = "SYNTH", n_bars = n, seed = 7, start = Date(2026, 1, 1),
)

ht_session(; journal = nothing, silence = Day(3)) = PaperTradingSession(
    "SYNTH", (() -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),),
    FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]);
    horizon_bars = 1, warmup = 700, refit_every = 50,
    interval = Day(1), interval_label = "1d", max_silence = silence, journal = journal,
)

function ht_feed!(session, series; shift_after = nothing, shift = Year(1))
    for (index, bar) in enumerate(series.bars)
        at = (shift_after !== nothing && index > shift_after) ? bar.timestamp + shift :
            bar.timestamp
        on_tick!(session, Quote("SYNTH", at, bar.close; volume = bar.volume))
    end
    return session
end

@testset "health: absence of evidence fails" begin
    @testset "a system that has done nothing may not trade" begin
        # Every check is written so that not being able to measure its input fails. A check
        # that passes when it cannot see is one that turns itself off exactly when something
        # has gone wrong.
        session = ht_session()
        health = check_health(session, DateTime(2026, 1, 1))
        @test !may_trade(health)
        failing = [status.name for status in problems(health)]
        @test :feed in failing          # never spoken
        @test :models in failing        # never fitted
        @test :trading_day in failing   # no session opened
        @test occursin("HALTED", summarise(health))
        @test occursin("no tick has ever arrived", summarise(health))
    end

    @testset "a healthy system may trade, and says why" begin
        session = ht_session()
        ht_feed!(session, ht_series())
        last_bar = last(ht_series().bars).timestamp
        health = check_health(session, last_bar)
        @test may_trade(health)
        @test isempty(problems(health))
        @test all(healthy, health.checks)
        @test occursin("TRADEABLE", summarise(health))
        @test occursin("SystemHealth", sprint(show, health))
    end

    @testset "silence halts it, and the clock is not consulted" begin
        session = ht_session()
        ht_feed!(session, ht_series())
        last_bar = last(ht_series().bars).timestamp
        @test may_trade(check_health(session, last_bar))
        @test !may_trade(check_health(session, last_bar + Day(30)))
        # `now` is an argument, so the verdict is reproducible rather than a race.
        @test check_health(session, last_bar).tradeable ==
            check_health(session, last_bar).tradeable
    end

    @testset "a journal it cannot write is a reason to stop" begin
        # A decision that cannot be written down cannot be explained afterwards, and a trade
        # that cannot be explained is one that should not have happened.
        session = ht_session(journal = "/proc/definitely/not/writable/journal.jsonl")
        ht_feed!(session, ht_series())
        health = check_health(session, last(ht_series().bars).timestamp)
        @test !may_trade(health)
        @test :journal in [status.name for status in problems(health)]
    end
end

@testset "recovery: the session stops rather than guesses" begin
    @testset "a gap in the feed halts the bar that follows it" begin
        # The check has to look at the gap *before* this tick. Accepting the tick is what
        # makes the feed look fresh, so a staleness test run afterwards can never fire.
        series = ht_series()
        healthy_run = ht_feed!(ht_session(), series)
        gapped = ht_feed!(ht_session(), series; shift_after = 850)

        @test session_report(healthy_run)["halted_bars"] == 0
        @test session_report(gapped)["halted_bars"] > 0
        @test session_report(gapped)["predictions"] <
            session_report(healthy_run)["predictions"]
    end

    @testset "the gap that preceded a tick is recorded before the clock moves" begin
        health = FeedHealth(max_silence = Minute(2))
        base = DateTime(2026, 1, 5, 10, 0)
        @test !arrived_after_silence(health)
        accept!(health, Quote("X", base, 100.0))
        @test !arrived_after_silence(health)          # nothing preceded it
        accept!(health, Quote("X", base + Minute(1), 100.0))
        @test !arrived_after_silence(health)
        accept!(health, Quote("X", base + Minute(30), 100.0))
        @test arrived_after_silence(health)
        @test health.last_gap == Minute(29)
        # And the feed looks fresh at that very moment, which is the trap.
        @test !is_stale(health, base + Minute(30))
    end

    @testset "a halted bar is recorded with the reason" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            session = ht_session(journal = path)
            ht_feed!(session, ht_series(); shift_after = 850)
            entries = [JSON3.read(line, Dict{String, Any}) for line in readlines(path)]
            halted = [e for e in entries if e["event"] == "halted"]
            @test !isempty(halted)
            @test !isempty(first(halted)["failing"])
            @test any(f -> f == "feed", first(halted)["failing"])
        end
    end

    @testset "a restart replays to the same place" begin
        # Two processes fed the same bars must reach the same state, or nothing about a
        # restart can be reasoned about.
        series = ht_series()
        first_run = ht_feed!(ht_session(), series)
        second_run = ht_feed!(ht_session(), series)
        left, right = session_report(first_run), session_report(second_run)
        for key in ("bars", "predictions", "fills", "settled", "refits", "halted_bars")
            @test left[key] == right[key]
        end
        @test left["equity"] == right["equity"]
    end
end
