# The live feed. Every test here is about a failure that is silent by default.

const FD_START = DateTime(2026, 1, 2, 9, 15)

fd_quote(offset::Period, price = 2_500.0; volume = 100.0, symbol = "RELIANCE") =
    Quote(symbol, FD_START + offset, price; volume = volume)

fd_ticks(n = 300; step = Second(10)) =
    Quote[fd_quote(step * i, 2_500.0 + 5 * sin(i / 7)) for i in 0:(n - 1)]

fd_session(; kwargs...) = FeedSession("RELIANCE"; kwargs...)

@testset "feed health" begin
    @testset "a feed that has never spoken is stale, not fresh" begin
        # Treating an absent feed as healthy until it proves otherwise is the assumption
        # that gets a system trading on nothing.
        health = FeedHealth(max_silence = Minute(2))
        @test is_stale(health, FD_START)
        @test silence(health, FD_START) === nothing
        @test health.last_tick_at === nothing
    end

    @testset "silence past the limit is staleness" begin
        health = FeedHealth(max_silence = Minute(2))
        @test accept!(health, fd_quote(Second(0))) === :accepted
        @test !is_stale(health, FD_START + Minute(1))
        @test !is_stale(health, FD_START + Minute(2))
        @test is_stale(health, FD_START + Minute(2) + Second(1))
        @test silence(health, FD_START + Minute(1)) == Minute(1)
        @test_throws ArgumentError FeedHealth(max_silence = Second(0))
    end

    @testset "a replayed history does not rewrite the present" begin
        # On a real socket an older tick is a reconnection replaying what it already sent.
        # Folding it in would rewrite a price the system has already acted on.
        health = FeedHealth()
        @test accept!(health, fd_quote(Minute(5))) === :accepted
        @test accept!(health, fd_quote(Minute(4))) === :out_of_order
        @test accept!(health, fd_quote(Minute(5))) === :duplicate
        @test health.n_accepted == 1
        @test health.n_out_of_order == 1
        @test health.n_duplicates == 1
        @test health.last_tick_at == FD_START + Minute(5)

        mark_stale!(health)
        @test health.n_stale == 1
        @test occursin("FeedHealth", sprint(show, health))
    end
end

@testset "bar aggregation" begin
    @testset "bars land on the interval, not on the first tick" begin
        # Two aggregators started at different times must agree on where the boundaries
        # are, or two machines running the same strategy disagree about what a bar is.
        early = BarAggregator("RELIANCE"; interval = Minute(1))
        late = BarAggregator("RELIANCE"; interval = Minute(1))
        @test bucket_of(early, DateTime(2026, 1, 2, 9, 15, 37)) ==
            DateTime(2026, 1, 2, 9, 15, 0)
        @test bucket_of(early, DateTime(2026, 1, 2, 9, 15, 0)) ==
            DateTime(2026, 1, 2, 9, 15, 0)
        @test bucket_of(late, DateTime(2026, 1, 2, 9, 16, 59)) ==
            DateTime(2026, 1, 2, 9, 16, 0)

        hourly = BarAggregator("RELIANCE"; interval = Hour(1), label = "1h")
        @test bucket_of(hourly, DateTime(2026, 1, 2, 9, 45)) == DateTime(2026, 1, 2, 9, 0)
    end

    @testset "a bar closes when the next one opens, never on a timer" begin
        # The last trade of a bucket is the close. A bar published when the clock passed
        # its end would be published before the market had finished deciding.
        aggregator = BarAggregator("RELIANCE"; interval = Minute(1))
        @test !has_open_bar(aggregator)
        @test push_tick!(aggregator, fd_quote(Second(0), 100.0)) === nothing
        @test has_open_bar(aggregator)
        @test push_tick!(aggregator, fd_quote(Second(30), 110.0)) === nothing
        @test push_tick!(aggregator, fd_quote(Second(50), 90.0)) === nothing

        closed = push_tick!(aggregator, fd_quote(Second(70), 105.0))
        @test closed !== nothing
        @test closed.timestamp == FD_START
        @test closed.open ≈ 100.0
        @test closed.high ≈ 110.0
        @test closed.low ≈ 90.0
        @test closed.close ≈ 90.0
        @test closed.volume ≈ 300.0
        @test closed.interval == "1m"

        # The tick that closed the old bar opened the new one.
        @test has_open_bar(aggregator)
        final = flush!(aggregator)
        @test final.open ≈ 105.0
        @test final.timestamp == FD_START + Minute(1)
        @test flush!(aggregator) === nothing
    end

    @testset "a gap in the feed is a gap, not a bar of nothing" begin
        # No tick means no bar. Inventing an empty bar for a silent minute would hand the
        # models a price that never printed.
        aggregator = BarAggregator("RELIANCE"; interval = Minute(1))
        push_tick!(aggregator, fd_quote(Second(0), 100.0))
        jumped = push_tick!(aggregator, fd_quote(Minute(10), 120.0))
        @test jumped.timestamp == FD_START
        @test jumped.close ≈ 100.0
        @test flush!(aggregator).timestamp == FD_START + Minute(10)
    end

    @testset "it refuses what it cannot aggregate" begin
        aggregator = BarAggregator("RELIANCE"; interval = Minute(1))
        push_tick!(aggregator, fd_quote(Minute(5), 100.0))
        @test_throws ArgumentError push_tick!(aggregator, fd_quote(Minute(3), 99.0))
        @test_throws ArgumentError push_tick!(
            aggregator, fd_quote(Minute(6), 99.0; symbol = "INFY"),
        )
        @test_throws ArgumentError BarAggregator("")
        @test_throws ArgumentError BarAggregator("A"; interval = Second(0))
        @test_throws ArgumentError build_bar(BarAggregator("A"))
    end
end

@testset "feed session" begin
    @testset "a recorded session replays to the same bars every time" begin
        ticks = fd_ticks(300)
        first_run = fd_session(; interval = Minute(1))
        second_run = fd_session(; interval = Minute(1))
        run_feed!(first_run, ReplayTickSource(ticks))
        run_feed!(second_run, ReplayTickSource(ticks))
        close_session!(first_run)
        close_session!(second_run)

        @test length(first_run.bars) == 50
        @test length(first_run.bars) == length(second_run.bars)
        for (left, right) in zip(first_run.bars, second_run.bars)
            @test left.timestamp == right.timestamp
            @test left.close == right.close
            @test left.volume == right.volume
        end
        @test all(bar -> Dates.second(bar.timestamp) == 0, first_run.bars)
        @test issorted([bar.timestamp for bar in first_run.bars])
        @test first_run.health.n_accepted == 300
        @test occursin("FeedSession", sprint(show, first_run))
    end

    @testset "a rejected tick never reaches the bar" begin
        session = fd_session(; interval = Minute(1))
        handle_tick!(session, fd_quote(Minute(5), 100.0))
        verdict, bar = handle_tick!(session, fd_quote(Minute(4), 999.0))
        @test verdict === :out_of_order
        @test bar === nothing
        # The stale price never touched the open bar.
        @test session.aggregator.high ≈ 100.0
        @test session.aggregator.close ≈ 100.0

        duplicate, _ = handle_tick!(session, fd_quote(Minute(5), 100.0))
        @test duplicate === :duplicate
        @test session.aggregator.n_ticks == 1
    end

    @testset "only this symbol's ticks are aggregated" begin
        mixed = Quote[
            fd_quote(Second(0), 100.0),
            fd_quote(Second(10), 5.0; symbol = "INFY"),
            fd_quote(Second(20), 110.0),
        ]
        session = fd_session(; interval = Minute(1))
        run_feed!(session, ReplayTickSource(mixed))
        bar = close_session!(session)
        @test bar.high ≈ 110.0
        @test bar.low ≈ 100.0
        @test session.health.n_accepted == 2
    end

    @testset "the source says when it is done" begin
        source = ReplayTickSource(fd_ticks(3))
        @test source_symbols(source) == ["RELIANCE"]
        @test !exhausted(source)
        for _ in 1:3
            @test next_tick!(source) !== nothing
        end
        @test exhausted(source)
        @test next_tick!(source) === nothing
    end

    @testset "a session that went quiet says so" begin
        # The decision the models depend on: a halted feed must be skipped, not absorbed as
        # a run of unchanged prices.
        session = fd_session(; interval = Minute(1), max_silence = Minute(2))
        run_feed!(session, ReplayTickSource(fd_ticks(6)))
        @test !is_stale(session.health, FD_START + Minute(1))
        @test is_stale(session.health, FD_START + Minute(10))
        @test silence(session.health, FD_START + Minute(10)) > Minute(2)
    end
end
