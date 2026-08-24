# Phase A: the live feed under hostile input.
#
# Every test here corresponds to a failure mode a real feed produces. Several were defects
# when this file was written; the rest establish that the behaviour is correct rather than
# merely current.

const FA_AT = DateTime(2026, 1, 5, 10, 0)   # a Monday, mid-session

@testset "feed audit: what a quote refuses" begin
    @testset "a price must be finite, not merely positive" begin
        # `Inf > 0` is true. An infinite price propagates into bars, features and posteriors
        # without ever raising, and it is not a price.
        @test_throws ArgumentError Quote("X", FA_AT, Inf)
        @test_throws ArgumentError Quote("X", FA_AT, NaN)
        @test_throws ArgumentError Quote("X", FA_AT, 0.0)
        @test_throws ArgumentError Quote("X", FA_AT, -1.0)
        @test Quote("X", FA_AT, 1.0e-8).last_price > 0
    end

    @testset "a volume must be finite and non-negative" begin
        # Volume reaches the participation cap and the liquidity gate. A NaN there makes
        # every downstream comparison false, which is the direction that trades.
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; volume = -5.0)
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; volume = NaN)
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; volume = Inf)
        @test Quote("X", FA_AT, 100.0; volume = 0.0).volume == 0.0
    end

    @testset "a book must be a book" begin
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; bid = -1.0, ask = 101.0)
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; bid = 99.0, ask = Inf)
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; bid = 99.0, ask = NaN)
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; bid_quantity = -1)
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; ask_quantity = -1)
        @test_throws ArgumentError Quote("X", FA_AT, 100.0; bid = 101.0, ask = 99.0)
        @test Quote("X", FA_AT, 100.0; bid = 99.0, ask = 101.0).bid ≈ 99.0
    end
end

@testset "feed audit: ordering, duplication and silence" begin
    @testset "a replayed history does not rewrite the present" begin
        # On a reconnect a socket resends what it already sent. Folding an older tick into
        # the open bar would rewrite a price the system has already acted on.
        session = FeedSession("X"; interval = Minute(1))
        handle_tick!(session, Quote("X", FA_AT, 100.0))
        verdict, bar = handle_tick!(session, Quote("X", FA_AT - Minute(5), 999.0))
        @test verdict === :out_of_order
        @test bar === nothing
        @test session.aggregator.high ≈ 100.0

        again, _ = handle_tick!(session, Quote("X", FA_AT, 100.0))
        @test again === :duplicate
        @test session.aggregator.n_ticks == 1
        @test session.health.n_out_of_order == 1
        @test session.health.n_duplicates == 1
    end

    @testset "a tick for another instrument is ignored, not fatal" begin
        # A socket carries several subscriptions. Throwing would kill a session meant to run
        # unattended over an event that is not an error.
        session = FeedSession("X"; interval = Minute(1))
        handle_tick!(session, Quote("X", FA_AT, 100.0))
        verdict, bar = handle_tick!(session, Quote("Y", FA_AT + Second(1), 5.0))
        @test verdict === :foreign
        @test bar === nothing
        @test session.aggregator.low ≈ 100.0
        @test session.health.n_accepted == 1
    end

    @testset "silence is stale, and having never spoken is stale too" begin
        health = FeedHealth(max_silence = Minute(2))
        @test is_stale(health, FA_AT)
        accept!(health, Quote("X", FA_AT, 100.0))
        @test !is_stale(health, FA_AT + Minute(2))
        @test is_stale(health, FA_AT + Minute(2) + Second(1))
    end
end

@testset "feed audit: bars and session boundaries" begin
    @testset "the overnight gap cannot be swallowed into a bar" begin
        # Structural rather than a rule: buckets are absolute timestamps floored to the
        # interval, so two ticks on different dates never share a bucket for any interval
        # shorter than a day. Asserted across intervals so a change to the flooring that
        # broke it would fail here.
        for interval in (Minute(1), Minute(15), Hour(1), Day(1))
            aggregator = BarAggregator("X"; interval = interval)
            push_tick!(aggregator, Quote("X", DateTime(2026, 1, 2, 15, 20), 100.0))
            closed = push_tick!(aggregator, Quote("X", DateTime(2026, 1, 5, 9, 20), 110.0))
            @test closed !== nothing
            @test Date(closed.timestamp) == Date(2026, 1, 2)
        end
    end

    @testset "a partial bar is not published until it closes" begin
        # The last trade of a bucket is its close. A bar published when the clock passed the
        # bucket end would be published before the market decided what its close was.
        aggregator = BarAggregator("X"; interval = Minute(1))
        @test push_tick!(aggregator, Quote("X", FA_AT, 100.0)) === nothing
        @test push_tick!(aggregator, Quote("X", FA_AT + Second(59), 105.0)) === nothing
        @test has_open_bar(aggregator)
        closed = push_tick!(aggregator, Quote("X", FA_AT + Second(61), 90.0))
        @test closed.close ≈ 105.0
        @test closed.high ≈ 105.0
        # And the tick that closed it opened the next.
        @test flush!(aggregator).open ≈ 90.0
    end

    @testset "a gap produces a gap, not an invented bar" begin
        aggregator = BarAggregator("X"; interval = Minute(1))
        push_tick!(aggregator, Quote("X", FA_AT, 100.0))
        jumped = push_tick!(aggregator, Quote("X", FA_AT + Hour(3), 120.0))
        @test jumped.timestamp == FA_AT
        @test flush!(aggregator).timestamp == FA_AT + Hour(3)
        # No empty bars were manufactured for the silent hours between.
    end
end

@testset "feed audit: market hours and the timezone boundary" begin
    hours = MarketHours()

    @testset "an epoch second is converted once, at the boundary" begin
        # Kite sends epoch seconds and `unix2datetime` yields UTC. Bucketing a UTC stamp as
        # though it were exchange-local puts every bar five and a half hours out of place
        # and moves the open to 03:45.
        epoch = 1_767_240_000
        @test unix2datetime(epoch) == DateTime(2026, 1, 1, 4, 0)
        @test exchange_time(hours, epoch) == DateTime(2026, 1, 1, 9, 30)
        @test exchange_time(hours, epoch) - unix2datetime(epoch) == IST_OFFSET
        @test IST_OFFSET == Minute(330)
    end

    @testset "the session has edges and weekends are closed" begin
        @test !is_open(hours, DateTime(2026, 1, 5, 9, 14, 59))
        @test is_open(hours, DateTime(2026, 1, 5, 9, 15))
        @test is_open(hours, DateTime(2026, 1, 5, 15, 30))
        @test !is_open(hours, DateTime(2026, 1, 5, 15, 30, 1))
        @test !is_open(hours, DateTime(2026, 1, 3, 11, 0))   # Saturday
        @test !is_open(hours, DateTime(2026, 1, 4, 11, 0))   # Sunday

        @test session_bounds(hours, Date(2026, 1, 5)) ==
            (DateTime(2026, 1, 5, 9, 15), DateTime(2026, 1, 5, 15, 30))
        @test !same_session(hours, DateTime(2026, 1, 2, 15, 30), DateTime(2026, 1, 5, 9, 15))
        @test same_session(hours, DateTime(2026, 1, 5, 9, 15), DateTime(2026, 1, 5, 15, 30))
        @test_throws ArgumentError MarketHours(open = Time(15, 30), close = Time(9, 15))
    end
end
