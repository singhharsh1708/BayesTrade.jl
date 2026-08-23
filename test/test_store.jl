const STORE_START = DateTime(2026, 1, 5, 10, 0)

storebar(; symbol = "RELIANCE", offset = 0, close = 100.0, interval = "1d") = Bar(
    symbol, STORE_START + Day(offset), close, close, close, close, 1000.0;
    interval = interval,
)

filled_store(n = 10; symbol = "RELIANCE") =
    InMemoryBarStore([storebar(symbol = symbol, offset = i, close = 100.0 + i) for i in 0:(n - 1)])

@testset "bar store" begin
    @testset "upsert" begin
        @testset "bars are stored in time order regardless of input order" begin
            store = InMemoryBarStore([storebar(offset = 3), storebar(offset = 1), storebar(offset = 2)])
            stamps = [bar.timestamp for bar in load_range(store, "RELIANCE")]
            @test issorted(stamps)
        end

        @testset "a repeated timestamp replaces rather than duplicates" begin
            # Vendors revise bars, and the revision is the better datum.
            store = InMemoryBarStore([storebar(close = 100.0)])
            upsert!(store, [storebar(close = 105.0)])
            bars = load_range(store, "RELIANCE")
            @test length(bars) == 1
            @test bars[1].close ≈ 105.0
        end

        @testset "upsert reports how many bars it took" begin
            store = InMemoryBarStore()
            @test upsert!(store, [storebar(offset = i) for i in 0:4]) == 5
        end

        @testset "symbols are kept apart" begin
            store = InMemoryBarStore([storebar(), storebar(symbol = "TCS")])
            @test symbols(store) == ["RELIANCE", "TCS"]
            @test bar_count(store, "RELIANCE") == 1
        end

        @testset "mixing intervals within a symbol is refused" begin
            store = InMemoryBarStore([storebar(interval = "1d")])
            @test_throws IntervalConflictError upsert!(
                store, [storebar(offset = 1, interval = "5m")],
            )
        end

        @testset "mixing intervals within one batch is refused" begin
            @test_throws IntervalConflictError InMemoryBarStore(
                [storebar(interval = "1d"), storebar(offset = 1, interval = "5m")],
            )
        end

        @testset "the same interval under a different symbol is fine" begin
            store = InMemoryBarStore([storebar(interval = "1d")])
            upsert!(store, [storebar(symbol = "TCS", interval = "5m")])
            @test bar_count(store, "TCS") == 1
        end
    end

    @testset "point-in-time reads" begin
        @testset "history never returns a bar from the future" begin
            store = filled_store(20)
            for index in 0:19
                as_of = STORE_START + Day(index)
                visible = history(store, "RELIANCE"; as_of = as_of)
                @test length(visible) == index + 1
                @test all(bar -> bar.timestamp <= as_of, visible)
            end
        end

        @testset "a moment between bars sees only what had closed" begin
            store = filled_store(10)
            @test length(history(store, "RELIANCE"; as_of = STORE_START + Day(3) + Hour(5))) == 4
        end

        @testset "a moment before the first bar sees nothing" begin
            @test isempty(history(filled_store(10), "RELIANCE"; as_of = STORE_START - Day(1)))
        end

        @testset "count keeps the most recent window" begin
            window = history(
                filled_store(20), "RELIANCE"; as_of = STORE_START + Day(15), count = 5,
            )
            @test length(window) == 5
            @test last(window).timestamp == STORE_START + Day(15)
            @test first(window).timestamp == STORE_START + Day(11)
        end

        @testset "degenerate windows" begin
            store = filled_store(3)
            @test length(history(store, "RELIANCE"; as_of = STORE_START + Day(10), count = 50)) == 3
            @test isempty(history(store, "RELIANCE"; as_of = STORE_START + Day(2), count = 0))
            @test_throws ArgumentError history(store, "RELIANCE"; as_of = STORE_START, count = -1)
        end

        @testset "since bounds the window from below" begin
            window = history(
                filled_store(20), "RELIANCE";
                as_of = STORE_START + Day(15), since = STORE_START + Day(10),
            )
            @test first(window).timestamp == STORE_START + Day(10)
            @test last(window).timestamp == STORE_START + Day(15)
        end

        @testset "an unknown symbol reads as empty rather than raising" begin
            store = filled_store(5)
            @test isempty(history(store, "UNKNOWN"; as_of = STORE_START))
            @test latest(store, "UNKNOWN"; as_of = STORE_START) === nothing
            @test bar_count(store, "UNKNOWN") == 0
            @test !("UNKNOWN" in store)
            @test "RELIANCE" in store
        end

        @testset "latest is the last bar that had closed" begin
            bar = latest(filled_store(10), "RELIANCE"; as_of = STORE_START + Day(4) + Hour(1))
            @test bar !== nothing
            @test bar.timestamp == STORE_START + Day(4)
        end
    end

    @testset "load_range can see past a simulated clock, which is why it is named apart" begin
        store = filled_store(10)
        as_of = STORE_START + Day(2)
        @test length(history(store, "RELIANCE"; as_of = as_of)) == 3
        @test length(load_range(store, "RELIANCE")) == 10
        bars = load_range(store, "RELIANCE", STORE_START + Day(2), STORE_START + Day(5))
        @test [bar.timestamp for bar in bars] == [STORE_START + Day(d) for d in 2:5]
    end

    @testset "coverage describes what is held" begin
        found = coverage(filled_store(10), "RELIANCE")
        @test found.symbol == "RELIANCE"
        @test found.interval == "1d"
        @test found.first == STORE_START
        @test found.last == STORE_START + Day(9)
        @test found.count == 10
        @test (STORE_START + Day(5)) in found
        @test !((STORE_START - Day(1)) in found)
        @test coverage(filled_store(5), "UNKNOWN") === nothing
    end

    @testset "clearing" begin
        store = InMemoryBarStore([storebar(), storebar(symbol = "TCS")])
        clear!(store, "RELIANCE")
        @test symbols(store) == ["TCS"]
        clear!(store)
        @test isempty(symbols(store))

        interval_locked = InMemoryBarStore([storebar(interval = "1d")])
        clear!(interval_locked, "RELIANCE")
        upsert!(interval_locked, [storebar(interval = "5m")])
        @test coverage(interval_locked, "RELIANCE") !== nothing
    end

    @testset "align marks every symbol at one moment" begin
        store = InMemoryBarStore(
            vcat(
                [storebar(offset = i) for i in 0:4],
                [storebar(symbol = "TCS", offset = i) for i in 0:4],
            ),
        )
        aligned = align(store, ["RELIANCE", "TCS"]; as_of = STORE_START + Day(2))
        @test Set(keys(aligned)) == Set(["RELIANCE", "TCS"])
        @test all(bar.timestamp == STORE_START + Day(2) for bar in values(aligned))
    end

    @testset "a symbol with no bar yet is dropped, not filled forward" begin
        # A stale bar presented as current is how a portfolio ends up marked at a price
        # that no longer exists.
        store = InMemoryBarStore([storebar(offset = 0), storebar(symbol = "TCS", offset = 5)])
        aligned = align(store, ["RELIANCE", "TCS"]; as_of = STORE_START + Day(1))
        @test Set(keys(aligned)) == Set(["RELIANCE"])
    end

    @testset "a generated series round trips and slices identically" begin
        s = generate_series(
            GaussianReturns(); symbol = "SYNTH", n_bars = 250, seed = 3,
            start = Date(2024, 1, 1),
        )
        store = InMemoryBarStore(s.bars)
        @test bar_count(store, "SYNTH") == 250
        @test length(all_bars(store)) == 250
        cutoff = s.bars[101].timestamp
        @test length(history(store, "SYNTH"; as_of = cutoff)) ==
            length(bars_until(s, cutoff))
    end
end
