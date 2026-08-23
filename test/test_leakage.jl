# Look-ahead tests. These must never be skipped.
#
# A leak does not announce itself. It shows up as a backtest that looks good, and the whole
# system is worth nothing if these pass by accident. Each test asserts a property that a
# realistic mistake would violate, not merely that a function returns something.
#
# This file grows with every layer that touches time: storage now, features next, then the
# backtester itself.

const LEAK_SERIES = generate_series(
    GaussianReturns(); symbol = "SYNTH", n_bars = 400, seed = 17, start = Date(2024, 1, 1),
)
const LEAK_STORE = InMemoryBarStore(LEAK_SERIES.bars)

@testset "look-ahead" begin
    @testset "no query ever returns a bar from the future" begin
        for bar in LEAK_SERIES.bars[1:17:end]
            visible = history(LEAK_STORE, "SYNTH"; as_of = bar.timestamp)
            @test last(visible).timestamp == bar.timestamp
            @test all(seen -> seen.timestamp <= bar.timestamp, visible)
        end
    end

    @testset "a rolling window never reaches forward" begin
        for bar in LEAK_SERIES.bars[51:37:end]
            window = history(LEAK_STORE, "SYNTH"; as_of = bar.timestamp, count = 20)
            @test length(window) == 20
            @test last(window).timestamp == bar.timestamp
        end
    end

    @testset "the bar being decided on is the last one visible" begin
        for index in 2:29:(length(LEAK_SERIES.bars) - 1)
            bar = LEAK_SERIES.bars[index]
            found = latest(LEAK_STORE, "SYNTH"; as_of = bar.timestamp)
            @test found !== nothing
            @test found.timestamp == bar.timestamp
            @test found.timestamp != LEAK_SERIES.bars[index + 1].timestamp
        end
    end

    @testset "marking a portfolio never uses tomorrow's price" begin
        for index in 1:41:(length(LEAK_SERIES.bars) - 1)
            as_of = LEAK_SERIES.bars[index].timestamp
            marks = align(LEAK_STORE, ["SYNTH"]; as_of = as_of)
            @test marks["SYNTH"].close == LEAK_SERIES.bars[index].close
        end
    end

    @testset "history is monotone in the observation time" begin
        # Later moments can only ever see more, never different.
        earlier = history(LEAK_STORE, "SYNTH"; as_of = LEAK_SERIES.bars[100].timestamp)
        later = history(LEAK_STORE, "SYNTH"; as_of = LEAK_SERIES.bars[200].timestamp)
        @test [bar.timestamp for bar in later[1:length(earlier)]] ==
            [bar.timestamp for bar in earlier]
    end

    @testset "a revised bar does not change what was visible before it" begin
        cutoff = LEAK_SERIES.bars[100].timestamp
        before = history(LEAK_STORE, "SYNTH"; as_of = cutoff)
        revised = InMemoryBarStore(LEAK_SERIES.bars)
        later = LEAK_SERIES.bars[300]
        upsert!(
            revised,
            [
                Bar(
                    later.symbol, later.timestamp, later.open, later.high * 3,
                    later.low, later.close, later.volume,
                ),
            ],
        )
        @test [bar.close for bar in history(revised, "SYNTH"; as_of = cutoff)] ==
            [bar.close for bar in before]
    end

    @testset "reported information stays invisible until it is reported" begin
        snapshot = FundamentalSnapshot(
            symbol = "RELIANCE",
            period_end = Date(2026, 3, 31),
            reported_at = DateTime(2026, 4, 25, 18),
            roe = 0.18,
        )
        @test !is_known_at(snapshot, DateTime(2026, 3, 31, 23, 59))
        @test !is_known_at(snapshot, snapshot.reported_at - Second(1))
        @test is_known_at(snapshot, snapshot.reported_at)

        published = DateTime(2026, 1, 2, 9, 15)
        item = NewsItem(
            symbol = "RELIANCE", published_at = published, headline = "Results beat",
        )
        @test !is_known_at(item, published - Second(1))
        @test is_known_at(item, published)
    end
end
