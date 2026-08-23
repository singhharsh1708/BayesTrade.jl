const SRC_START = Date(2024, 1, 1)
const SRC_STOP = Date(2024, 3, 29)

mutable struct FlakySource <: BarSource
    remaining::Int
    calls::Int
    error::DataSourceError
    inner::SyntheticSource
end

FlakySource(failures::Int, error::DataSourceError = TransientSourceError("connection reset")) =
    FlakySource(failures, 0, error, SyntheticSource())

BayesTrade.source_name(::FlakySource) = "flaky"

function BayesTrade.fetch_bars(source::FlakySource, symbol::AbstractString; kwargs...)
    source.calls += 1
    if source.remaining > 0
        source.remaining -= 1
        throw(source.error)
    end
    return fetch_bars(source.inner, symbol; kwargs...)
end

struct RecordingSleep
    delays::Vector{Float64}
end
RecordingSleep() = RecordingSleep(Float64[])
(sleeper::RecordingSleep)(seconds::Real) = push!(sleeper.delays, Float64(seconds))

@testset "data sources" begin
    @testset "synthetic source" begin
        @testset "one bar per trading day, carrying the requested symbol" begin
            bars = fetch_bars(SyntheticSource(), "RELIANCE"; start = SRC_START, stop = SRC_STOP)
            @test length(bars) == 65
            @test Date(first(bars).timestamp) == SRC_START
            @test all(bar -> dayofweek(bar.timestamp) <= 5, bars)
            @test Set(bar.symbol for bar in bars) == Set(["RELIANCE"])
        end

        @testset "a window with no trading days is empty" begin
            @test isempty(
                fetch_bars(
                    SyntheticSource(), "RELIANCE";
                    start = Date(2024, 1, 6), stop = Date(2024, 1, 7),
                ),
            )
        end

        @testset "invalid windows and intervals are rejected" begin
            source = SyntheticSource()
            @test_throws ArgumentError fetch_bars(
                source, "RELIANCE"; start = SRC_STOP, stop = SRC_START,
            )
            @test_throws ArgumentError fetch_bars(
                source, "RELIANCE"; start = SRC_START, stop = SRC_STOP, interval = "5m",
            )
            @test_throws ArgumentError fetch_bars(
                source, "RELIANCE"; start = Date(1900, 1, 1), stop = Date(2026, 1, 1),
            )
        end

        @testset "each symbol follows its own reproducible path" begin
            source = SyntheticSource()
            reliance = fetch_bars(source, "RELIANCE"; start = SRC_START, stop = SRC_STOP)
            tcs = fetch_bars(source, "TCS"; start = SRC_START, stop = SRC_STOP)
            again = fetch_bars(source, "RELIANCE"; start = SRC_START, stop = SRC_STOP)
            @test [b.close for b in reliance] != [b.close for b in tcs]
            @test [b.close for b in reliance] == [b.close for b in again]
        end

        @testset "the seed derivation does not depend on session salting" begin
            @test seed_for(SyntheticSource(seed = 7), "RELIANCE") ==
                seed_for(SyntheticSource(seed = 7), "RELIANCE")
            @test seed_for(SyntheticSource(seed = 7), "RELIANCE") !=
                seed_for(SyntheticSource(seed = 8), "RELIANCE")
        end

        @testset "variation can be switched off for a controlled experiment" begin
            source = SyntheticSource(vary_by_symbol = false, seed = 11)
            @test seed_for(source, "RELIANCE") == seed_for(source, "TCS") == 11
            reliance = fetch_bars(source, "RELIANCE"; start = SRC_START, stop = SRC_STOP)
            tcs = fetch_bars(source, "TCS"; start = SRC_START, stop = SRC_STOP)
            @test [b.close for b in reliance] == [b.close for b in tcs]
        end

        @testset "the process is configurable" begin
            source = SyntheticSource(process = AR1Returns(phi = 0.4), seed = 3)
            @test length(fetch_bars(source, "R"; start = SRC_START, stop = SRC_STOP)) == 65
        end
    end

    @testset "retrying source" begin
        @testset "a transient failure is retried and succeeds" begin
            inner = FlakySource(2)
            source = RetryingSource(
                inner; attempts = 3, backoff_seconds = 1.0, sleeper = RecordingSleep(),
            )
            @test !isempty(fetch_bars(source, "R"; start = SRC_START, stop = SRC_STOP))
            @test inner.calls == 3
        end

        @testset "backoff doubles and is capped" begin
            source = RetryingSource(
                FlakySource(0); backoff_seconds = 2.0, max_backoff_seconds = 5.0,
            )
            @test [delay_for(source, attempt) for attempt in 0:3] == [2.0, 4.0, 5.0, 5.0]
        end

        @testset "it waits between attempts but not after the last" begin
            sleeper = RecordingSleep()
            source = RetryingSource(
                FlakySource(2); attempts = 3, backoff_seconds = 1.0, sleeper = sleeper,
            )
            fetch_bars(source, "R"; start = SRC_START, stop = SRC_STOP)
            @test sleeper.delays == [1.0, 2.0]
        end

        @testset "exhausting the attempts reports how many were made" begin
            inner = FlakySource(99)
            source = RetryingSource(
                inner; attempts = 3, backoff_seconds = 0.0, sleeper = RecordingSleep(),
            )
            @test_throws TransientSourceError fetch_bars(
                source, "R"; start = SRC_START, stop = SRC_STOP,
            )
            @test inner.calls == 3
        end

        @testset "an unknown symbol is not retried" begin
            # It will still be unknown on the third attempt, and burning the budget on it
            # delays every symbol behind it in the run.
            inner = FlakySource(99, SymbolNotFoundError("no such ticker"))
            source = RetryingSource(inner; attempts = 5, sleeper = RecordingSleep())
            @test_throws SymbolNotFoundError fetch_bars(
                source, "NOPE"; start = SRC_START, stop = SRC_STOP,
            )
            @test inner.calls == 1
        end

        @testset "a healthy source is called once and never sleeps" begin
            inner = FlakySource(0)
            sleeper = RecordingSleep()
            RetryingSource(inner; sleeper = sleeper) |>
                source -> fetch_bars(source, "R"; start = SRC_START, stop = SRC_STOP)
            @test inner.calls == 1
            @test isempty(sleeper.delays)
        end

        @testset "the wrapper reports the vendor, not itself" begin
            source = RetryingSource(SyntheticSource())
            @test source_name(source) == "synthetic"
            @test supported_intervals(source) == supported_intervals(SyntheticSource())
        end

        @testset "a zero attempt budget is rejected" begin
            @test_throws ArgumentError RetryingSource(SyntheticSource(); attempts = 0)
        end
    end

    @testset "ingestion" begin
        @testset "it writes every symbol into the store" begin
            store = InMemoryBarStore()
            report = ingest!(
                SyntheticSource(), store, ["RELIANCE", "TCS", "INFY"];
                start = SRC_START, stop = SRC_STOP,
            )
            @test is_complete(report)
            @test succeeded(report) == ["INFY", "RELIANCE", "TCS"]
            @test symbols(store) == ["INFY", "RELIANCE", "TCS"]
            @test total_written(report) == sum(bar_count(store, s) for s in symbols(store))
            @test occursin("synthetic", summarise(report))
            @test !occursin("failed", summarise(report))
        end

        @testset "one bad symbol does not abort the run" begin
            store = InMemoryBarStore()
            report = ingest!(
                FlakySource(0, SymbolNotFoundError("delisted")), store, ["RELIANCE"];
                start = SRC_START, stop = SRC_STOP,
            )
            @test is_complete(report)

            broken = FlakySource(99, SymbolNotFoundError("delisted on 2023-11-02"))
            report = ingest!(
                broken, store, ["DELISTED"]; start = SRC_START, stop = SRC_STOP,
            )
            @test !is_complete(report)
            @test failed(report) == ["DELISTED"]
            @test occursin("delisted on 2023-11-02", report.failures["DELISTED"])
            @test occursin("failed:", summarise(report))
        end

        @testset "an empty vendor response is recorded as a failure" begin
            report = ingest!(
                SyntheticSource(), InMemoryBarStore(), ["RELIANCE"];
                start = Date(2024, 1, 6), stop = Date(2024, 1, 6),
            )
            @test report.failures["RELIANCE"] == "vendor returned no bars"
        end

        @testset "an unsupported interval is recorded per symbol" begin
            report = ingest!(
                SyntheticSource(), InMemoryBarStore(), ["RELIANCE"];
                start = SRC_START, stop = SRC_STOP, interval = "5m",
            )
            @test occursin("does not provide", report.failures["RELIANCE"])
        end

        @testset "re-ingesting the same window does not duplicate bars" begin
            store = InMemoryBarStore()
            source = SyntheticSource()
            ingest!(source, store, ["RELIANCE"]; start = SRC_START, stop = SRC_STOP)
            before = bar_count(store, "RELIANCE")
            ingest!(source, store, ["RELIANCE"]; start = SRC_START, stop = SRC_STOP)
            @test bar_count(store, "RELIANCE") == before
        end
    end

    @testset "an incomplete source says what it is missing" begin
        struct BareSource <: BarSource end
        @test_throws ArgumentError source_name(BareSource())
        @test_throws ArgumentError fetch_bars(BareSource(), "R")
    end
end
