const ENGINE_SERIES = generate_series(
    GaussianReturns(); symbol = "SYNTH", n_bars = 120, seed = 21, start = Date(2024, 1, 1),
)
const ENGINE_STORE = InMemoryBarStore(ENGINE_SERIES.bars)

engine() = FeatureEngine(ENGINE_STORE, FeatureSet([LogReturn(1), SimpleReturn(20)]))

@testset "feature engine" begin
    @testset "features_at" begin
        @testset "a warmed up moment produces every feature" begin
            vector = features_at(engine(), "SYNTH", ENGINE_SERIES.bars[51].timestamp)
            @test is_complete(vector)
            @test vector.symbol == "SYNTH"
            @test vector.as_of == ENGINE_SERIES.bars[51].timestamp
        end

        @testset "an early moment reports what is still warming up" begin
            vector = features_at(engine(), "SYNTH", ENGINE_SERIES.bars[6].timestamp)
            @test vector.missing_features == [:return_20]
            @test haskey(vector, :log_return_1)
        end

        @testset "an unknown symbol or an early moment produces an empty vector" begin
            @test features_at(engine(), "SYNTH", DateTime(2020, 1, 1)).n_bars == 0
            @test features_at(engine(), "UNKNOWN", ENGINE_SERIES.bars[51].timestamp).n_bars == 0
        end

        @testset "the window is capped at the set's warm-up" begin
            @test warmup_bars(engine()) == 21
            @test features_at(engine(), "SYNTH", ENGINE_SERIES.bars[101].timestamp).n_bars == 21
        end

        @testset "values match a hand computed reference" begin
            bars = load_range(ENGINE_STORE, "SYNTH")
            expected = bars[61].close / bars[41].close - 1
            @test require(features_at(engine(), "SYNTH", bars[61].timestamp), :return_20) ≈
                expected
        end
    end

    @testset "walk" begin
        @testset "it yields one vector per bar, oldest first" begin
            vectors = walk(engine(), "SYNTH")
            @test length(vectors) == length(ENGINE_SERIES.bars)
            stamps = [vector.as_of for vector in vectors]
            @test issorted(stamps)
            @test stamps == [bar.timestamp for bar in ENGINE_SERIES.bars]
        end

        @testset "complete_only drops the warm-up period" begin
            complete = walk(engine(), "SYNTH"; complete_only = true)
            @test length(complete) == length(ENGINE_SERIES.bars) - 20
            @test all(is_complete, complete)
        end

        @testset "the window can be bounded" begin
            start, stop = ENGINE_SERIES.bars[31].timestamp, ENGINE_SERIES.bars[41].timestamp
            vectors = walk(engine(), "SYNTH"; start = start, stop = stop)
            @test first(vectors).as_of == start
            @test last(vectors).as_of == stop
        end

        @testset "walking agrees with computing each moment directly" begin
            walked = Dict(vector.as_of => vector for vector in walk(engine(), "SYNTH"))
            for bar in ENGINE_SERIES.bars[1:19:end]
                direct = features_at(engine(), "SYNTH", bar.timestamp)
                @test walked[bar.timestamp].values == direct.values
            end
        end

        @testset "an unknown symbol walks to nothing" begin
            @test isempty(walk(engine(), "UNKNOWN"))
        end
    end

    @testset "first_complete_at" begin
        @testset "it names the earliest fully warmed moment" begin
            first_moment = first_complete_at(engine(), "SYNTH")
            @test first_moment == ENGINE_SERIES.bars[21].timestamp
            @test is_complete(features_at(engine(), "SYNTH", first_moment))
            @test !is_complete(
                features_at(engine(), "SYNTH", ENGINE_SERIES.bars[20].timestamp),
            )
        end

        @testset "a series shorter than the warm-up has no such moment" begin
            short = FeatureEngine(
                InMemoryBarStore(ENGINE_SERIES.bars[1:10]), FeatureSet([SimpleReturn(20)]),
            )
            @test first_complete_at(short, "SYNTH") === nothing
            @test first_complete_at(engine(), "UNKNOWN") === nothing
        end
    end

    @testset "staleness" begin
        @testset "a vector records both the question and the answer" begin
            moment = ENGINE_SERIES.bars[51].timestamp
            vector = features_at(engine(), "SYNTH", moment)
            @test vector.as_of == moment
            @test vector.data_as_of == moment
            @test staleness(vector) == Millisecond(0)
        end

        @testset "asking between bars reports how old the data is" begin
            last_bar = ENGINE_SERIES.bars[end].timestamp
            vector = features_at(engine(), "SYNTH", last_bar + Day(3))
            @test vector.data_as_of == last_bar
            @test staleness(vector) == Day(3)
            @test is_stale(vector, Day(1))
            @test !is_stale(vector, Day(5))
        end

        @testset "a vector with no data at all counts as stale" begin
            # Treating "nothing known" as fresh is how a system ends up trading on a prior.
            vector = features_at(engine(), "UNKNOWN", ENGINE_SERIES.bars[51].timestamp)
            @test staleness(vector) === nothing
            @test is_stale(vector, Day(365))
        end

        @testset "a vector cannot claim data from after it was asked" begin
            @test_throws ArgumentError FeatureVector(
                symbol = "SYNTH",
                as_of = ENGINE_SERIES.bars[11].timestamp,
                data_as_of = ENGINE_SERIES.bars[12].timestamp,
                n_bars = 1,
            )
            @test_throws ArgumentError FeatureVector(
                symbol = "SYNTH", as_of = ENGINE_SERIES.bars[11].timestamp, n_bars = 5,
            )
        end
    end

    @testset "isolation" begin
        @testset "one symbol's features do not depend on another's presence" begin
            other = generate_series(
                GaussianReturns(); symbol = "OTHER", n_bars = 120, seed = 99,
                start = Date(2024, 1, 1),
            )
            set = FeatureSet([SimpleReturn(20)])
            alone = FeatureEngine(InMemoryBarStore(ENGINE_SERIES.bars), set)
            together = FeatureEngine(
                InMemoryBarStore(vcat(ENGINE_SERIES.bars, other.bars)), set,
            )
            moment = ENGINE_SERIES.bars[81].timestamp
            @test features_at(alone, "SYNTH", moment).values ==
                features_at(together, "SYNTH", moment).values
        end

        @testset "adding later bars does not change an earlier vector" begin
            set = FeatureSet([SimpleReturn(20)])
            early = FeatureEngine(InMemoryBarStore(ENGINE_SERIES.bars[1:60]), set)
            full = FeatureEngine(InMemoryBarStore(ENGINE_SERIES.bars), set)
            moment = ENGINE_SERIES.bars[60].timestamp
            @test features_at(early, "SYNTH", moment).values ==
                features_at(full, "SYNTH", moment).values
        end

        @testset "a bar arriving between decision points is seen immediately" begin
            store = InMemoryBarStore(ENGINE_SERIES.bars[1:60])
            local_engine = FeatureEngine(store, FeatureSet([LogReturn(1)]))
            moment = ENGINE_SERIES.bars[61].timestamp
            @test features_at(local_engine, "SYNTH", moment).data_as_of ==
                ENGINE_SERIES.bars[60].timestamp
            upsert!(store, [ENGINE_SERIES.bars[61]])
            @test features_at(local_engine, "SYNTH", moment).data_as_of == moment
        end
    end
end
