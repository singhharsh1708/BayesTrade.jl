@testset "csv persistence" begin
    csvbar(; symbol = "RELIANCE", offset = 0, close = 100.0) = Bar(
        symbol, DateTime(2026, 1, 5, 10, 0) + Day(offset),
        close, close + 1, close - 1, close, 1234.5,
    )

    @testset "bars survive a write and read unchanged" begin
        mktempdir() do dir
            original = [csvbar(offset = i, close = 100.0 + i) for i in 0:19]
            path = joinpath(dir, "reliance.csv")
            @test write_bars(path, original) == 20
            restored = read_bars(path)
            @test length(restored) == 20
            @test all(
                getfield(a, f) == getfield(b, f)
                    for (a, b) in zip(original, restored) for f in fieldnames(Bar)
            )
        end
    end

    @testset "rows are written oldest first regardless of input order" begin
        mktempdir() do dir
            path = joinpath(dir, "bars.csv")
            write_bars(path, [csvbar(offset = 3), csvbar(offset = 1), csvbar(offset = 2)])
            @test issorted([bar.timestamp for bar in read_bars(path)])
        end
    end

    @testset "a generated series survives the round trip" begin
        mktempdir() do dir
            s = generate_series(
                GaussianReturns(); symbol = "SYNTH", n_bars = 500, seed = 2,
                start = Date(2024, 1, 1),
            )
            path = joinpath(dir, "synth.csv")
            write_bars(path, s.bars)
            restored = read_bars(path)
            @test [bar.close for bar in restored] == [bar.close for bar in s.bars]
            @test [bar.timestamp for bar in restored] == [bar.timestamp for bar in s.bars]
        end
    end

    @testset "missing parent directories are created" begin
        mktempdir() do dir
            path = joinpath(dir, "nested", "deeper", "bars.csv")
            write_bars(path, [csvbar()])
            @test isfile(path)
        end
    end

    @testset "malformed files fail at load, not three layers later" begin
        mktempdir() do dir
            @test_throws BarFileError read_bars(joinpath(dir, "absent.csv"))

            empty_path = joinpath(dir, "empty.csv")
            write(empty_path, "")
            @test_throws BarFileError read_bars(empty_path)

            partial = joinpath(dir, "partial.csv")
            write(partial, "symbol,timestamp,close\nRELIANCE,2026-01-05T10:00:00,100\n")
            @test_throws BarFileError read_bars(partial)

            invalid = joinpath(dir, "invalid.csv")
            write_bars(invalid, [csvbar(offset = 0), csvbar(offset = 1)])
            lines = readlines(invalid)
            lines[3] = replace(lines[3], "1234.5" => "not-a-number")
            write(invalid, join(lines, '\n') * '\n')
            @test_throws BarFileError read_bars(invalid)
        end
    end

    @testset "a row violating OHLC ordering is rejected on read" begin
        mktempdir() do dir
            path = joinpath(dir, "bad.csv")
            write(
                path,
                "symbol,timestamp,open,high,low,close,volume,interval\n" *
                    "RELIANCE,2026-01-05T10:00:00,100,99,101,100,1000,1d\n",
            )
            @test_throws BarFileError read_bars(path)
        end
    end

    @testset "a store survives a save and load" begin
        mktempdir() do dir
            store = InMemoryBarStore(
                vcat(
                    [csvbar(offset = i) for i in 0:9],
                    [csvbar(symbol = "TCS", offset = i) for i in 0:4],
                ),
            )
            @test save_store(store, dir) == Dict("RELIANCE" => 10, "TCS" => 5)
            restored = load_store(dir)
            @test symbols(restored) == symbols(store)
            @test bar_count(restored, "RELIANCE") == 10
            @test sort(filter(endswith(".csv"), readdir(dir))) == ["RELIANCE.csv", "TCS.csv"]
        end
    end

    @testset "symbols with path characters stay loadable" begin
        mktempdir() do dir
            save_store(InMemoryBarStore([csvbar(symbol = "NSE:RELIANCE")]), dir)
            @test symbols(load_store(dir)) == ["NSE:RELIANCE"]
        end
    end

    @testset "an empty store writes nothing" begin
        mktempdir() do dir
            @test isempty(save_store(InMemoryBarStore(), dir))
            @test isempty(symbols(load_store(dir)))
            @test_throws BarFileError load_store(joinpath(dir, "absent"))
        end
    end
end
