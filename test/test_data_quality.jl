qseries(n = 300; seed = 5) = generate_series(
    GaussianReturns(annual_volatility = 0.25); symbol = "SYNTH", n_bars = n,
    seed = seed, start = Date(2024, 1, 1),
).bars

flatbars(n = 60; close = 100.0, offset = 0) = Bar[
    Bar("SYNTH", DateTime(2024, 1, 1, 10, 0) + Day(offset + i), close, close, close, close, 1000.0)
        for i in 0:(n - 1)
]

checks(bars; kwargs...) = Set(issue.check for issue in validate_bars(bars; kwargs...).issues)

rescale(bar, factor) = Bar(
    bar.symbol, bar.timestamp, bar.open * factor, bar.high * factor,
    bar.low * factor, bar.close * factor, bar.volume; interval = bar.interval,
)

@testset "data quality" begin
    @testset "a generated series raises nothing" begin
        report = validate_bars(qseries())
        @test isempty(report.issues)
        @test is_usable(report)
        @test report.n_bars == 300
        @test !occursin('\n', summarise(report))
    end

    @testset "an empty series is reported as empty rather than crashing" begin
        report = validate_bars(Bar[])
        @test report.n_bars == 0
        @test report.first === nothing
    end

    @testset "a mixed symbol sequence is a programming error" begin
        bars = qseries(50)
        other = Bar("OTHER", bars[1].timestamp, 100.0, 100.0, 100.0, 100.0, 1.0)
        @test_throws ArgumentError validate_bars(vcat(bars, [other]))
    end

    @testset "duplicate timestamps are an error" begin
        bars = qseries(50)
        report = validate_bars(vcat(bars, [bars[11]]))
        @test :duplicate_timestamp in Set(issue.check for issue in report.issues)
        @test !is_usable(report)
    end

    @testset "out of order input is noted but not fatal" begin
        report = validate_bars(reverse(qseries(50)))
        @test [issue.check for issue in of_severity(report, INFO)] == [:unordered_series]
        @test is_usable(report)
    end

    @testset "too little history to fit anything is an error" begin
        @test :insufficient_history in Set(issue.check for issue in errors(validate_bars(qseries(10))))
        @test is_usable(validate_bars(qseries(10); min_bars = 5))
    end

    @testset "a long hole in the download is flagged" begin
        bars = qseries(100)
        report = validate_bars(vcat(bars[1:40], bars[61:end]))
        gaps = [issue for issue in report.issues if issue.check === :calendar_gap]
        @test length(gaps) == 1
        @test first(gaps).severity === WARNING
        @test first(gaps).observed >= 19
    end

    @testset "an ordinary holiday length gap is not flagged" begin
        bars = qseries(100)
        @test !(:calendar_gap in checks(vcat(bars[1:40], bars[43:end])))
        @test :calendar_gap in checks(vcat(bars[1:40], bars[43:end]); max_gap_bars = 0)
    end

    @testset "zero volume is a warning, and an error when it is most of the series" begin
        bars = qseries(100)
        one_bad = copy(bars)
        one_bad[11] = Bar(
            "SYNTH", bars[11].timestamp, bars[11].open, bars[11].high,
            bars[11].low, bars[11].close, 0.0,
        )
        report = validate_bars(one_bad)
        @test first(i for i in report.issues if i.check === :zero_volume).severity === WARNING
        @test is_usable(report)

        many_bad = copy(bars)
        for index in 1:20
            b = bars[index]
            many_bad[index] = Bar("SYNTH", b.timestamp, b.open, b.high, b.low, b.close, 0.0)
        end
        @test :zero_volume in Set(issue.check for issue in errors(validate_bars(many_bad)))
    end

    @testset "a stopped feed is flagged" begin
        report = validate_bars(flatbars(40))
        stale = [issue for issue in report.issues if issue.check === :stale_price]
        @test !isempty(stale)
        @test first(stale).observed == 40.0
    end

    @testset "a stale run at the end of the series is still found" begin
        bars = qseries(100)
        tail = flatbars(10; close = last(bars).close, offset = 200)
        @test :stale_price in checks(vcat(bars, tail))
    end

    @testset "an unadjusted split is named as one" begin
        bars = qseries(300)
        halved = vcat(bars[1:150], [rescale(bar, 0.5) for bar in bars[151:end]])
        report = validate_bars(halved)
        splits = [issue for issue in report.issues if issue.check === :suspected_split]
        @test length(splits) == 1
        @test first(splits).severity === ERROR
        @test first(splits).observed ≈ 2.0
        @test !is_usable(report)
    end

    @testset "a bonus issue ratio is also recognised" begin
        bars = qseries(300)
        split = vcat(bars[1:150], [rescale(bar, 0.2) for bar in bars[151:end]])
        splits = [i for i in validate_bars(split).issues if i.check === :suspected_split]
        @test first(splits).observed ≈ 5.0
    end

    @testset "a large move that is not a split ratio is only a warning" begin
        bars = qseries(300)
        spiked = copy(bars)
        spiked[151] = rescale(bars[151], exp(-0.5))
        report = validate_bars(spiked)
        spikes = [issue for issue in report.issues if issue.check === :price_spike]
        @test !isempty(spikes)
        @test all(issue.severity === WARNING for issue in spikes)
    end

    @testset "the robust scale catches a split a classical one would miss" begin
        # One outlier inflates a standard deviation enough to hide itself. On a short
        # series the split contributes most of the classical variance, so its own z-score
        # falls under the threshold. A median absolute deviation is unmoved by it.
        bars = qseries(60)
        halved = vcat(bars[1:30], [rescale(bar, 0.5) for bar in bars[31:end]])
        returns = diff(log.([bar.close for bar in halved]))
        classical_z = abs(minimum(returns) - mean(returns)) / std(returns)
        @test classical_z < 12.0
        @test :suspected_split in checks(halved)
    end

    @testset "degenerate series do not divide by zero" begin
        found = checks(flatbars(60))
        @test !(:price_spike in found)
        @test !(:suspected_split in found)
        @test !(:price_spike in checks(qseries(500)))
    end

    @testset "split ratios are matched in log space" begin
        @test split_ratio(-log(2.0)) ≈ 2.0
        @test split_ratio(-log(2.04)) ≈ 2.0   # a split plus that day's own return
        @test split_ratio(log(5.0)) ≈ 5.0     # a reverse split
        @test split_ratio(-0.5) === nothing
        @test split_ratio(-0.01) === nothing
    end

    @testset "the summary lists every finding" begin
        summary = summarise(validate_bars(flatbars(40)))
        @test occursin("SYNTH: 40 bars", summary)
        @test occursin("stale_price", summary)
    end
end
