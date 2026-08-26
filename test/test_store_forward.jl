# A bounded forward read, and the quadratic allocation it removed.
#
# Labels need to look forward: a label at time t is defined by what happened after it, which is
# exactly the property a feature must not have. The only forward read available was one that
# returned everything from a moment to the end of history, so a caller wanting three bars got all
# of them, and building labels for ten thousand bars allocated eleven gigabytes.

sf_series(n; seed = 20260825) = generate_series(
    AR1Returns(phi = 0.35, annual_drift = 0.05);
    symbol = "SF", n_bars = n, seed = seed, start = Date(2016, 1, 4),
)

sf_engine(bars) = FeatureEngine(
    InMemoryBarStore(bars),
    FeatureSet(Feature[LogReturn(1), LogReturn(5), RealisedVolatility(20), Momentum(10)]),
)

"""
    reference_label(store, symbol; as_of, horizon_bars)

The label as the previous implementation computed it: every bar from here to the end of history,
then the first `horizon_bars` of them. Kept as the reference precisely because it is the thing
that was replaced.
"""
function reference_label(store, symbol; as_of, horizon_bars)
    entry = latest(store, symbol; as_of = as_of)
    entry === nothing && return nothing
    ahead = Bar[
        bar for bar in load_range(store, symbol, entry.timestamp)
            if bar.timestamp > entry.timestamp
    ]
    length(ahead) < horizon_bars && return nothing
    holding = view(ahead, 1:horizon_bars)
    exit_bar = holding[end]
    entry_price = entry.close
    return (
        realised_at = exit_bar.timestamp,
        forward_log_return = log(exit_bar.close / entry_price),
        max_adverse_excursion = min(
            0.0, minimum(log(bar.low / entry_price) for bar in holding),
        ),
        max_favourable_excursion = max(
            0.0, maximum(log(bar.high / entry_price) for bar in holding),
        ),
    )
end

@testset "a bounded forward read" begin
    bars = sf_series(200).bars
    store = InMemoryBarStore(bars)
    stamps = [bar.timestamp for bar in bars]

    @testset "it returns the next bars and no others" begin
        for index in (1, 50, 199)
            for count in (1, 3, 10)
                ahead = upcoming(store, "SF"; after = stamps[index], count = count)
                expected = min(count, length(bars) - index)
                @test length(ahead) == expected
                for (offset, bar) in enumerate(ahead)
                    @test bar === bars[index + offset]
                end
            end
        end
    end

    @testset "strictly after, so the bar being labelled is not part of what happens next" begin
        # The boundary. A bar landing exactly on the bound is the one the label is about.
        ahead = upcoming(store, "SF"; after = stamps[10], count = 1)
        @test only(ahead).timestamp > stamps[10]
        @test only(ahead) === bars[11]

        # A moment between two bars picks up the next one.
        between = stamps[10] + Hour(1)
        @test between < stamps[11]
        @test only(upcoming(store, "SF"; after = between, count = 1)) === bars[11]
    end

    @testset "the end of history returns what there is, not an error" begin
        @test isempty(upcoming(store, "SF"; after = last(stamps), count = 5))
        @test length(upcoming(store, "SF"; after = stamps[end - 2], count = 10)) == 2
        @test isempty(upcoming(store, "SF"; after = stamps[1], count = 0))
        @test isempty(upcoming(store, "UNKNOWN"; after = stamps[1], count = 5))
        @test_throws ArgumentError upcoming(store, "SF"; after = stamps[1], count = -1)
    end
end

@testset "labels are unchanged by the bounded read" begin
    @testset "every label matches the previous implementation exactly" begin
        # Not approximately. The optimisation is only worth having if it is invisible in the
        # output, so this compares against the code it replaced, bit for bit.
        bars = sf_series(1500).bars
        engine = sf_engine(bars)
        for horizon in (1, 5, 20)
            examples = build_training_set(engine, "SF"; horizon_bars = horizon)
            @test !isempty(examples)
            for example in examples
                reference = reference_label(
                    engine.store, "SF";
                    as_of = example.features.as_of, horizon_bars = horizon,
                )
                @test reference !== nothing
                @test example.label.realised_at === reference.realised_at
                @test example.label.forward_log_return === reference.forward_log_return
                @test example.label.max_adverse_excursion ===
                    reference.max_adverse_excursion
                @test example.label.max_favourable_excursion ===
                    reference.max_favourable_excursion
            end
        end
    end

    @testset "a label still sees the whole horizon, not just its endpoint" begin
        # The excursions read every bar in the holding window. A bounded read that fetched one
        # bar would leave the endpoint right and the excursions wrong.
        bars = sf_series(300).bars
        engine = sf_engine(bars)
        examples = build_training_set(engine, "SF"; horizon_bars = 10)
        moved = [
            example for example in examples
                if example.label.max_favourable_excursion >
                abs(example.label.forward_log_return)
        ]
        @test !isempty(moved)      # some window ranged further than it finished
    end

    @testset "a horizon longer than the remaining history yields no label" begin
        bars = sf_series(60).bars
        engine = sf_engine(bars)
        examples = build_training_set(engine, "SF"; horizon_bars = 5)
        @test all(
            example -> example.label.realised_at <= last(bars).timestamp, examples,
        )
        # The last few bars cannot be labelled, and are not.
        @test maximum(example.features.as_of for example in examples) <=
            bars[end - 5].timestamp
    end
end

@testset "building labels is linear in the number of bars" begin
    # The property that was violated. Time was already near linear; allocation was not, because
    # every row copied the entire remaining history to keep a handful of bars.
    engine_small = sf_engine(sf_series(1000).bars)
    engine_large = sf_engine(sf_series(4000).bars)

    build_training_set(engine_small, "SF"; horizon_bars = 1)      # compile first
    build_training_set(engine_large, "SF"; horizon_bars = 1)

    small = @allocated build_training_set(engine_small, "SF"; horizon_bars = 1)
    large = @allocated build_training_set(engine_large, "SF"; horizon_bars = 1)

    # Four times the bars must not cost dramatically more than four times the memory. Before
    # this change the same comparison was roughly sixteen.
    @test large < 8 * small
    @test small > 0
end
