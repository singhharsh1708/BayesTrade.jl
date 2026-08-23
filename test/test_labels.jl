const LABEL_SERIES = generate_series(
    GaussianReturns(); symbol = "SYNTH", n_bars = 200, seed = 41, start = Date(2024, 1, 1),
)
const LABEL_STORE = InMemoryBarStore(LABEL_SERIES.bars)
const HORIZON = 5

label_engine() = FeatureEngine(LABEL_STORE, minimal_feature_set())

@testset "forward labels" begin
    @testset "the return is measured from the decision close" begin
        index = 100
        label = forward_label(
            LABEL_STORE, "SYNTH";
            as_of = LABEL_SERIES.bars[index].timestamp, horizon_bars = HORIZON,
        )
        @test label !== nothing
        @test label.forward_log_return ≈
            log(LABEL_SERIES.bars[index + HORIZON].close / LABEL_SERIES.bars[index].close)
        @test label.realised_at == LABEL_SERIES.bars[index + HORIZON].timestamp
    end

    @testset "it is realised strictly after the decision" begin
        for index in 1:17:150
            label = forward_label(
                LABEL_STORE, "SYNTH";
                as_of = LABEL_SERIES.bars[index].timestamp, horizon_bars = HORIZON,
            )
            @test label !== nothing
            @test label.realised_at > label.as_of
        end
    end

    @testset "an unrealised horizon returns nothing rather than a short one" begin
        for offset in 0:(HORIZON - 1)
            near_end = LABEL_SERIES.bars[end - offset].timestamp
            @test forward_label(
                LABEL_STORE, "SYNTH"; as_of = near_end, horizon_bars = HORIZON,
            ) === nothing
        end
    end

    @testset "the final realisable decision is exactly horizon bars from the end" begin
        last_usable = LABEL_SERIES.bars[end - HORIZON].timestamp
        @test forward_label(
            LABEL_STORE, "SYNTH"; as_of = last_usable, horizon_bars = HORIZON,
        ) !== nothing
    end

    @testset "excursions use the intraday extremes" begin
        # A stop is hit by the path, not the close.
        index = 80
        label = forward_label(
            LABEL_STORE, "SYNTH";
            as_of = LABEL_SERIES.bars[index].timestamp, horizon_bars = HORIZON,
        )
        entry = LABEL_SERIES.bars[index].close
        holding = LABEL_SERIES.bars[(index + 1):(index + HORIZON)]
        @test label.max_adverse_excursion ≈
            min(0.0, minimum(log(bar.low / entry) for bar in holding))
        @test label.max_favourable_excursion ≈
            max(0.0, maximum(log(bar.high / entry) for bar in holding))
    end

    @testset "the excursions bracket the return" begin
        # A trade cannot end better than its best moment or worse than its worst.
        for index in 1:11:150
            label = forward_label(
                LABEL_STORE, "SYNTH";
                as_of = LABEL_SERIES.bars[index].timestamp, horizon_bars = HORIZON,
            )
            @test label.max_adverse_excursion <= label.forward_log_return
            @test label.forward_log_return <= label.max_favourable_excursion
        end
    end

    @testset "an unknown symbol has no label" begin
        @test forward_label(
            LABEL_STORE, "UNKNOWN";
            as_of = LABEL_SERIES.bars[11].timestamp, horizon_bars = 1,
        ) === nothing
    end

    @testset "a non-positive horizon is rejected" begin
        @test_throws ArgumentError forward_label(
            LABEL_STORE, "SYNTH";
            as_of = LABEL_SERIES.bars[11].timestamp, horizon_bars = 0,
        )
    end
end

@testset "label invariants" begin
    basis(; kwargs...) = Label(;
        symbol = "SYNTH",
        as_of = LABEL_SERIES.bars[11].timestamp,
        realised_at = LABEL_SERIES.bars[16].timestamp,
        horizon_bars = 5,
        forward_log_return = 0.02,
        max_adverse_excursion = -0.01,
        max_favourable_excursion = 0.03,
        kwargs...,
    )

    @testset "a label realised at or before its decision is rejected" begin
        @test_throws ArgumentError basis(realised_at = LABEL_SERIES.bars[6].timestamp)
        @test_throws ArgumentError basis(realised_at = LABEL_SERIES.bars[11].timestamp)
    end

    @testset "excursion signs are enforced" begin
        @test_throws ArgumentError basis(max_adverse_excursion = 0.01)
        @test_throws ArgumentError basis(max_favourable_excursion = -0.01)
    end

    @testset "a non-positive horizon is rejected" begin
        @test_throws ArgumentError basis(horizon_bars = 0)
    end

    @testset "mismatched pairing is rejected" begin
        features = features_at(label_engine(), "SYNTH", LABEL_SERIES.bars[21].timestamp)
        @test_throws ArgumentError TrainingExample(features, basis())
        matching = features_at(label_engine(), "SYNTH", LABEL_SERIES.bars[11].timestamp)
        @test_throws ArgumentError TrainingExample(matching, basis(symbol = "OTHER"))
    end

    @testset "a positive label is recognised" begin
        @test is_positive(basis())
        @test !is_positive(basis(forward_log_return = -0.02))
    end
end

@testset "training sets" begin
    @testset "every row pairs matching moments" begin
        for example in build_training_set(label_engine(), "SYNTH"; horizon_bars = HORIZON)
            @test example.features.as_of == example.label.as_of
            @test example.label.realised_at > example.features.as_of
        end
    end

    @testset "the unrealised tail is dropped" begin
        examples = build_training_set(label_engine(), "SYNTH"; horizon_bars = HORIZON)
        warmup = warmup_bars(label_engine()) - 1
        @test length(examples) == length(LABEL_SERIES.bars) - warmup - HORIZON
        @test last(examples).label.realised_at == last(LABEL_SERIES.bars).timestamp
    end

    @testset "a longer horizon drops more of the tail" begin
        short = build_training_set(label_engine(), "SYNTH"; horizon_bars = 1)
        long = build_training_set(label_engine(), "SYNTH"; horizon_bars = 20)
        @test length(short) - length(long) == 19
    end

    @testset "incomplete feature rows are excluded by default" begin
        examples = build_training_set(label_engine(), "SYNTH"; horizon_bars = HORIZON)
        @test all(example -> is_complete(example.features), examples)
    end

    @testset "warm-up rows can be kept deliberately" begin
        kept = build_training_set(
            label_engine(), "SYNTH"; horizon_bars = HORIZON, complete_only = false,
        )
        @test length(kept) >
            length(build_training_set(label_engine(), "SYNTH"; horizon_bars = HORIZON))
    end

    @testset "the window can be bounded" begin
        start = LABEL_SERIES.bars[51].timestamp
        stop = LABEL_SERIES.bars[101].timestamp
        examples = build_training_set(
            label_engine(), "SYNTH"; horizon_bars = HORIZON, start = start, stop = stop,
        )
        @test first(examples).features.as_of >= start
        @test last(examples).features.as_of <= stop
    end
end

@testset "presets" begin
    @testset "the minimal set is three features warming up in a quarter" begin
        set = minimal_feature_set()
        @test length(set) == 3
        @test columns(set) == [:log_return_1, :momentum_20_1, :volatility_20]
        @test required_bars(set) <= 65
    end

    @testset "the default set warms up in about a trading year" begin
        set = default_feature_set()
        @test length(set) == 21
        @test 120 <= required_bars(set) <= 260
        @test length(unique(columns(set))) == 21
    end

    @testset "the default set produces every value once warmed up" begin
        series = generate_series(
            GaussianReturns(); symbol = "SYNTH", n_bars = 400, seed = 13,
            start = Date(2022, 1, 3),
        )
        engine = FeatureEngine(InMemoryBarStore(series.bars), default_feature_set())
        vector = features_at(engine, "SYNTH", last(series.bars).timestamp)
        @test is_complete(vector)
        @test length(vector.values) == 21
        @test all(isfinite, values(vector.values))
    end
end
