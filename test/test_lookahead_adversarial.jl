# An independent look-ahead audit, run against the assumption that the previous fixes were
# sufficient rather than on top of it.
#
# The construction throughout is the same one, and it is the only one that actually proves
# anything: compute a quantity at time t with the whole future present, then mutate that future
# beyond recognition and compute it again. Nothing about time t may move. A test that only feeds
# the pipeline data up to t proves nothing, because there is no future there to leak.

"""
    poison(series, from; factor)

The same series with everything after `from` replaced by something wildly different.

Prices are scaled rather than randomised so the bars stay valid: an OHLC that no longer
satisfies low <= open, close <= high would be rejected by the constructor and the test would
pass for the wrong reason. The scale is large enough that any leak moves a number visibly.
"""
function poison(series::SyntheticSeries, from::Int; factor::Real = 4.0)
    bars = copy(series.bars)
    for index in from:length(bars)
        bar = bars[index]
        # Reverse the direction as well as the level: a leak that only reads magnitude and a
        # leak that only reads sign are different bugs and both have to move something.
        flip = isodd(index) ? factor : 1 / factor
        bars[index] = Bar(
            bar.symbol, bar.timestamp,
            bar.open * flip, bar.high * flip, bar.low * flip, bar.close * flip,
            bar.volume * 7 + 1000;
            interval = bar.interval,
        )
    end
    return SyntheticSeries(
        series.symbol, bars, series.path, series.parameters, series.seed,
    )
end

la_series(; n_bars = 700, seed = 4242) = generate_series(
    AR1Returns(phi = 0.4, annual_drift = 0.05);
    symbol = "LEAK", n_bars = n_bars, seed = seed, start = Date(2022, 1, 3),
)

la_features() = FeatureSet(
    Feature[
        LogReturn(1), LogReturn(5), RealisedVolatility(20), Momentum(10),
        PriceZScore(20), DrawdownFromHigh(30),
    ],
)

la_engine(series) = FeatureEngine(InMemoryBarStore(series.bars), la_features())

la_models() = (
    () -> BayesianReturnModel([:log_return_1, :momentum_10_5]; horizon_bars = 1),
    () -> BayesianVolatilityModel(; horizon_bars = 1),
    () -> MarketRegimeModel(; horizon_bars = 1),
)

@testset "adversarial look-ahead" begin
    series = la_series()
    cut = 450                                   # everything from here on gets poisoned
    boundary = series.bars[cut].timestamp
    poisoned = poison(series, cut)

    @testset "the poison is drastic enough to notice" begin
        # If mutating the future did not change anything downstream, every invariance below
        # would hold vacuously and the whole file would be theatre.
        @test series.bars[1:(cut - 1)] == poisoned.bars[1:(cut - 1)]
        @test series.bars[cut].close != poisoned.bars[cut].close
        clean = features_at(la_engine(series), "LEAK", last(series.bars).timestamp)
        dirty = features_at(la_engine(poisoned), "LEAK", last(poisoned.bars).timestamp)
        @test clean.values != dirty.values      # after the boundary, everything moves
    end

    @testset "a feature vector cannot see past its own timestamp" begin
        clean_engine = la_engine(series)
        dirty_engine = la_engine(poisoned)
        checked = 0
        for index in 60:(cut - 1)
            as_of = series.bars[index].timestamp
            clean = features_at(clean_engine, "LEAK", as_of)
            dirty = features_at(dirty_engine, "LEAK", as_of)
            @test clean.as_of == dirty.as_of
            @test clean.n_bars == dirty.n_bars
            @test keys(clean.values) == keys(dirty.values)
            for name in keys(clean.values)
                # Bit for bit. A feature that reads one future bar and averages it away is
                # still a feature that reads a future bar.
                @test clean.values[name] === dirty.values[name]
            end
            checked += 1
        end
        @test checked > 380
    end

    @testset "walking the whole history is the same walk" begin
        # Stops one bar short of the boundary: the bar at the boundary is itself poisoned, so
        # its own features are supposed to differ.
        last_clean = series.bars[cut - 1].timestamp
        clean = walk(la_engine(series), "LEAK"; stop = last_clean)
        dirty = walk(la_engine(poisoned), "LEAK"; stop = last_clean)
        @test length(clean) == length(dirty)
        @test !isempty(clean)
        for (left, right) in zip(clean, dirty)
            @test left.as_of == right.as_of
            @test left.values == right.values
        end
    end

    @testset "a training row's features do not depend on rows after it" begin
        clean = build_training_set(la_engine(series), "LEAK"; horizon_bars = 1)
        dirty = build_training_set(la_engine(poisoned), "LEAK"; horizon_bars = 1)
        before = [row for row in clean if row.label.realised_at < boundary]
        @test length(before) > 300
        by_time = Dict(row.features.as_of => row for row in dirty)
        for row in before
            other = by_time[row.features.as_of]
            @test row.features.values == other.features.values
            # The label is allowed to differ only once its own realisation lands in the
            # poisoned stretch, which is what makes the cutoff above the right one.
            @test row.label.forward_log_return === other.label.forward_log_return
            @test row.label.realised_at == other.label.realised_at
        end
    end

    @testset "a walk-forward prediction does not move when its future is destroyed" begin
        # The strongest statement in the file. Same model, same config, same index, and the
        # only difference is history the model must not have been able to reach.
        config = WalkForwardConfig(initial_train = 200, refit_every = 20)
        clean = walk_forward(
            () -> BayesianReturnModel([:log_return_1, :momentum_10_5]; horizon_bars = 1),
            build_training_set(la_engine(series), "LEAK"; horizon_bars = 1), config,
        )
        dirty = walk_forward(
            () -> BayesianReturnModel([:log_return_1, :momentum_10_5]; horizon_bars = 1),
            build_training_set(la_engine(poisoned), "LEAK"; horizon_bars = 1), config,
        )
        paired = Dict(record.as_of => record for record in dirty)
        compared = 0
        for record in clean
            record.realised_at < boundary || continue
            other = paired[record.as_of]
            @test mean(record.predictive) === mean(other.predictive)
            @test std(record.predictive) === std(other.predictive)
            @test record.train_rows == other.train_rows
            @test record.train_realised_through == other.train_realised_through
            # The fitted parameters themselves, not only what they produced.
            @test record.model.params_hash == other.model.params_hash
            compared += 1
        end
        @test compared > 150
    end

    @testset "a replay decision does not move when its future is destroyed" begin
        config = ReplayConfig(warmup = 200, refit_every = 25)
        clean = replay(
            la_models(),
            build_training_set(la_engine(series), "LEAK"; horizon_bars = 1); config = config,
        )
        dirty = replay(
            la_models(),
            build_training_set(la_engine(poisoned), "LEAK"; horizon_bars = 1); config = config,
        )
        paired = Dict(record.as_of => record for record in dirty.records)
        limits = RiskLimits()
        compared = 0
        for record in clean.records
            record.realised_at < boundary || continue
            other = paired[record.as_of]
            @test mean(record.prediction) === mean(other.prediction)
            @test std(record.prediction) === std(other.prediction)
            @test probability_positive(record.prediction) ===
                probability_positive(other.prediction)
            # The pool weights are scored on realised outcomes, which is exactly where a
            # future outcome would leak in if the scoring ran a bar early.
            @test record.prediction.weights.probabilities ==
                other.prediction.weights.probabilities
            # And the decision the system would have taken, which is the thing that matters.
            left = decide(record.prediction, limits)
            right = decide(other.prediction, limits)
            @test left.action === right.action
            @test left.target_weight === right.target_weight
            compared += 1
        end
        @test compared > 150
    end

    @testset "training data never realises after the prediction it feeds" begin
        # The mutation audit on this file found the hole this closes. Comparing a clean run
        # against a poisoned one cannot see a leak that moves both runs the same way: delete
        # the walk-forward embargo and every invariance above still holds, because the
        # training window widened identically on both sides.
        #
        # This is the direct statement instead. Whatever the training window was, its last
        # realised outcome must already have happened when the prediction was made.
        for horizon in (1, 5)
            examples = build_training_set(
                la_engine(series), "LEAK"; horizon_bars = horizon,
            )
            records = walk_forward(
                () -> BayesianReturnModel(
                    [:log_return_1, :momentum_10_5]; horizon_bars = horizon,
                ),
                examples, WalkForwardConfig(initial_train = 200, refit_every = 20),
            )
            @test !isempty(records)
            for record in records
                # Strictly before. An outcome that realises at the moment of the decision is
                # the boundary case, and the boundary is where these bugs live.
                @test record.train_realised_through < record.as_of
            end

            # The embargo is exactly the minimum needed rather than a round number someone
            # liked: it is applied to the example index, so an h-bar horizon leaves a one-bar
            # gap, and one bar less puts the last training label on the decision itself.
            horizon > 1 || continue
            tight = walk_forward(
                () -> BayesianReturnModel(
                    [:log_return_1, :momentum_10_5]; horizon_bars = horizon,
                ),
                examples,
                WalkForwardConfig(
                    initial_train = 200, refit_every = 20, embargo_bars = horizon - 1,
                ),
            )
            @test any(
                record -> record.train_realised_through >= record.as_of, tight,
            )
        end
    end

    @testset "the scaler is fitted on the training window, not the sample" begin
        # Normalising over the whole sample is the classic leak that survives every unit test,
        # because the scaled features still look perfectly reasonable.
        names = [:log_return_1, :momentum_10_5]
        # The design matrix the scaler is fitted on, assembled the way the model assembles it.
        design(rows) = reduce(
            vcat, [permutedims(design_row(row.features, names)) for row in rows],
        )
        examples = build_training_set(la_engine(series), "LEAK"; horizon_bars = 1)
        early = fit_scaler(names, design(examples[1:200]))
        late = fit_scaler(names, design(examples[1:400]))
        @test early.centres != late.centres || early.scales != late.scales

        poisoned_examples = build_training_set(
            la_engine(poisoned), "LEAK"; horizon_bars = 1,
        )
        # The first 200 rows are identical in both, so a scaler fitted on them must be too.
        @test fit_scaler(names, design(poisoned_examples[1:200])).centres == early.centres
        @test fit_scaler(names, design(poisoned_examples[1:200])).scales == early.scales
    end

    @testset "a streaming session reaches the same state as the offline path" begin
        # Two different code paths over the same bars. If one of them can see further than the
        # other, they disagree.
        offline = replay(
            la_models(),
            build_training_set(la_engine(series), "LEAK"; horizon_bars = 1);
            config = ReplayConfig(warmup = 200, refit_every = 25),
        )
        session = PaperTradingSession(
            "LEAK", la_models(), la_features();
            horizon_bars = 1, warmup = 200, refit_every = 25,
            interval = Day(1), interval_label = "1d", max_silence = Day(3),
            journal = nothing,
        )
        for bar in series.bars
            on_tick!(session, Quote("LEAK", bar.timestamp, bar.close; volume = bar.volume))
        end
        report = session_report(session)
        @test report["predictions"] > 0
        # Not equality of counts, which differ by how each path handles its tail, but the
        # session must never have scored more predictions than the offline path saw bars.
        @test report["settled"] <= length(offline.records) + report["pending"] + 5
        @test report["halted_bars"] == 0
    end
end
