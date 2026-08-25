# Phase G and J: can a decision be reconstructed from the file, and can a restart avoid
# trading a bar the previous process already traded.

rc_series(; n = 900) = generate_series(
    AR1Returns(phi = 0.55, annual_drift = 0.0);
    symbol = "SYNTH", n_bars = n, seed = 7, start = Date(2026, 1, 1),
)

rc_session(journal) = PaperTradingSession(
    "SYNTH", (() -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),),
    FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]);
    horizon_bars = 1, warmup = 700, refit_every = 50,
    interval = Day(1), interval_label = "1d", max_silence = Day(3), journal = journal,
)

function rc_feed!(session, series, range = 1:length(series.bars))
    for bar in series.bars[range]
        on_tick!(session, Quote("SYNTH", bar.timestamp, bar.close; volume = bar.volume))
    end
    return session
end

@testset "decision records" begin
    @testset "a decision can be reconstructed from the file alone" begin
        # The question the record has to answer is why the system acted here. Answering it
        # later from a summary is not possible: the features, the posteriors and the prices
        # are all gone by then.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            rc_feed!(rc_session(path), rc_series())
            entries = [JSON3.read(line, Dict{String, Any}) for line in readlines(path)]
            bars = [e for e in entries if e["event"] == "bar"]
            @test !isempty(bars)

            entry = first(bars)
            for key in (
                    "as_of", "symbol", "close", "volume", "reference_price", "features",
                    "components", "fused", "action", "evidence", "requested", "approved",
                    "risk_checks", "equity",
                )
                @test haskey(entry, key)
            end
            @test !isempty(entry["features"])
            @test !isempty(entry["components"])

            component = first(entry["components"])
            for key in ("model", "version", "weight", "mean", "sd", "uncertainty")
                @test haskey(component, key)
            end

            fused = entry["fused"]
            @test haskey(fused, "probability_up")
            @test haskey(fused, "lower")
            @test haskey(fused, "upper")
            @test haskey(fused, "epistemic_share")
            @test fused["lower"] < fused["upper"]

            # Every gate, with what it saw and what it allowed.
            @test !isempty(entry["risk_checks"])
            check = first(entry["risk_checks"])
            @test haskey(check, "name")
            @test haskey(check, "status")
        end
    end

    @testset "a fill records what it actually cost" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            rc_feed!(rc_session(path), rc_series())
            entries = [JSON3.read(line, Dict{String, Any}) for line in readlines(path)]
            filled = [
                e for e in entries
                    if e["event"] == "bar" && e["filled"] !== nothing && e["filled"] != 0
            ]
            @test !isempty(filled)
            entry = first(filled)
            @test entry["order_id"] !== nothing
            @test entry["fill_price"] !== nothing
            @test entry["fees"] > 0
            # Slippage against the price on the screen when the decision was made, recorded
            # rather than derived later, because the reference price is gone by then.
            @test entry["slippage"] !== nothing
            @test entry["fill_price"] ≈ entry["reference_price"] + entry["slippage"]
        end
    end

    @testset "nothing unwritable reaches the file" begin
        # JSON has no NaN and no infinity, and both occur legitimately: a skipped gate has
        # no observed value and an unfitted model has infinite uncertainty. Writing one
        # raises, the journal reports a failed write, and the session halts. Null is the
        # honest encoding.
        @test BayesTrade.jsonable(1.5) === 1.5
        @test BayesTrade.jsonable(NaN) === nothing
        @test BayesTrade.jsonable(Inf) === nothing
        @test BayesTrade.jsonable(-Inf) === nothing
        @test BayesTrade.jsonable(nothing) === nothing

        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            session = rc_feed!(rc_session(path), rc_series())
            @test !session.journal_failed
            @test session.counters.halted_bars == 0
            for line in readlines(path)
                @test !occursin("NaN", line)
                @test !occursin("Inf", line)
            end
        end
    end
end

@testset "restart without trading the same bar twice" begin
    @testset "a journal says how far the last process got" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            first_process = rc_feed!(rc_session(path), rc_series(), 1:800)
            state = read_journal(path)
            @test !isempty(state)
            @test state.last_as_of !== nothing
            @test state.fills == session_report(first_process)["fills"]
            @test occursin("JournalState", sprint(show, state))
        end
    end

    @testset "a restart skips what was already handled" begin
        # Every re-processed bar is a second order for a decision already taken. Here that
        # corrupts the record; against a real venue it is a duplicate trade.
        mktempdir() do dir
            series = rc_series()
            path = joinpath(dir, "journal.jsonl")
            first_process = rc_feed!(rc_session(path), series, 1:800)
            before = session_report(first_process)["fills"]

            naive = rc_feed!(rc_session(nothing), series)
            recovered = rc_session(nothing)
            resume!(recovered, path)
            rc_feed!(recovered, series)
            after = session_report(recovered)

            @test after["replayed"] > 700
            @test after["fills"] == 0
            @test before + after["fills"] < session_report(naive)["fills"]
            @test already_handled(recovered, first(series.bars))
            @test !already_handled(recovered, last(series.bars))
        end
    end

    @testset "a torn final line does not discard the record" begin
        # A crash cuts the last write in half. Throwing away weeks of good history over one
        # truncated line would be the worse failure.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            rc_feed!(rc_session(path), rc_series(), 1:800)
            whole = read_journal(path)

            open(path, "a") do handle
                write(handle, "{\"event\":\"bar\",\"as_of\":\"2029-")
            end
            torn = read_journal(path)
            @test torn.entries == whole.entries
            @test torn.last_as_of == whole.last_as_of
        end
    end

    @testset "an absent journal is an empty one, not an error" begin
        mktempdir() do dir
            state = read_journal(joinpath(dir, "never-written.jsonl"))
            @test isempty(state)
            @test state.last_as_of === nothing
            session = rc_session(nothing)
            @test !already_handled(session, first(rc_series().bars))
        end
    end
end
