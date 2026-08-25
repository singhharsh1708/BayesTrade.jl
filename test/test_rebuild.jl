# Rebuilding the position book on restart.
#
# A resumed session used to start with an empty book and a correct watermark, which is safe where
# positions are notional and wrong everywhere else. These cover what it reconstructs, and more
# importantly what it refuses to reconstruct.

rb_series(; n_bars = 700, seed = 7) = generate_series(
    AR1Returns(phi = 0.35, annual_drift = 0.05);
    symbol = "RB", n_bars = n_bars, seed = seed, start = Date(2026, 1, 2),
)

rb_models() = (
    () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
    () -> BayesianVolatilityModel(; horizon_bars = 1),
)

rb_session(journal) = PaperTradingSession(
    "RB", rb_models(), FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]);
    horizon_bars = 1, warmup = 300, refit_every = 50,
    interval = Day(1), interval_label = "1d", max_silence = Day(3), journal = journal,
)

rb_feed!(session, bars) = for bar in bars
    on_tick!(session, Quote("RB", bar.timestamp, bar.close; volume = bar.volume))
end

"""
    write_journal(path, entries)

A journal written by hand, so a case that a running session rarely produces can still be tested.
"""
function write_journal(path::AbstractString, entries)
    open(path, "w") do handle
        for entry in entries
            JSON3.write(handle, entry)
            println(handle)
        end
    end
    return path
end

rb_start(; cash = 1.0e6) = Dict{String, Any}(
    "event" => "session_started", "schema" => 1, "symbol" => "RB",
    "starting_cash" => cash, "horizon_bars" => 1, "warmup" => 300,
)

function rb_fill(;
        symbol = "RB", as_of, quantity, price, fees = 0.0, close = price, equity,
        order_id = "o-1",
    )
    return Dict{String, Any}(
        "event" => "bar", "schema" => 1, "symbol" => symbol, "as_of" => string(as_of),
        "close" => close, "filled" => quantity, "fill_price" => price, "fees" => fees,
        "order_id" => order_id, "equity" => equity, "positions" => 1,
    )
end

@testset "rebuilding an account from its journal" begin
    @testset "a real session's book is reconstructed exactly" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            session = rb_session(path)
            rb_feed!(session, rb_series().bars[1:500])
            @test session_report(session)["fills"] > 20

            account = rebuild_account(path)
            @test account.consistent
            @test isempty(account.problems)
            @test account.starting_cash == 1.0e6
            @test account.cash ≈ session.broker.cash rtol = 1.0e-9
            @test length(account.positions) == length(session.broker.positions)
            for (symbol, position) in session.broker.positions
                rebuilt = account.positions[symbol]
                @test rebuilt.quantity ≈ position.quantity rtol = 1.0e-9
                @test rebuilt.average_price ≈ position.average_price rtol = 1.0e-6
            end
            @test account.rebuilt_equity ≈ account.journalled_equity rtol =
                REBUILD_EQUITY_TOLERANCE
            @test occursin("Consistent", rebuild_report(account))
        end
    end

    @testset "an empty account rebuilds as empty, and that is consistent" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            session = rb_session(path)
            rb_feed!(session, rb_series().bars[1:200])       # never fits, never trades
            account = rebuild_account(path)
            @test account.consistent
            @test isempty(account.positions)
            @test account.cash == 1.0e6
            @test account.n_fills == 0
        end
    end

    @testset "several positions are rebuilt independently" begin
        # A session trades one symbol, so this journal is written by hand. The arithmetic it
        # exercises is per-symbol accumulation, which is what a multi-symbol deployment needs.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_start(),
                    rb_fill(
                        symbol = "AAA", as_of = DateTime(2026, 3, 2), quantity = 100.0,
                        price = 50.0, fees = 5.0, equity = 999_995.0,
                    ),
                    rb_fill(
                        symbol = "BBB", as_of = DateTime(2026, 3, 3), quantity = -40.0,
                        price = 200.0, fees = 8.0, equity = 999_987.0,
                    ),
                    rb_fill(
                        symbol = "AAA", as_of = DateTime(2026, 3, 4), quantity = 100.0,
                        price = 60.0, fees = 6.0, close = 60.0, equity = 1_000_981.0,
                    ),
                ],
            )
            account = rebuild_account(path)
            @test account.consistent
            @test length(account.positions) == 2
            @test account.positions["AAA"].quantity == 200.0
            # Weighted by size, which is what an average price is.
            @test account.positions["AAA"].average_price ≈ 55.0
            @test account.positions["BBB"].quantity == -40.0
            @test account.positions["BBB"].average_price ≈ 200.0
            @test account.cash ≈ 1.0e6 - (100 * 50 + 5) - (-40 * 200 + 8) - (100 * 60 + 6)
        end
    end

    @testset "a position closed out leaves no position" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_start(),
                    rb_fill(as_of = DateTime(2026, 3, 2), quantity = 100.0, price = 50.0, equity = 1.0e6),
                    rb_fill(as_of = DateTime(2026, 3, 3), quantity = -100.0, price = 55.0, equity = 1_000_500.0),
                ],
            )
            account = rebuild_account(path)
            @test isempty(account.positions)
            @test account.cash ≈ 1.0e6 + 500
            @test account.n_fills == 2
        end
    end

    @testset "a partial fill is rebuilt at the quantity that actually filled" begin
        # The journal records what filled, not what was asked for, so a partial fill needs no
        # special handling here. This asserts that, because assuming it is where the bug goes.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_start(),
                    rb_fill(as_of = DateTime(2026, 3, 2), quantity = 37.0, price = 50.0, equity = 1.0e6),
                ],
            )
            account = rebuild_account(path)
            @test account.positions["RB"].quantity == 37.0
            @test account.cash ≈ 1.0e6 - 37 * 50
        end
    end

    @testset "a rejected order changes nothing" begin
        # A rejected order is journalled with no fill. It must not move the book, and it must
        # not count as a fill either.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_start(),
                    Dict{String, Any}(
                        "event" => "bar", "symbol" => "RB",
                        "as_of" => string(DateTime(2026, 3, 2)), "close" => 50.0,
                        "filled" => 0.0, "fill_price" => nothing, "fees" => nothing,
                        "order_id" => "o-9", "detail" => "rejected: insufficient cash",
                        "equity" => 1.0e6,
                    ),
                ],
            )
            account = rebuild_account(path)
            @test account.consistent
            @test isempty(account.positions)
            @test account.n_fills == 0
            @test account.cash == 1.0e6
        end
    end

    @testset "crossing through zero resets the average price" begin
        # Averaging a long into a short produces a number that is neither, and it is the
        # number every P&L is computed from afterwards.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_start(),
                    rb_fill(as_of = DateTime(2026, 3, 2), quantity = 100.0, price = 50.0, equity = 1.0e6),
                    rb_fill(as_of = DateTime(2026, 3, 3), quantity = -150.0, price = 60.0, close = 60.0, equity = 1_001_000.0),
                ],
            )
            account = rebuild_account(path)
            @test account.positions["RB"].quantity == -50.0
            @test account.positions["RB"].average_price ≈ 60.0     # not a blend of 50 and 60
        end
    end
end

@testset "a reconstruction is checked before it is trusted" begin
    @testset "no opening balance means the cash is unknowable" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [rb_fill(as_of = DateTime(2026, 3, 2), quantity = 100.0, price = 50.0, equity = 1.0e6)],
            )
            account = rebuild_account(path)
            @test !account.consistent
            @test any(occursin("session_started", problem) for problem in account.problems)
            @test account.starting_cash === nothing
            @test occursin("INCONSISTENT", rebuild_report(account))
        end
    end

    @testset "equity that does not add up is refused" begin
        # The check that catches a missed fill: the replay implies one account and the journal
        # last recorded another.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_start(),
                    rb_fill(
                        as_of = DateTime(2026, 3, 2), quantity = 100.0, price = 50.0,
                        close = 50.0, equity = 2.0e6,      # nothing here produces two million
                    ),
                ],
            )
            account = rebuild_account(path)
            @test !account.consistent
            @test any(occursin("disagrees", problem) for problem in account.problems)
        end
    end

    @testset "a fill with no price cannot be replayed" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_start(),
                    Dict{String, Any}(
                        "event" => "bar", "symbol" => "RB",
                        "as_of" => string(DateTime(2026, 3, 2)), "close" => 50.0,
                        "filled" => 100.0, "fill_price" => nothing, "equity" => 1.0e6,
                    ),
                ],
            )
            account = rebuild_account(path)
            @test !account.consistent
            @test any(occursin("no price", problem) for problem in account.problems)
        end
    end

    @testset "a torn final line is tolerated and a shredded file is not" begin
        # One truncated write is an ordinary crash. Many are a file that cannot be trusted.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            session = rb_session(path)
            rb_feed!(session, rb_series().bars[1:500])
            open(path, "a") do handle
                write(handle, "{\"event\":\"bar\",\"as_of\":\"2029-")
            end
            @test rebuild_account(path).consistent

            open(path, "a") do handle
                for _ in 1:5
                    println(handle, "}}}not json{{{")
                end
            end
            shredded = rebuild_account(path)
            @test !shredded.consistent
            @test any(occursin("unreadable lines", problem) for problem in shredded.problems)
        end
    end

    @testset "an absent journal is not a rebuilt account" begin
        mktempdir() do dir
            account = rebuild_account(joinpath(dir, "never-written.jsonl"))
            @test !account.consistent
            @test account.n_fills == 0
            @test any(occursin("no journal", problem) for problem in account.problems)
        end
    end
end

@testset "resume installs what it rebuilt, or refuses" begin
    @testset "a restart holds what the previous process held" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            series = rb_series()
            first_process = rb_session(path)
            rb_feed!(first_process, series.bars[1:500])
            held = copy(first_process.broker.positions)
            cash = first_process.broker.cash
            @test !isempty(held)

            resumed = rb_session(nothing)
            resume!(resumed, path)
            @test resumed.rebuild !== nothing
            @test resumed.rebuild.consistent
            @test length(resumed.broker.positions) == length(held)
            for (symbol, position) in held
                @test resumed.broker.positions[symbol].quantity ≈ position.quantity rtol = 1.0e-9
            end
            @test resumed.broker.cash ≈ cash rtol = 1.0e-9
            @test may_trade(check_health(resumed, series.bars[500].timestamp)) ||
                !isempty(problems(check_health(resumed, series.bars[500].timestamp)))
        end
    end

    @testset "the drawdown limit does not rearm across a restart" begin
        # Peak equity has to come back too. Without it a session already deep in a hole
        # believes it is at its high, and the drawdown limit stops binding exactly when it
        # matters most.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            first_process = rb_session(path)
            rb_feed!(first_process, rb_series().bars[1:500])
            peak = first_process.peak_equity

            resumed = rb_session(nothing)
            resume!(resumed, path)
            @test resumed.peak_equity > 0
            @test resumed.peak_equity ≈ peak rtol = 0.05
            @test resumed.peak_equity >= equity(resumed.broker) - 1.0e-6
        end
    end

    @testset "an inconsistent rebuild installs nothing and stops trading" begin
        # A confident wrong book is worse than an empty one: the empty one trades nothing
        # until somebody looks, and the wrong one sizes every decision against a position
        # that is not there.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            write_journal(
                path,
                [
                    rb_fill(
                        as_of = DateTime(2026, 3, 2), quantity = 100.0, price = 50.0,
                        equity = 1.0e6,
                    ),
                ],
            )
            resumed = rb_session(nothing)
            resume!(resumed, path)
            @test !resumed.rebuild.consistent
            @test isempty(resumed.broker.positions)          # nothing installed
            health = check_health(resumed, DateTime(2026, 3, 3))
            @test !may_trade(health)
            @test :state_rebuild in Set(check.name for check in problems(health))
        end
    end

    @testset "installing can be declined, and then the venue is the authority" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            first_process = rb_session(path)
            rb_feed!(first_process, rb_series().bars[1:500])

            resumed = rb_session(nothing)
            resume!(resumed, path; install_positions = false)
            @test isempty(resumed.broker.positions)
            @test resumed.rebuild !== nothing
            @test resumed.rebuild.consistent            # it still rebuilt, it just did not install
            @test resumed.watermark !== nothing
        end
    end

    @testset "the existing no-double-trading behaviour is preserved" begin
        # The property the watermark was added for, re-asserted here because this change
        # rewrote the function that provides it.
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            series = rb_series()
            first_process = rb_session(path)
            rb_feed!(first_process, series.bars[1:500])

            naive = rb_session(nothing)
            rb_feed!(naive, series.bars)

            recovered = rb_session(nothing)
            resume!(recovered, path)
            rb_feed!(recovered, series.bars)
            report = session_report(recovered)

            @test report["replayed"] > 490
            @test report["fills"] < session_report(naive)["fills"]
            @test already_handled(recovered, first(series.bars))
            @test !already_handled(recovered, last(series.bars))
        end
    end

    @testset "the restart is journalled with what it did and did not install" begin
        mktempdir() do dir
            source = joinpath(dir, "source.jsonl")
            target = joinpath(dir, "target.jsonl")
            first_process = rb_session(source)
            rb_feed!(first_process, rb_series().bars[1:500])
            cp(source, target)

            resumed = rb_session(nothing)
            resume!(resumed, target)
            entries = [JSON3.read(line, Dict{String, Any}) for line in readlines(target)]
            record = only(e for e in entries if e["event"] == "resumed")
            @test record["rebuild_consistent"] == true
            @test record["positions_installed"] == true
            @test record["rebuilt_positions"] >= 0
            @test haskey(record, "rebuilt_cash")
            @test haskey(record, "problems")
        end
    end
end
