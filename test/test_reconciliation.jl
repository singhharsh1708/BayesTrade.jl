# Position reconciliation: the venue is the authority, and a disagreement stops new trading.
#
# Every scenario the brief names is here, each built deterministically rather than produced by
# running a session and hoping the interesting case turns up.

const RC_NOW = DateTime(2026, 8, 26, 15, 30)

rc_local_position(; symbol = "ACME", quantity = 100.0, price = 250.0) = Position(
    symbol = symbol, quantity = quantity, average_price = price, last_price = price,
    opened_at = RC_NOW - Day(3),
)

rc_venue_position(; symbol = "ACME", quantity = 100.0, price = 250.0) =
    ExternalPosition(symbol = symbol, quantity = quantity, average_price = price)

function rc_local(; positions = Position[], pending = Order[], as_of = RC_NOW, cash = 1.0e6)
    return LocalAccount(
        as_of, Dict(position.symbol => position for position in positions),
        collect(pending), cash,
    )
end

rc_venue(; positions = ExternalPosition[], pending = ExternalOrder[], as_of = RC_NOW) =
    VenueSnapshot(as_of = as_of, positions = positions, pending = pending)

rc_order(; id = "o-1", symbol = "ACME", side = BUY_SIDE, quantity = 10.0) = Order(
    id = id, symbol = symbol, side = side, quantity = quantity, placed_at = RC_NOW - Minute(1),
)

rc_venue_order(; id = "o-1", symbol = "ACME", side = BUY_SIDE, quantity = 10.0) =
    ExternalOrder(id = id, symbol = symbol, side = side, quantity = quantity)

kinds(result) = Set(item.kind for item in result.discrepancies)

"""
A source that answers with whatever it is given, or raises.
"""
struct StubVenue <: AccountSource
    answer::Any
end
BayesTrade.fetch_account(source::StubVenue, ::DateTime) =
    source.answer isa Exception ? throw(source.answer) : source.answer

@testset "reconciliation" begin
    @testset "matching books permit trading" begin
        result = reconcile(
            rc_local(positions = [rc_local_position()]),
            rc_venue(positions = [rc_venue_position()]),
        )
        @test result.status === RECONCILE_MATCHED
        @test reconciled(result)
        @test may_open_new_positions(result)
        @test isempty(result.discrepancies)
        @test result.n_local == 1
        @test result.n_venue == 1
        @test occursin("agree", reconciliation_report(result))
    end

    @testset "an empty account on both sides is a match, not an absence of evidence" begin
        result = reconcile(rc_local(), rc_venue())
        @test reconciled(result)
        @test result.n_local == 0
        @test result.n_venue == 0
    end

    @testset "local holds more than the broker" begin
        # The shape of a fill that was assumed and never happened.
        result = reconcile(
            rc_local(positions = [rc_local_position(quantity = 150.0)]),
            rc_venue(positions = [rc_venue_position(quantity = 100.0)]),
        )
        @test !reconciled(result)
        @test !may_open_new_positions(result)
        @test :quantity_mismatch in kinds(result)
        item = only(result.discrepancies)
        @test item.local_value == 150.0
        @test item.venue_value == 100.0
        @test occursin("150.0", item.detail)
        @test occursin("100.0", item.detail)
    end

    @testset "the broker holds more than local" begin
        # The shape of a fill that happened twice, or one the session never saw.
        result = reconcile(
            rc_local(positions = [rc_local_position(quantity = 100.0)]),
            rc_venue(positions = [rc_venue_position(quantity = 200.0)]),
        )
        @test !reconciled(result)
        @test :quantity_mismatch in kinds(result)
        @test only(result.discrepancies).venue_value == 200.0
    end

    @testset "a position the venue does not know about" begin
        result = reconcile(
            rc_local(positions = [rc_local_position()]), rc_venue(),
        )
        @test !reconciled(result)
        @test :missing_position in kinds(result)
        @test only(result.discrepancies).venue_value === nothing
    end

    @testset "a position the session does not know about" begin
        # A manual trade in the broker's own app is the ordinary cause, and it is the one a
        # local book can never detect on its own.
        result = reconcile(
            rc_local(), rc_venue(positions = [rc_venue_position(symbol = "MANUAL")]),
        )
        @test !reconciled(result)
        @test :unexpected_position in kinds(result)
        item = only(result.discrepancies)
        @test item.symbol == "MANUAL"
        @test item.local_value === nothing
    end

    @testset "average prices that disagree beyond rounding" begin
        # Same quantity, different cost basis: the shape of a partial fill counted at the
        # wrong price, and it moves every P&L number without moving a position count.
        within = reconcile(
            rc_local(positions = [rc_local_position(price = 250.0)]),
            rc_venue(positions = [rc_venue_position(price = 250.01)]),
        )
        @test reconciled(within)          # a hundredth on 250 is rounding, not a difference

        beyond = reconcile(
            rc_local(positions = [rc_local_position(price = 250.0)]),
            rc_venue(positions = [rc_venue_position(price = 262.0)]),
        )
        @test !reconciled(beyond)
        @test :average_price_mismatch in kinds(beyond)
    end

    @testset "a venue row of zero is a closed position, not a position" begin
        # Venues keep listing a symbol after it is closed. `Position` refuses a zero quantity
        # for the same reason, so both sides agree nothing is held.
        result = reconcile(
            rc_local(), rc_venue(positions = [rc_venue_position(quantity = 0.0)]),
        )
        @test reconciled(result)

        # But a zero against something we believe we hold is a real disagreement, and one of
        # the more alarming ones: the position we are sizing against is not there.
        vanished = reconcile(
            rc_local(positions = [rc_local_position(quantity = 100.0)]),
            rc_venue(positions = [rc_venue_position(quantity = 0.0)]),
        )
        @test !reconciled(vanished)
        @test :quantity_mismatch in kinds(vanished)
        @test only(vanished.discrepancies).venue_value == 0.0
    end

    @testset "pending orders are reconciled too" begin
        matched = reconcile(
            rc_local(pending = [rc_order()]), rc_venue(pending = [rc_venue_order()]),
        )
        @test reconciled(matched)

        # The venue has an order we do not know about: a duplicate submission, or one whose
        # acknowledgement was lost.
        theirs = reconcile(rc_local(), rc_venue(pending = [rc_venue_order()]))
        @test !reconciled(theirs)
        @test :unexpected_order in kinds(theirs)

        # We believe an order is live and the venue does not: it filled, or it was rejected,
        # and either way the next decision is being made against a position that has moved.
        ours = reconcile(rc_local(pending = [rc_order()]), rc_venue())
        @test !reconciled(ours)
        @test :missing_order in kinds(ours)

        # Same identifier, different order.
        differs = reconcile(
            rc_local(pending = [rc_order(quantity = 10.0)]),
            rc_venue(pending = [rc_venue_order(quantity = 25.0)]),
        )
        @test !reconciled(differs)
        @test :order_mismatch in kinds(differs)

        sided = reconcile(
            rc_local(pending = [rc_order(side = BUY_SIDE)]),
            rc_venue(pending = [rc_venue_order(side = SELL_SIDE)]),
        )
        @test !reconciled(sided)
        @test :order_mismatch in kinds(sided)
    end

    @testset "a stale snapshot is not evidence about now" begin
        # An old snapshot agreeing with the local book proves nothing, and a reconciliation
        # that ignores the age of its evidence passes during an outage.
        fresh = reconcile(
            rc_local(), rc_venue(as_of = RC_NOW - Minute(1)); as_of = RC_NOW,
        )
        @test reconciled(fresh)

        old = reconcile(
            rc_local(), rc_venue(as_of = RC_NOW - Hour(2)); as_of = RC_NOW,
        )
        @test !reconciled(old)
        @test :stale_snapshot in kinds(old)

        # A snapshot from the future is also not evidence: clocks disagree, and the one thing
        # it cannot be is a reading of the account as it is now.
        ahead = reconcile(
            rc_local(), rc_venue(as_of = RC_NOW + Minute(30)); as_of = RC_NOW,
        )
        @test !reconciled(ahead)
        @test :stale_snapshot in kinds(ahead)

        loose = reconcile(
            rc_local(), rc_venue(as_of = RC_NOW - Hour(2));
            as_of = RC_NOW, tolerances = ReconciliationTolerances(max_staleness = Day(1)),
        )
        @test reconciled(loose)
    end

    @testset "every discrepancy is reported, not only the first" begin
        # An operator handed one problem at a time fixes one, retries, and finds another. The
        # whole picture was available at the start.
        result = reconcile(
            rc_local(
                positions = [
                    rc_local_position(symbol = "A", quantity = 100.0),
                    rc_local_position(symbol = "B", quantity = 50.0),
                ],
                pending = [rc_order(id = "o-9", symbol = "A")],
                as_of = RC_NOW,
            ),
            rc_venue(
                positions = [
                    rc_venue_position(symbol = "A", quantity = 140.0),
                    rc_venue_position(symbol = "C", quantity = 10.0),
                ],
                as_of = RC_NOW - Hour(3),
            ),
        )
        @test !reconciled(result)
        @test length(result.discrepancies) >= 4
        for kind in (
                :quantity_mismatch, :missing_position, :unexpected_position,
                :missing_order, :stale_snapshot,
            )
            @test kind in kinds(result)
        end
    end

    @testset "an account that cannot be read is not a match" begin
        # Three states, not two. Nothing was compared, and reporting that as either agreement
        # or disagreement is a claim the evidence does not support.
        failing = reconcile(rc_local(), StubVenue(ErrorException("connection reset")))
        @test failing.status === RECONCILE_UNAVAILABLE
        @test !reconciled(failing)
        @test !may_open_new_positions(failing)
        @test :unavailable in kinds(failing)
        @test occursin("connection reset", only(failing.discrepancies).detail)
        @test failing.venue_as_of === nothing

        # And a source that answers with the wrong thing is the same situation.
        wrong = reconcile(rc_local(), StubVenue("not a snapshot"))
        @test wrong.status === RECONCILE_UNAVAILABLE

        working = reconcile(rc_local(), StubVenue(rc_venue()))
        @test reconciled(working)
    end

    @testset "the venue cannot report the same symbol twice" begin
        @test_throws ArgumentError VenueSnapshot(
            as_of = RC_NOW,
            positions = [rc_venue_position(), rc_venue_position(quantity = 5.0)],
        )
    end

    @testset "an external position must be a position" begin
        @test_throws ArgumentError ExternalPosition(
            symbol = "", quantity = 1.0, average_price = 1.0,
        )
        for bad in (Inf, NaN)
            @test_throws ArgumentError ExternalPosition(
                symbol = "X", quantity = bad, average_price = 1.0,
            )
            @test_throws ArgumentError ExternalPosition(
                symbol = "X", quantity = 1.0, average_price = bad,
            )
        end
        @test_throws ArgumentError ExternalPosition(
            symbol = "X", quantity = 1.0, average_price = 0.0,
        )
        @test_throws ArgumentError ExternalOrder(
            id = "", symbol = "X", side = BUY_SIDE, quantity = 1.0,
        )
        @test_throws ArgumentError ExternalOrder(
            id = "o", symbol = "X", side = BUY_SIDE, quantity = 0.0,
        )
    end

    @testset "the result is journalled and reads like a record" begin
        result = reconcile(
            rc_local(positions = [rc_local_position()]), rc_venue(),
        )
        record = reconciliation_record(result)
        @test record["event"] == "reconciliation"
        @test record["status"] == "mismatched"
        @test record["n_local"] == 1
        @test record["n_venue"] == 0
        @test length(record["discrepancies"]) == 1
        # It has to survive the journal, which means it has to be JSON.
        round_tripped = JSON3.read(JSON3.write(record), Dict{String, Any})
        @test round_tripped["status"] == "mismatched"
        @test round_tripped["discrepancies"][1]["kind"] == "missing_position"

        report = reconciliation_report(result)
        @test occursin("MISMATCHED", report)
        @test occursin("NEW TRADING MUST STOP", report)
        # And it says what not to do, because adopting the venue's numbers is the tempting
        # move and it erases the evidence of which side was wrong.
        @test occursin("erases the evidence", report)
        @test occursin("Reconciliation", sprint(show, result))
    end
end

@testset "a session refuses to trade against a book it cannot confirm" begin
    rs_series() = generate_series(
        AR1Returns(phi = 0.35, annual_drift = 0.05);
        symbol = "RECON", n_bars = 500, seed = 3, start = Date(2026, 1, 2),
    )

    function rs_session(; journal = nothing)
        session = PaperTradingSession(
            "RECON",
            (() -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),),
            FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]);
            horizon_bars = 1, warmup = 300, refit_every = 50,
            interval = Day(1), interval_label = "1d", max_silence = Day(3),
            journal = journal,
        )
        for bar in rs_series().bars
            on_tick!(session, Quote("RECON", bar.timestamp, bar.close; volume = bar.volume))
        end
        return session
    end

    @testset "never reconciled is not a failure in paper mode" begin
        # There is no external account to disagree with, so absence of a comparison is not
        # evidence of a problem. A live broker requires one, and the runbook says so.
        session = rs_session()
        @test session.reconciliation === nothing
        health = check_health(session, session.feed.health.last_tick_at)
        @test :reconciliation ∉ Set(check.name for check in health.checks)
        @test may_trade(health)
    end

    @testset "a mismatch stops the next decision" begin
        session = rs_session()
        moment = session.feed.health.last_tick_at
        wrong = VenueSnapshot(
            as_of = moment,
            positions = [
                ExternalPosition(symbol = "RECON", quantity = 12345.0, average_price = 900.0),
            ],
        )
        result = reconcile!(session, wrong)
        @test !reconciled(result)
        @test session.reconciliation === result
        health = check_health(session, moment)
        @test !may_trade(health)
        @test :reconciliation in Set(check.name for check in problems(health))
    end

    @testset "agreement restores it, and the agreement is recorded" begin
        mktempdir() do dir
            path = joinpath(dir, "journal.jsonl")
            session = rs_session(journal = path)
            moment = session.feed.health.last_tick_at
            agreeing = VenueSnapshot(
                as_of = moment,
                positions = [
                    ExternalPosition(
                            symbol = position.symbol, quantity = position.quantity,
                            average_price = position.average_price,
                        ) for position in values(session.broker.positions)
                ],
            )
            result = reconcile!(session, agreeing)
            @test reconciled(result)
            @test may_trade(check_health(session, moment))

            # A match is evidence too: the useful question tomorrow is usually when the books
            # last agreed, not when they stopped.
            entries = [JSON3.read(line, Dict{String, Any}) for line in readlines(path)]
            records = [e for e in entries if e["event"] == "reconciliation"]
            @test length(records) == 1
            @test records[1]["status"] == "matched"
        end
    end

    @testset "an unreadable venue stops trading as firmly as a mismatch" begin
        session = rs_session()
        moment = session.feed.health.last_tick_at
        result = reconcile!(session, StubVenue(ErrorException("timeout")))
        @test result.status === RECONCILE_UNAVAILABLE
        @test !may_trade(check_health(session, moment))
    end

    @testset "reconciling before anything has been fed is refused, not guessed" begin
        empty_session = PaperTradingSession(
            "RECON",
            (() -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),),
            FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]);
            horizon_bars = 1, warmup = 300, refit_every = 50,
            interval = Day(1), interval_label = "1d", max_silence = Day(3),
        )
        @test_throws ArgumentError reconcile!(empty_session, VenueSnapshot(as_of = RC_NOW))
        # Given a moment explicitly, it works.
        result = reconcile!(
            empty_session, VenueSnapshot(as_of = RC_NOW); as_of = RC_NOW,
        )
        @test reconciled(result)
    end
end
