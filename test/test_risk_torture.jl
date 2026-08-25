# The risk engine, attacked rather than exercised.
#
# The validation baseline ran 1,478 decisions through this component and not one gate failed.
# That is not evidence the gates work; it is evidence the baseline never reached them. Every
# scenario below is built to reach one.

const RT_WHEN = DateTime(2026, 3, 2, 15, 30)

rt_position(; quantity = 100.0, price = 1000.0, sector = "energy") = Position(
    symbol = "RT", quantity = quantity, average_price = price, last_price = price,
    opened_at = RT_WHEN - Day(5), sector = sector,
)

function rt_book(;
        equity = 1.0e6, cash = 1.0e6, positions = Dict{String, Position}(),
        peak = nothing, day_start = nothing,
    )
    return Portfolio(
        equity = equity, cash = cash, positions = positions,
        peak_equity = peak, day_start_equity = day_start, as_of = RT_WHEN,
    )
end

rt_intent(action = BUY; weight = 0.05, symbol = "RT") = TradeIntent(
    symbol = symbol, as_of = RT_WHEN, horizon_bars = 1, action = action,
    target_weight = weight, reason = nothing,
    evidence = Dict{Symbol, Float64}(:expected_return => 0.01),
)

named(ruling, name) = only(check for check in ruling.checks if check.name === name)
statuses(ruling) = Dict(check.name => check.status for check in ruling.checks)

@testset "risk engine under attack" begin
    @testset "an oversized request is trimmed, not refused" begin
        # The headroom under a limit is a trade that genuinely satisfies it. Refusing the
        # whole thing throws away a position the account is entitled to hold.
        limits = RiskLimits(max_position_weight = 0.1)
        ruling = review(rt_intent(BUY; weight = 0.4), rt_book(), limits)
        @test approved(ruling)
        @test ruling.requested_weight == 0.4
        @test ruling.approved_weight ≈ 0.1
        @test named(ruling, :position_weight).status === PASS
    end

    @testset "a position already over its limit can still be trimmed" begin
        # The failure this guards against locks in the breach: every trim charged as though
        # it were a purchase, so the engine refuses the only trade that fixes the problem.
        limits = RiskLimits(max_position_weight = 0.1)
        over = rt_book(
            equity = 1.0e6, cash = 0.0,
            positions = Dict("RT" => rt_position(quantity = 300.0)),
        )
        @test position_weight(over, "RT") > limits.max_position_weight

        trim = review(rt_intent(SELL; weight = 0.2), over, limits)
        @test approved(trim)
        @test trim.approved_weight > 0

        # And the same size in the other direction is not allowed, because that increases it.
        add = review(rt_intent(BUY; weight = 0.2), over, limits)
        @test !approved(add)
    end

    @testset "a full liquidation of an over-limit position is allowed" begin
        limits = RiskLimits(max_position_weight = 0.1)
        over = rt_book(
            equity = 1.0e6, cash = 0.0,
            positions = Dict("RT" => rt_position(quantity = 400.0)),
        )
        weight = position_weight(over, "RT")
        ruling = review(rt_intent(SELL; weight = weight), over, limits)
        @test approved(ruling)
        @test ruling.approved_weight ≈ weight
    end

    @testset "a short being covered is reducing, not opening" begin
        # A buy against a short unwinds exposure exactly as a sell against a long does. An
        # engine that only knows about longs charges this one as an increase.
        limits = RiskLimits(max_position_weight = 0.1)
        short = rt_book(
            equity = 1.0e6, cash = 1.5e6,
            positions = Dict("RT" => rt_position(quantity = -300.0)),
        )
        cover = review(rt_intent(BUY; weight = 0.2), short, limits)
        @test approved(cover)
        deepen = review(rt_intent(SELL; weight = 0.2), short, limits)
        @test !approved(deepen)
    end

    @testset "an account-level breach refuses outright" begin
        # A smaller trade does not fix a halted account or a breached drawdown, so there is
        # nothing to reduce to.
        limits = RiskLimits(max_daily_loss = 0.02, max_drawdown = 0.1)

        lost = rt_book(equity = 9.0e5, cash = 9.0e5, day_start = 1.0e6)
        loss_ruling = review(rt_intent(), lost, limits)
        @test !approved(loss_ruling)
        @test loss_ruling.approved_weight == 0.0
        @test named(loss_ruling, :daily_loss).status === FAIL

        drawn = rt_book(equity = 8.0e5, cash = 8.0e5, peak = 1.0e6, day_start = 8.0e5)
        drawdown_ruling = review(rt_intent(), drawn, limits)
        @test !approved(drawdown_ruling)
        @test named(drawdown_ruling, :drawdown).status === FAIL

        halted = review(rt_intent(), rt_book(), limits; halted = true)
        @test !approved(halted)
        @test named(halted, :kill_switch).status === FAIL
    end

    @testset "a refused account never reports ceilings it did not reach" begin
        # A ruling is an audit record. Showing a position-weight check that was never
        # evaluated invents a fact about a decision that was already over.
        limits = RiskLimits(max_daily_loss = 0.02)
        lost = rt_book(equity = 9.0e5, cash = 9.0e5, day_start = 1.0e6)
        ruling = review(rt_intent(), lost, limits)
        reached = statuses(ruling)
        @test !haskey(reached, :position_weight)
        @test !haskey(reached, :portfolio_exposure)
        @test haskey(reached, :daily_loss)
    end

    @testset "every simultaneous breach is recorded, not just the first" begin
        limits = RiskLimits(
            max_daily_loss = 0.02, max_drawdown = 0.05,
            max_annualised_volatility = 0.3, min_daily_turnover = 1.0e7,
        )
        wrecked = rt_book(equity = 8.0e5, cash = 8.0e5, peak = 1.0e6, day_start = 1.0e6)
        ruling = review(
            rt_intent(), wrecked, limits;
            halted = true, annualised_volatility = 2.5, daily_turnover = 1000.0,
        )
        @test !approved(ruling)
        failed_names = Set(check.name for check in failures(ruling))
        for name in (:kill_switch, :daily_loss, :drawdown, :volatility, :liquidity)
            @test name in failed_names
        end
    end

    @testset "a gate with nothing to measure is skipped, and that is visible" begin
        # Skipped is not passed. It is recorded as its own status precisely so that a reader
        # can tell a limit that held from a limit that never ran, and a session that forgets
        # to supply an input does not silently disable a limit without saying so.
        ruling = review(rt_intent(), rt_book(), RiskLimits())
        for name in (:volatility, :liquidity, :sector_exposure)
            @test named(ruling, name).status === SKIPPED
            @test isnan(named(ruling, name).observed)
        end
        @test approved(ruling)

        supplied = review(
            rt_intent(), rt_book(), RiskLimits();
            annualised_volatility = 0.2, daily_turnover = 1.0e9, sector = "energy",
        )
        for name in (:volatility, :liquidity, :sector_exposure)
            @test named(supplied, name).status !== SKIPPED
        end
    end

    @testset "illiquidity and extreme volatility refuse" begin
        limits = RiskLimits(min_daily_turnover = 1.0e7, max_annualised_volatility = 0.6)
        thin = review(rt_intent(), rt_book(), limits; daily_turnover = 5.0e5)
        @test !approved(thin)
        @test named(thin, :liquidity).status === FAIL
        @test named(thin, :liquidity).observed == 5.0e5

        wild = review(rt_intent(), rt_book(), limits; annualised_volatility = 3.0)
        @test !approved(wild)
        @test named(wild, :volatility).status === FAIL

        # And the boundary in both directions, because that is where the comparison lives.
        @test approved(review(rt_intent(), rt_book(), limits; daily_turnover = 1.0e7))
        @test approved(
            review(rt_intent(), rt_book(), limits; annualised_volatility = 0.6),
        )
    end

    @testset "sector and portfolio ceilings bind together" begin
        # RiskLimits refuses an incoherent set, so these nest the way it requires: a
        # position cannot exceed its sector, and a sector cannot exceed the portfolio.
        limits = RiskLimits(
            max_position_weight = 0.2, max_sector_exposure = 0.2,
            max_portfolio_exposure = 0.3,
        )
        book = rt_book(
            equity = 1.0e6, cash = 5.0e5,
            positions = Dict(
                "OTHER" => Position(
                    symbol = "OTHER", quantity = 150.0, average_price = 1000.0,
                    last_price = 1000.0, opened_at = RT_WHEN - Day(9), sector = "energy",
                ),
            ),
        )
        ruling = review(
            rt_intent(BUY; weight = 0.4), book, limits; sector = "energy",
        )
        # The tightest ceiling wins, and it is the one with the least headroom.
        @test approved(ruling)
        @test ruling.approved_weight <= limits.max_sector_exposure
        @test ruling.approved_weight <= limits.max_portfolio_exposure
        @test named(ruling, :sector_exposure).status === PASS
    end

    @testset "no headroom at all fails rather than approving nothing" begin
        # A refusal whose every check says PASS names nothing as its cause, and this is the
        # one component whose whole purpose is being auditable afterwards.
        limits = RiskLimits(
            max_position_weight = 0.1, max_sector_exposure = 0.1,
            max_portfolio_exposure = 0.1,
        )
        full = rt_book(
            equity = 1.0e6, cash = 0.0,
            positions = Dict(
                "OTHER" => Position(
                    symbol = "OTHER", quantity = 200.0, average_price = 1000.0,
                    last_price = 1000.0, opened_at = RT_WHEN - Day(9),
                ),
            ),
        )
        ruling = review(rt_intent(BUY; weight = 0.05), full, limits)
        @test !approved(ruling)
        @test ruling.approved_weight == 0.0
        @test !isempty(failures(ruling))
        @test :portfolio_exposure in Set(check.name for check in failures(ruling))
    end

    @testset "approved never exceeds requested, whatever the headroom" begin
        # The invariant that stops a ceiling from becoming a floor.
        limits = RiskLimits(
            max_position_weight = 0.9, max_sector_exposure = 0.9,
            max_portfolio_exposure = 0.9,
        )
        for weight in (0.001, 0.01, 0.05, 0.2, 0.5)
            ruling = review(rt_intent(BUY; weight = weight), rt_book(), limits)
            @test ruling.approved_weight <= ruling.requested_weight
        end
        over = rt_book(
            equity = 1.0e6, cash = 0.0,
            positions = Dict("RT" => rt_position(quantity = 500.0)),
        )
        for weight in (0.01, 0.1, 0.5)
            ruling = review(rt_intent(SELL; weight = weight), over, limits)
            @test ruling.approved_weight <= ruling.requested_weight
        end
    end

    @testset "a non-actionable intent is ruled on without touching a limit" begin
        for action in (HOLD, NO_TRADE)
            intent = TradeIntent(
                symbol = "RT", as_of = RT_WHEN, horizon_bars = 1, action = action,
                target_weight = 0.0, reason = INSUFFICIENT_DATA,
                evidence = Dict{Symbol, Float64}(),
            )
            ruling = review(intent, rt_book(), RiskLimits())
            @test !approved(ruling)
            @test ruling.approved_weight == 0.0
            @test only(ruling.checks).name === :actionable
            @test only(ruling.checks).status === SKIPPED
        end
    end

    @testset "the open-position count binds only when opening" begin
        limits = RiskLimits(max_open_positions = 1)
        held = rt_book(
            equity = 1.0e6, cash = 5.0e5,
            positions = Dict("RT" => rt_position(quantity = 50.0)),
        )
        # Adding to what is already held is not a new position.
        @test named(review(rt_intent(BUY), held, limits), :open_positions).status ===
            SKIPPED
        # A different symbol is, and there is no room for it.
        fresh = review(rt_intent(BUY; symbol = "NEW"), held, limits)
        @test named(fresh, :open_positions).status === FAIL
        @test !approved(fresh)
    end
end

@testset "the reported drawdown is the worst one" begin
    # The defect this covers: session_report called the drawdown at the final bar
    # "drawdown_pct" and printed it beside "return_pct", where it reads as the run's drawdown
    # and will be compared against one. A run can end near its high with a brutal trough
    # behind it.
    series = generate_series(
        AR1Returns(phi = 0.3, annual_drift = 0.04);
        symbol = "DD", n_bars = 900, seed = 99, start = Date(2023, 1, 2),
    )
    session = PaperTradingSession(
        "DD",
        (
            () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
        ),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]);
        horizon_bars = 1, warmup = 400, refit_every = 50,
        interval = Day(1), interval_label = "1d", max_silence = Day(3),
    )
    for bar in series.bars
        on_tick!(session, Quote("DD", bar.timestamp, bar.close; volume = bar.volume))
    end
    report = session_report(session)

    @test haskey(report, "max_drawdown_pct")
    @test haskey(report, "current_drawdown_pct")
    # Neither is named so the other could be mistaken for it.
    @test !haskey(report, "drawdown_pct")
    @test report["max_drawdown_pct"] >= report["current_drawdown_pct"]
    @test report["max_drawdown_pct"] >= 0
    @test report["current_drawdown_pct"] >= 0
    # It has to be a maximum over the run rather than a restatement of the final bar, and on
    # any run that ever gave anything back those two differ.
    @test session.max_drawdown >= 0
    @test session.peak_equity >= equity(session.broker)
end
