# Portfolio state, the decision engine, and the risk engine that outranks it.

rk_limits(; kwargs...) = RiskLimits(; kwargs...)

function rk_prediction(;
        mu = 0.03, sd = 0.02, epistemic = nothing, symbol = "RELIANCE",
        as_of = DateTime(2026, 1, 2), horizon = 5, df = 8.0,
    )
    share = epistemic === nothing ? 0.2 * sd^2 : epistemic
    return FusedPrediction(
        symbol = symbol, as_of = as_of, horizon_bars = horizon,
        distribution = OpinionPool((student_t(mu, sd, df),), [1.0]),
        epistemic_variance = share,
        weights = LabelledCategorical([MOMENTUM], [1.0]),
        sources = [
            ModelVersion(name = MOMENTUM, version = v"0.1.0", params_hash = "a"^32),
        ],
    )
end

rk_portfolio(; equity = 1_000_000.0, kwargs...) =
    Portfolio(; equity = equity, cash = equity, as_of = DateTime(2026, 1, 2), kwargs...)

rk_position(symbol, weight; equity = 1_000_000.0, sector = "energy") = Position(
    symbol = symbol, quantity = weight * equity / 100.0, average_price = 100.0,
    last_price = 100.0, opened_at = DateTime(2026, 1, 1), sector = sector,
)

@testset "portfolio state" begin
    long = rk_position("RELIANCE", 0.1)
    short = Position(
        symbol = "INFY", quantity = -50.0, average_price = 1_500.0, last_price = 1_450.0,
        opened_at = DateTime(2026, 1, 1), sector = "tech",
    )

    @testset "direction lives in the sign of the quantity" begin
        @test is_long(long)
        @test is_short(short)
        @test exposure(short) < 0
        @test gross_exposure(short) > 0
        # A short that fell is a short in profit, from the same expression as the long.
        @test unrealised_pnl(short) ≈ 2_500.0
        @test unrealised_pnl(long) ≈ 0.0
        @test occursin("Position", sprint(show, long))
    end

    @testset "both directions consume exposure" begin
        portfolio = rk_portfolio(
            positions = Dict("RELIANCE" => long, "INFY" => short),
        )
        @test n_positions(portfolio) == 2
        @test position_weight(portfolio, "RELIANCE") ≈ 0.1
        @test position_weight(portfolio, "ABSENT") == 0.0
        # Long one name and short another is exposed twice, not not at all.
        @test portfolio_exposure(portfolio) ≈ 0.1 + 72_500 / 1_000_000
        @test sector_exposure(portfolio, "tech") ≈ 72_500 / 1_000_000
        @test sector_exposure(portfolio, "pharma") == 0.0
        @test occursin("Portfolio", sprint(show, portfolio))
    end

    @testset "drawdown and daily loss are measured against carried history" begin
        portfolio = rk_portfolio(
            equity = 900_000.0, peak_equity = 1_200_000.0, day_start_equity = 1_000_000.0,
        )
        @test drawdown(portfolio) ≈ 0.25
        @test daily_loss(portfolio) ≈ 0.1
        # Up on the day is not a negative loss.
        @test daily_loss(
            rk_portfolio(
                equity = 1_100_000.0, peak_equity = 1_200_000.0,
                day_start_equity = 1_000_000.0
            )
        ) == 0.0
        @test drawdown(rk_portfolio()) == 0.0
    end

    @testset "state that cannot be true is refused" begin
        @test_throws ArgumentError Position(
            symbol = "", quantity = 1.0, average_price = 1.0, last_price = 1.0,
            opened_at = DateTime(2026, 1, 1),
        )
        @test_throws ArgumentError Position(
            symbol = "A", quantity = 0.0, average_price = 1.0, last_price = 1.0,
            opened_at = DateTime(2026, 1, 1),
        )
        @test_throws ArgumentError Position(
            symbol = "A", quantity = 1.0, average_price = -1.0, last_price = 1.0,
            opened_at = DateTime(2026, 1, 1),
        )
        @test_throws ArgumentError rk_portfolio(equity = 0.0)
        @test_throws ArgumentError rk_portfolio(equity = 1_000.0, peak_equity = 500.0)
        @test_throws ArgumentError Portfolio(
            equity = 1.0, cash = 1.0, positions = Dict("WRONG" => long),
            as_of = DateTime(2026, 1, 2),
        )
    end
end

@testset "decision engine" begin
    limits = rk_limits()

    @testset "an edge is acted on and its absence is not" begin
        strong = decide(rk_prediction(mu = 0.03, sd = 0.02), limits)
        @test strong.action === BUY
        @test is_actionable(strong)
        @test strong.target_weight > 0
        @test strong.reason === nothing
        @test strong.evidence[:probability_positive] > 0.9

        flat = decide(rk_prediction(mu = 0.0, sd = 0.02), limits)
        @test flat.action === NO_TRADE
        @test !is_actionable(flat)
        @test flat.reason === EDGE_TOO_SMALL
        @test flat.target_weight == 0.0

        bearish = decide(rk_prediction(mu = -0.03, sd = 0.02), limits)
        @test bearish.action === SELL
        @test bearish.target_weight > 0
    end

    @testset "the dangerous tail depends on which way the trade goes" begin
        # A large loss on a long is a large move down; on a short it is a large move up. A
        # downward-only gate would refuse the bearish trade for a risk it does not carry,
        # and would wave through the short whose upside tail is the dangerous one.
        bearish = rk_prediction(mu = -0.03, sd = 0.02)
        @test probability_below(bearish.distribution, -limits.large_loss_threshold) >
            limits.max_probability_large_loss
        @test decide(bearish, limits).action === SELL
        @test decide(bearish, limits).evidence[:probability_large_loss] ≈
            probability_above(bearish.distribution, limits.large_loss_threshold)

        bullish = rk_prediction(mu = 0.03, sd = 0.02)
        @test decide(bullish, limits).evidence[:probability_large_loss] ≈
            probability_below(bullish.distribution, -limits.large_loss_threshold)

        # A short whose upside tail is genuinely fat is still refused. Measured: P(up) is
        # 0.359 so the direction gate lets it through as a sell, and P(return > 5%) is
        # 0.173 against a ceiling of 0.15, which is the risk a downward-only gate would
        # have been blind to.
        fat = rk_prediction(mu = -0.03, sd = 0.08, epistemic = 1.0e-6)
        @test probability_above(fat.distribution, 0.0) <= 1 - limits.min_probability_positive
        @test probability_above(fat.distribution, limits.large_loss_threshold) >
            limits.max_probability_large_loss
        @test decide(fat, limits).action === NO_TRADE
        @test decide(fat, limits).reason === LOSS_PROBABILITY_TOO_HIGH
    end

    @testset "every refusal names its condition" begin
        # Wide enough and the direction gate stops it first, which is the more fundamental
        # refusal and so the one that should be recorded.
        wide = decide(rk_prediction(mu = 0.03, sd = 0.2), limits)
        @test wide.reason === EDGE_TOO_SMALL

        uncertain = decide(
            rk_prediction(mu = 0.03, sd = 0.02, epistemic = 0.9 * var(student_t(0.03, 0.02, 8.0))),
            limits,
        )
        @test uncertain.reason === UNCERTAINTY_TOO_HIGH

        disagreeing = decide(
            rk_prediction(mu = 0.03, sd = 0.02, epistemic = 0.5 * var(student_t(0.03, 0.02, 8.0))),
            limits,
        )
        @test disagreeing.reason === MODEL_DISAGREEMENT

        stale = decide(
            rk_prediction(mu = 0.03, sd = 0.02), limits;
            now = DateTime(2026, 2, 1), max_age = Day(3),
        )
        @test stale.reason === STALE_POSTERIOR
        # Fresh enough is not stale.
        @test decide(
            rk_prediction(mu = 0.03, sd = 0.02), limits;
            now = DateTime(2026, 1, 3), max_age = Day(3),
        ).action === BUY
    end

    @testset "a wider posterior sizes smaller, with no rule saying so" begin
        # The reason the models emit distributions rather than point forecasts.
        # Held at the same P(up) of 0.614 so only the width differs, and scored under a
        # loosened tail gate so that what is measured is the sizing rule rather than the
        # ceiling: at these widths the budget binds and the position cap does not.
        sizing = rk_limits(max_probability_large_loss = 0.5)
        narrow = decide(rk_prediction(mu = 0.03, sd = 0.1), sizing)
        wider = decide(rk_prediction(mu = 0.045, sd = 0.15), sizing)
        @test narrow.target_weight < sizing.max_position_weight
        @test probability_positive(rk_prediction(mu = 0.03, sd = 0.1)) ≈
            probability_positive(rk_prediction(mu = 0.045, sd = 0.15)) atol = 1.0e-9
        @test narrow.action === BUY
        @test wider.action === BUY
        @test wider.target_weight < narrow.target_weight

        # And never above the position ceiling however sharp it looks.
        @test decide(rk_prediction(mu = 0.05, sd = 0.001), limits).target_weight ==
            limits.max_position_weight
        # The budget is what is lost if the tail move happens.
        prediction = rk_prediction(mu = 0.03, sd = 0.1)
        weight = size_by_risk(prediction, limits, BUY)
        @test weight < limits.max_position_weight
        @test weight * downside(prediction, 0.05, BUY) ≈
            limits.risk_budget_per_trade rtol = 1.0e-9
    end

    @testset "the adverse tail is the one the trade can lose on" begin
        # Taking the larger magnitude of the two tails reads as conservative and is not: on
        # a strongly bullish posterior the larger magnitude is the upside, so a better edge
        # would report a bigger risk and size smaller. Measured before this was fixed, a
        # posterior centred at six per cent reported a downside of 9.7 per cent, which was
        # its ninety-fifth percentile gain.
        bullish = rk_prediction(mu = 0.03, sd = 0.02)
        @test downside(bullish, 0.05, BUY) ≈ -quantile(bullish.distribution, 0.05)
        @test downside(bullish, 0.05, BUY) < downside(rk_prediction(mu = 0.01, sd = 0.02), 0.05, BUY)

        # A short loses on a move up, so its adverse tail is the other one, and the two are
        # mirror images.
        bearish = rk_prediction(mu = -0.03, sd = 0.02)
        @test downside(bearish, 0.05, SELL) ≈ quantile(bearish.distribution, 0.95)
        @test downside(bearish, 0.05, SELL) ≈ downside(bullish, 0.05, BUY)

        # A tail already on the profitable side loses nothing at that probability, and the
        # floor keeps the division finite rather than claiming certainty.
        certain = rk_prediction(mu = 0.1, sd = 0.01)
        @test quantile(certain.distribution, 0.05) > 0
        @test downside(certain, 0.05, BUY) == 1.0e-4
        @test isfinite(size_by_risk(certain, limits, BUY))
    end

    @testset "a gate that cannot read its input fails closed" begin
        # An unmeasurable epistemic variance cannot be built at all, which is the stronger
        # protection: the guard downstream exists because NaN > x is false, and a gate that
        # cannot read its input must never read it as safe.
        @test_throws ArgumentError FusedPrediction(
            symbol = "RELIANCE", as_of = DateTime(2026, 1, 2), horizon_bars = 5,
            distribution = OpinionPool((student_t(0.03, 0.02, 8.0),), [1.0]),
            epistemic_variance = NaN,
            weights = LabelledCategorical([MOMENTUM], [1.0]),
            sources = [
                ModelVersion(name = MOMENTUM, version = v"0.1.0", params_hash = "a"^32),
            ],
        )

        # A Student-t below two degrees of freedom has infinite variance. The epistemic
        # share is then a finite number over an infinite one, which is zero, so a posterior
        # with no idea at all would read as the most confident input the gate has seen.
        formless = rk_prediction(mu = 0.03, sd = 0.02, df = 1.5, epistemic = 1.0e-4)
        @test !isfinite(var(formless.distribution))
        @test epistemic_share(formless) == 0.0
        @test decide(formless, limits).action === NO_TRADE
        @test decide(formless, limits).reason === UNCERTAINTY_TOO_HIGH
    end

    @testset "a posterior from the future is not the freshest one" begin
        # A one-sided comparison would treat a clock error as the newest evidence available.
        ahead = rk_prediction(as_of = DateTime(2026, 6, 1))
        @test decide(ahead, limits; now = DateTime(2026, 1, 2), max_age = Day(3)).reason ===
            STALE_POSTERIOR
    end

    @testset "the gates come from the limits, not from the code" begin
        strict = rk_limits(min_probability_positive = 0.95)
        prediction = rk_prediction(mu = 0.02, sd = 0.02)
        @test decide(prediction, rk_limits()).action === BUY
        @test decide(prediction, strict).reason === EDGE_TOO_SMALL

        # The wide posterior fails the direction gate before the tail gate is reached, so
        # loosening the tail alone changes nothing: the reason moves, the answer does not.
        loose = rk_limits(max_probability_large_loss = 0.9, min_confidence = 0.0)
        @test decide(rk_prediction(mu = 0.03, sd = 0.2), loose).reason === EDGE_TOO_SMALL
        permissive = rk_limits(
            max_probability_large_loss = 0.9, min_confidence = 0.0,
            min_probability_positive = 0.52,
        )
        # A gate at exactly a half makes every prediction a trade, so it is refused
        # outright rather than left as a configuration a typo can reach.
        @test_throws ArgumentError rk_limits(min_probability_positive = 0.5)
        @test_throws ArgumentError rk_limits(min_probability_positive = 0.3)
        @test decide(rk_prediction(mu = 0.03, sd = 0.2), permissive).action === BUY
    end

    @testset "an intent that contradicts itself cannot be built" begin
        @test_throws ArgumentError TradeIntent(
            symbol = "A", as_of = DateTime(2026, 1, 1), horizon_bars = 1,
            action = BUY, target_weight = 0.0,
        )
        @test_throws ArgumentError TradeIntent(
            symbol = "A", as_of = DateTime(2026, 1, 1), horizon_bars = 1,
            action = NO_TRADE, target_weight = 0.05,
        )
        @test_throws ArgumentError TradeIntent(
            symbol = "A", as_of = DateTime(2026, 1, 1), horizon_bars = 1,
            action = BUY, target_weight = 0.05, reason = EDGE_TOO_SMALL,
        )
        @test_throws ArgumentError decide(rk_prediction(), rk_limits(); tail = 0.9)
    end
end

@testset "risk engine" begin
    limits = rk_limits()
    intent = decide(rk_prediction(mu = 0.03, sd = 0.02), limits)

    @testset "it approves a clean trade and records every rule" begin
        ruling = review(intent, rk_portfolio(), limits; sector = "energy")
        @test approved(ruling)
        @test ruling.approved_weight ≈ intent.target_weight
        @test !was_reduced(ruling)
        @test isempty(failures(ruling))
        @test ruling.mode === PAPER
        @test length(ruling.checks) >= 7
        @test occursin("APPROVED", summarise(ruling))
        @test occursin("kill_switch", summarise(ruling))
        @test occursin("RiskRuling", sprint(show, ruling))
    end

    @testset "it can never approve more than was asked for" begin
        # The invariant the architecture rests on, enforced by the type rather than by care.
        @test_throws ArgumentError RiskRuling(
            symbol = "A", as_of = DateTime(2026, 1, 1), action = BUY,
            requested_weight = 0.01, approved_weight = 0.02,
            checks = RiskCheck[], mode = PAPER,
        )
        @test_throws ArgumentError RiskRuling(
            symbol = "A", as_of = DateTime(2026, 1, 1), action = BUY,
            requested_weight = 0.05, approved_weight = 0.05,
            checks = [RiskCheck(:x, FAIL, "no", 1.0, 0.0)], mode = PAPER,
        )
        for _ in 1:50
            ruling = review(intent, rk_portfolio(), limits)
            @test ruling.approved_weight <= ruling.requested_weight
        end
    end

    @testset "the kill switch refuses everything" begin
        ruling = review(intent, rk_portfolio(), limits; halted = true)
        @test !approved(ruling)
        @test ruling.approved_weight == 0.0
        @test :kill_switch in [check.name for check in failures(ruling)]
        @test occursin("REFUSED", summarise(ruling))
    end

    @testset "an account in trouble stops trading everything" begin
        # There is no smaller version of a breached drawdown, so these refuse rather than
        # reduce.
        drawn = rk_portfolio(
            equity = 800_000.0, peak_equity = 1_000_000.0, day_start_equity = 800_000.0,
        )
        ruling = review(intent, drawn, limits)
        @test !approved(ruling)
        @test :drawdown in [check.name for check in failures(ruling)]

        bleeding = rk_portfolio(
            equity = 950_000.0, peak_equity = 1_000_000.0, day_start_equity = 1_000_000.0,
        )
        losing = review(intent, bleeding, limits)
        @test !approved(losing)
        @test :daily_loss in [check.name for check in failures(losing)]
    end

    @testset "a ceiling reduces the trade rather than refusing it" begin
        # The headroom under a limit is a trade that genuinely satisfies the limit.
        held = Dict("RELIANCE" => rk_position("RELIANCE", 0.03))
        ruling = review(intent, rk_portfolio(positions = held), limits; sector = "energy")
        @test approved(ruling)
        @test was_reduced(ruling)
        @test ruling.approved_weight ≈ limits.max_position_weight - 0.03
        @test isempty(failures(ruling))

        # Already at the ceiling means nothing left to add, and the ceiling that stopped it
        # is named. A refusal whose every check says PASS records nothing as its cause, in
        # the one component whose whole purpose is being auditable afterwards.
        full = Dict("RELIANCE" => rk_position("RELIANCE", limits.max_position_weight))
        blocked = review(intent, rk_portfolio(positions = full), limits; sector = "energy")
        @test !approved(blocked)
        @test :position_weight in [check.name for check in failures(blocked)]
        @test !isempty(failures(blocked))
    end

    @testset "the book and the sector are ceilings too" begin
        # Four names at five per cent is twenty per cent of a twenty-five per cent sector
        # ceiling, so there is exactly five per cent of headroom left to trade into.
        crowded = Dict(
            string("NAME", index) => rk_position(string("NAME", index), 0.05; sector = "energy")
                for index in 1:4
        )
        ruling = review(intent, rk_portfolio(positions = crowded), limits; sector = "energy")
        @test approved(ruling)
        @test ruling.approved_weight ≈ min(intent.target_weight, 0.05)
        @test isempty(failures(ruling))

        # A sector already at its ceiling has no headroom at all.
        at_ceiling = Dict(
            string("NAME", index) => rk_position(string("NAME", index), 0.05; sector = "energy")
                for index in 1:5
        )
        blocked_sector = review(
            intent, rk_portfolio(positions = at_ceiling), limits; sector = "energy",
        )
        @test !approved(blocked_sector)
        @test :sector_exposure in [check.name for check in failures(blocked_sector)]
        # And a different sector is unaffected by it.
        @test approved(
            review(intent, rk_portfolio(positions = at_ceiling), limits; sector = "pharma"),
        )

        # The book ceiling, at it and under it. Six positions of ten per cent rather than
        # twelve of five: twelve trips the open-positions limit first, so the book gate is
        # never reached and the assertion passes on a refusal that had nothing to do with
        # it.
        packed = Dict(
            string("NAME", index) => rk_position(
                    string("NAME", index), 0.1;
                    sector = string("sector", index)
                )
                for index in 1:6
        )
        full_book = rk_portfolio(positions = packed)
        @test portfolio_exposure(full_book) ≈ limits.max_portfolio_exposure
        @test n_positions(full_book) < limits.max_open_positions
        book = review(intent, full_book, limits; sector = "fresh")
        @test !approved(book)
        @test :portfolio_exposure in [check.name for check in failures(book)]

        roomy = Dict(
            string("NAME", index) => rk_position(
                    string("NAME", index), 0.1;
                    sector = string("sector", index)
                )
                for index in 1:5
        )
        under = review(intent, rk_portfolio(positions = roomy), limits; sector = "fresh")
        @test approved(under)
        @test under.approved_weight ≈ min(intent.target_weight, 0.1)
        @test isempty(failures(under))
    end

    @testset "too many names refuses a new one but not an existing one" begin
        many = Dict(
            string("NAME", index) => rk_position(
                    string("NAME", index), 0.01;
                    sector = string("sector", index)
                )
                for index in 1:limits.max_open_positions
        )
        portfolio = rk_portfolio(positions = many)
        @test !approved(review(intent, portfolio, limits; sector = "fresh"))

        # Adding to a name already held is not opening a new position.
        with_held = copy(many)
        with_held["RELIANCE"] = rk_position("RELIANCE", 0.01)
        delete!(with_held, "NAME1")
        topped = review(intent, rk_portfolio(positions = with_held), limits; sector = "energy")
        @test approved(topped)
        @test :open_positions in
            [check.name for check in topped.checks if check.status === SKIPPED]
    end

    @testset "a limit it was not given is skipped, not passed" begin
        # A limit that was never evaluated is not a limit that was satisfied.
        ruling = review(intent, rk_portfolio(), limits)
        skipped = [check.name for check in ruling.checks if check.status === SKIPPED]
        @test :volatility in skipped
        @test :liquidity in skipped
        @test :sector_exposure in skipped

        # The sector default was the dangerous one: a placeholder sector holds nothing, so
        # its exposure is zero and a real ceiling reads as satisfied.
        at_ceiling = Dict(
            string("NAME", index) => rk_position(string("NAME", index), 0.05; sector = "energy")
                for index in 1:5
        )
        crowded = rk_portfolio(positions = at_ceiling)
        @test !approved(review(intent, crowded, limits; sector = "energy"))
        @test approved(review(intent, crowded, limits))
        @test :sector_exposure in
            [
            check.name for check in review(intent, crowded, limits).checks
                if check.status === SKIPPED
        ]

        supplied = review(
            intent, rk_portfolio(), limits;
            sector = "energy", annualised_volatility = 0.2, daily_turnover = 1.0e9,
        )
        @test all(check -> check.status !== SKIPPED, supplied.checks)
        @test approved(supplied)

        wild = review(intent, rk_portfolio(), limits; annualised_volatility = 5.0)
        @test !approved(wild)
        @test :volatility in [check.name for check in failures(wild)]

        illiquid = review(intent, rk_portfolio(), limits; daily_turnover = 1.0)
        @test !approved(illiquid)
        @test :liquidity in [check.name for check in failures(illiquid)]
    end

    @testset "a sell is ruled on the same way a buy is" begin
        bearish = decide(rk_prediction(mu = -0.03, sd = 0.02), limits)
        @test bearish.action === SELL
        ruling = review(bearish, rk_portfolio(), limits; sector = "energy")
        @test approved(ruling)
        @test ruling.action === SELL
        @test ruling.approved_weight ≈ bearish.target_weight
        # A short consumes exposure exactly as a long does.
        short_book = Dict(
            "OTHER" => Position(
                symbol = "OTHER", quantity = -500.0, average_price = 1_000.0,
                last_price = 1_000.0, opened_at = DateTime(2026, 1, 1), sector = "energy",
            ),
        )
        crowded = review(
            bearish, rk_portfolio(positions = short_book), limits; sector = "energy",
        )
        @test crowded.approved_weight < bearish.target_weight
    end

    @testset "a position over its ceiling can still be trimmed" begin
        # The inversion this guards against: charging a reducing trade for headroom locks
        # in the very breach the ceiling exists to prevent, and refuses the only trade that
        # would fix it.
        over = Dict(
            "RELIANCE" => Position(
                symbol = "RELIANCE", quantity = 800.0, average_price = 100.0,
                last_price = 100.0, opened_at = DateTime(2026, 1, 1), sector = "energy",
            ),
        )
        breached = rk_portfolio(positions = over)
        @test position_weight(breached, "RELIANCE") > limits.max_position_weight

        trim = decide(rk_prediction(mu = -0.03, sd = 0.02), limits)
        @test trim.action === SELL
        trimming = review(trim, breached, limits; sector = "energy")
        @test approved(trimming)
        @test trimming.approved_weight ≈ trim.target_weight
        @test isempty(failures(trimming))

        # Adding to it is still refused, which is the other half of the rule.
        adding = review(intent, breached, limits; sector = "energy")
        @test !approved(adding)
        @test :position_weight in [check.name for check in failures(adding)]

        # A short over its ceiling is trimmed by buying, symmetrically.
        short = Dict(
            "RELIANCE" => Position(
                symbol = "RELIANCE", quantity = -800.0, average_price = 100.0,
                last_price = 100.0, opened_at = DateTime(2026, 1, 1), sector = "energy",
            ),
        )
        covering = review(intent, rk_portfolio(positions = short), limits; sector = "energy")
        @test approved(covering)
        @test !approved(review(trim, rk_portfolio(positions = short), limits; sector = "energy"))
    end

    @testset "there is nothing to rule on when the decision declined" begin
        declined = decide(rk_prediction(mu = 0.0, sd = 0.02), limits)
        ruling = review(declined, rk_portfolio(), limits)
        @test !approved(ruling)
        @test ruling.requested_weight == 0.0
        @test :actionable in [check.name for check in ruling.checks]
        @test occursin("edge_too_small", summarise(ruling))
    end

    @testset "the ruling is reproducible from the limits alone" begin
        # No model, no prediction: the same inputs must give the same ruling forever.
        portfolio = rk_portfolio(positions = Dict("RELIANCE" => rk_position("RELIANCE", 0.02)))
        first_ruling = review(intent, portfolio, limits; sector = "energy")
        second_ruling = review(intent, portfolio, limits; sector = "energy")
        @test first_ruling.approved_weight == second_ruling.approved_weight
        @test [check.status for check in first_ruling.checks] ==
            [check.status for check in second_ruling.checks]

        tighter = rk_limits(max_position_weight = 0.03)
        @test review(intent, portfolio, tighter; sector = "energy").approved_weight ≈ 0.01
    end
end
