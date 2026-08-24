# The paper broker. Most of these tests are about the costs that make a paper record evidence
# rather than an artefact.

bk_quote(price = 2_500.0; volume = 1.0e6, at = DateTime(2026, 1, 2, 10), symbol = "RELIANCE") =
    Quote(symbol, at, price; volume = volume)

bk_broker(; kwargs...) = PaperBroker(; kwargs...)

function bk_order(
        broker; side = BUY_SIDE, quantity = 100.0, kind = MARKET, limit = nothing,
        at = DateTime(2026, 1, 2, 10), symbol = "RELIANCE",
    )
    return Order(
        id = next_order_id!(broker), symbol = symbol, side = side, quantity = quantity,
        order_type = kind, limit_price = limit, placed_at = at,
    )
end

@testset "orders and fills" begin
    @testset "direction has one source of truth" begin
        broker = bk_broker()
        buy = bk_order(broker)
        sell = bk_order(broker; side = SELL_SIDE)
        @test is_buy(buy)
        @test !is_buy(sell)
        @test signed_quantity(buy) ≈ 100.0
        @test signed_quantity(sell) ≈ -100.0
        # A signed quantity plus a side is two sources of truth for one fact.
        @test_throws ArgumentError Order(
            id = "x", symbol = "A", side = BUY_SIDE, quantity = -1.0,
            placed_at = DateTime(2026, 1, 1),
        )
        @test_throws ArgumentError Order(
            id = "", symbol = "A", side = BUY_SIDE, quantity = 1.0,
            placed_at = DateTime(2026, 1, 1),
        )
        @test_throws ArgumentError Order(
            id = "x", symbol = "A", side = BUY_SIDE, quantity = 1.0, order_type = LIMIT,
            placed_at = DateTime(2026, 1, 1),
        )
    end

    @testset "cash moves the way a trade moves it" begin
        bought = Fill(
            order_id = "a", symbol = "A", side = BUY_SIDE, quantity = 10.0, price = 100.0,
            commission = 3.0, filled_at = DateTime(2026, 1, 1),
        )
        sold = Fill(
            order_id = "b", symbol = "A", side = SELL_SIDE, quantity = 10.0, price = 100.0,
            commission = 3.0, filled_at = DateTime(2026, 1, 1),
        )
        @test cash_flow(bought) ≈ -1_003.0
        @test cash_flow(sold) ≈ 997.0
        # The commission is paid in both directions, so a round trip at one price loses.
        @test cash_flow(bought) + cash_flow(sold) ≈ -6.0
        @test_throws ArgumentError Fill(
            order_id = "a", symbol = "A", side = BUY_SIDE, quantity = 0.0, price = 1.0,
            commission = 0.0, filled_at = DateTime(2026, 1, 1),
        )
        @test_throws ArgumentError Fill(
            order_id = "a", symbol = "A", side = BUY_SIDE, quantity = 1.0, price = 1.0,
            commission = -1.0, filled_at = DateTime(2026, 1, 1),
        )
    end

    @testset "a receipt cannot lie about what happened" begin
        broker = bk_broker()
        order = bk_order(broker)
        fill = Fill(
            order_id = order.id, symbol = "RELIANCE", side = BUY_SIDE, quantity = 100.0,
            price = 2_500.0, commission = 1.0, filled_at = DateTime(2026, 1, 2),
        )
        @test was_filled(OrderReceipt(order, FILLED, fill, "ok"))
        @test !was_filled(OrderReceipt(order, REJECTED, nothing, "no"))
        @test_throws ArgumentError OrderReceipt(order, FILLED, nothing, "no fill")
        @test_throws ArgumentError OrderReceipt(order, REJECTED, fill, "both")
    end
end

@testset "paper broker" begin
    @testset "slippage always moves against the order" begin
        # Modelling it as a spread that sometimes helps would be modelling a different
        # market from the one anybody trades in.
        costs = PaperCosts(slippage_rate = 0.001)
        @test fill_price(costs, BUY_SIDE, 100.0) ≈ 100.1
        @test fill_price(costs, SELL_SIDE, 100.0) ≈ 99.9

        broker = bk_broker()
        receipt = place_order!(broker, bk_order(broker), bk_quote(2_500.0))
        @test receipt.status === FILLED
        @test receipt.fill.price > 2_500.0
        selling = place_order!(
            broker, bk_order(broker; side = SELL_SIDE), bk_quote(2_500.0),
        )
        @test selling.fill.price < 2_500.0
    end

    @testset "a round trip at an unchanged price loses money" begin
        # The whole reason costs are charged here: a strategy that looks profitable before
        # them is not a finding.
        broker = bk_broker(starting_cash = 1.0e6)
        before = equity(broker)
        place_order!(broker, bk_order(broker), bk_quote(2_500.0))
        place_order!(broker, bk_order(broker; side = SELL_SIDE), bk_quote(2_500.0))
        @test isempty(broker.positions)
        @test equity(broker) < before
        # Two commissions and two crossings of the slippage.
        @test before - equity(broker) ≈
            2 * 100.0 * 2_500.0 * (0.0003 + 0.0005) rtol = 0.02
    end

    @testset "an order larger than the bar does not simply fill" begin
        broker = bk_broker()
        receipt = place_order!(
            broker, bk_order(broker; quantity = 1.0e6), bk_quote(2_500.0; volume = 1_000.0),
        )
        @test receipt.status === PARTIALLY_FILLED
        @test receipt.fill.quantity ≈ 100.0
        @test occursin("limited by volume", receipt.detail)
        @test tradeable_quantity(PaperCosts(), 500.0, 1_000.0) ≈ 100.0
        # No volume recorded means no constraint to apply rather than a refusal.
        @test tradeable_quantity(PaperCosts(), 500.0, 0.0) ≈ 500.0
    end

    @testset "it refuses what the cash cannot cover" begin
        broker = bk_broker(starting_cash = 100.0)
        receipt = place_order!(broker, bk_order(broker), bk_quote(2_500.0))
        @test receipt.status === REJECTED
        @test receipt.fill === nothing
        @test occursin("in cash", receipt.detail)
        @test broker.cash ≈ 100.0
        @test isempty(broker.positions)
    end

    @testset "averaging happens on the way in, not on the way out" begin
        broker = bk_broker()
        place_order!(broker, bk_order(broker; quantity = 100.0), bk_quote(2_000.0))
        first_price = broker.positions["RELIANCE"].average_price
        place_order!(broker, bk_order(broker; quantity = 100.0), bk_quote(3_000.0))
        averaged = broker.positions["RELIANCE"]
        @test averaged.quantity ≈ 200.0
        @test first_price < averaged.average_price < 3_000.0 * 1.001

        # Reducing leaves the average alone: the cost of what remains has not changed.
        place_order!(broker, bk_order(broker; side = SELL_SIDE, quantity = 50.0), bk_quote(4_000.0))
        reduced = broker.positions["RELIANCE"]
        @test reduced.quantity ≈ 150.0
        @test reduced.average_price ≈ averaged.average_price

        # Crossing through zero is a new position, not an adjusted one.
        place_order!(
            broker, bk_order(broker; side = SELL_SIDE, quantity = 250.0), bk_quote(5_000.0),
        )
        flipped = broker.positions["RELIANCE"]
        @test flipped.quantity ≈ -100.0
        @test flipped.average_price ≈ 5_000.0 * (1 - 0.0005)
    end

    @testset "closing a position removes it" begin
        broker = bk_broker()
        place_order!(broker, bk_order(broker; quantity = 100.0), bk_quote(2_500.0))
        @test haskey(broker.positions, "RELIANCE")
        place_order!(
            broker, bk_order(broker; side = SELL_SIDE, quantity = 100.0), bk_quote(2_600.0),
        )
        @test !haskey(broker.positions, "RELIANCE")
        @test length(broker.fills) == 2
    end

    @testset "a limit order that the market has not reached is working, not rejected" begin
        broker = bk_broker()
        away = place_order!(
            broker, bk_order(broker; kind = LIMIT, limit = 2_000.0), bk_quote(2_500.0),
        )
        @test away.status === OPEN
        @test away.fill === nothing
        @test occursin("not reached", away.detail)

        reachable = place_order!(
            broker, bk_order(broker; kind = LIMIT, limit = 2_600.0), bk_quote(2_500.0),
        )
        @test reachable.status === FILLED
        @test reachable.fill.price ≈ 2_600.0

        cancelled = cancel_order!(broker, away.order.id)
        @test cancelled.status === CANCELLED
        @test cancel_order!(broker, reachable.order.id).status === FILLED
        @test_throws ArgumentError cancel_order!(broker, "nope")
    end

    @testset "equity is cash plus what is held, marked to market" begin
        broker = bk_broker(starting_cash = 1.0e6)
        place_order!(broker, bk_order(broker; quantity = 100.0), bk_quote(2_500.0))
        held = equity(broker)
        mark_to_market!(broker, [bk_quote(3_000.0)])
        @test equity(broker) > held
        @test broker.positions["RELIANCE"].last_price ≈ 3_000.0
        # A quote for something not held changes nothing.
        mark_to_market!(broker, [bk_quote(1.0; symbol = "INFY")])
        @test broker.positions["RELIANCE"].last_price ≈ 3_000.0

        book = portfolio(broker; as_of = DateTime(2026, 1, 3))
        @test book.equity ≈ equity(broker)
        @test book.cash ≈ broker.cash
        @test n_positions(book) == 1
    end

    @testset "it is paper, and says so" begin
        broker = bk_broker()
        @test broker_mode(broker) === PAPER
        @test !is_live(broker)
        @test occursin("PaperBroker", sprint(show, broker))
        @test_throws ArgumentError PaperBroker(starting_cash = 0.0)
        @test_throws ArgumentError PaperCosts(slippage_rate = 1.5)
        @test_throws ArgumentError PaperCosts(max_participation = 0.0)
        @test_throws ArgumentError place_order!(
            broker, bk_order(broker), bk_quote(100.0; symbol = "INFY"),
        )
    end
end

@testset "from ruling to order" begin
    limits = RiskLimits()
    prediction = rk_prediction(mu = 0.03, sd = 0.02)
    intent = decide(prediction, limits)

    @testset "it sends the approved size, never the requested one" begin
        # A trade the risk engine reduced cannot be restored by an arithmetic slip here.
        broker = bk_broker(starting_cash = 1.0e6)
        held = Dict("RELIANCE" => rk_position("RELIANCE", 0.03))
        ruling = review(
            intent, rk_portfolio(positions = held), limits; sector = "energy",
        )
        @test was_reduced(ruling)

        order = order_from_ruling(broker, ruling, bk_quote(2_500.0), 1.0e6)
        @test order !== nothing
        @test order.side === BUY_SIDE
        @test order.quantity ≈ ruling.approved_weight * 1.0e6 / 2_500.0
        @test order.quantity < ruling.requested_weight * 1.0e6 / 2_500.0
    end

    @testset "a refusal produces no order at all" begin
        broker = bk_broker()
        refused = review(intent, rk_portfolio(), limits; halted = true)
        @test !approved(refused)
        @test order_from_ruling(broker, refused, bk_quote(2_500.0), 1.0e6) === nothing

        declined = decide(rk_prediction(mu = 0.0, sd = 0.02), limits)
        nothing_to_do = review(declined, rk_portfolio(), limits)
        @test order_from_ruling(broker, nothing_to_do, bk_quote(2_500.0), 1.0e6) === nothing
    end

    @testset "a sell becomes a sell" begin
        broker = bk_broker()
        bearish = decide(rk_prediction(mu = -0.03, sd = 0.02), limits)
        ruling = review(bearish, rk_portfolio(), limits; sector = "energy")
        order = order_from_ruling(broker, ruling, bk_quote(2_500.0), 1.0e6)
        @test order.side === SELL_SIDE
        @test order.quantity > 0
    end

    @testset "the whole path runs end to end" begin
        broker = bk_broker(starting_cash = 1.0e6)
        price = bk_quote(2_500.0)
        book = portfolio(broker; as_of = DateTime(2026, 1, 2))
        ruling = review(intent, book, limits; sector = "energy")
        order = order_from_ruling(broker, ruling, price, book.equity)
        receipt = place_order!(broker, order, price)

        @test receipt.status === FILLED
        @test haskey(broker.positions, "RELIANCE")
        after = portfolio(broker; as_of = DateTime(2026, 1, 2))
        @test position_weight(after, "RELIANCE") <= limits.max_position_weight + 1.0e-3
        @test after.equity < 1.0e6
    end
end
