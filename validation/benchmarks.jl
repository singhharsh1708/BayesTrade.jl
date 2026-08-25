#!/usr/bin/env julia
#
#   julia --project=validation validation/benchmarks.jl
#
# Sections 14 to 17 of the validation brief.
#
# Every strategy here is run through the same harness, over the same bars, paying the same costs,
# scored on the same metrics. That is the only way a comparison means anything, and it is the
# reason the harness applies costs itself rather than letting each strategy account for its own.
#
# BayesTrade is not tuned to win. Its parameters are the package defaults.

include(joinpath(@__DIR__, "harness.jl"))

"""
    Costs

What a change of position costs, as fractions of the traded notional.

`slippage` is charged on every unit of turnover and always against the trade, never for it.
"""
Base.@kwdef struct Costs
    commission::Float64 = 0.0003
    slippage::Float64 = 0.0005
end

scaled(costs::Costs, factor::Real) =
    Costs(commission = costs.commission * factor, slippage = costs.slippage * factor)

"""
    run_weights(bars, weights, costs)

Equity from a sequence of target weights.

The weight for bar `i` is the position held *into* bar `i + 1`, so a strategy is never paid for
a bar it decided on after seeing. Turnover is charged when the weight changes.
"""
function run_weights(bars::Vector{Bar}, weights::Vector{Float64}, costs::Costs)
    length(weights) == length(bars) ||
        throw(ArgumentError("one weight per bar, got $(length(weights)) for $(length(bars))"))
    equity = 1.0
    held = 0.0
    curve = Float64[equity]
    turnover = 0.0
    wins = 0
    trades = 0
    gains = 0.0
    losses = 0.0
    returns = Float64[]

    for index in 1:(length(bars) - 1)
        opening = equity
        target = weights[index]
        traded = abs(target - held)
        if traded > 0
            cost = traded * (costs.commission + costs.slippage)
            equity *= (1 - cost)
            turnover += traded
        end
        held = target
        step = bars[index + 1].close / bars[index].close - 1
        profit = held * step
        equity *= (1 + profit)
        push!(curve, equity)
        # Net of what it cost to get there. Recording the gross move instead makes every
        # risk-adjusted number blind to costs, and a cost sensitivity sweep then reports the
        # same Sharpe at half cost and at five times it.
        push!(returns, equity / opening - 1)
        if !iszero(held)
            trades += 1
            profit > 0 ? (wins += 1; gains += profit) : (losses -= profit)
        end
    end
    return (
        equity = equity, curve = curve, turnover = turnover, returns = returns,
        wins = wins, trades = trades, gains = gains, losses = losses,
    )
end

"""
    metrics(result, n_bars)

The numbers section 16 asks for, from one run.
"""
function metrics(result, n_bars::Int)
    years = n_bars / BARS_PER_YEAR
    curve = result.curve
    peak = curve[1]
    worst = 0.0
    for value in curve
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    end
    returns = result.returns
    spread = isempty(returns) ? 0.0 : std(returns)
    downside = isempty(returns) ? 0.0 :
        sqrt(mean([min(value, 0.0)^2 for value in returns]))
    average = isempty(returns) ? 0.0 : mean(returns)
    # A strategy can be wiped out, and a negative equity has no real root. Reporting -100%
    # is both true and the only thing that can be compared against the others.
    growth = result.equity <= 0 ? -1.0 : result.equity^(1 / years) - 1
    exposed = count(!iszero, result.returns)
    return Dict{String, Any}(
        "cagr" => rounded(growth, 4),
        "bars_with_a_position" => exposed,
        "share_of_bars_exposed" => rounded(
            isempty(result.returns) ? 0.0 : exposed / length(result.returns), 4,
        ),
        "total_return" => rounded(result.equity - 1, 4),
        "sharpe" => rounded(spread > 0 ? average / spread * ANNUALISER : 0.0, 3),
        "sortino" => rounded(downside > 0 ? average / downside * ANNUALISER : 0.0, 3),
        "max_drawdown" => rounded(worst, 4),
        "volatility" => rounded(spread * ANNUALISER, 4),
        "turnover" => rounded(result.turnover, 2),
        "win_rate" => rounded(result.trades > 0 ? result.wins / result.trades : 0.0, 4),
        "profit_factor" => rounded(result.losses > 0 ? result.gains / result.losses : 0.0, 3),
    )
end

# ---------------------------------------------------------------------------- baselines

buy_and_hold(bars) = fill(1.0, length(bars))

function random_weights(bars; seed = 4242)
    generator = MersenneTwister(seed)
    return [rand(generator) < 0.5 ? -1.0 : 1.0 for _ in bars]
end

"""
    momentum_weights(bars; lookback)

Long when the trailing return is positive, short when it is not.

Uses bars strictly before the decision, which is the same rule the rest of the system follows.
"""
function momentum_weights(bars; lookback = 20)
    weights = zeros(Float64, length(bars))
    for index in (lookback + 1):length(bars)
        weights[index] = bars[index].close > bars[index - lookback].close ? 1.0 : -1.0
    end
    return weights
end

function moving_average_weights(bars; fast = 10, slow = 50)
    weights = zeros(Float64, length(bars))
    closes = [bar.close for bar in bars]
    for index in (slow + 1):length(bars)
        quick = mean(closes[(index - fast + 1):index])
        slow_average = mean(closes[(index - slow + 1):index])
        weights[index] = quick > slow_average ? 1.0 : 0.0
    end
    return weights
end

"""
    bayestrade_weights(bars, examples; limits)

The system's own answer, as a weight per bar.

The decision and risk engines are used exactly as the live session uses them, including the
volatility gate, so this is the strategy the package would actually run rather than a sketch of
it.
"""
function bayestrade_weights(bars, examples; limits = RiskLimits())
    report = replay(
        standard_factories(), examples;
        config = ReplayConfig(warmup = 500, refit_every = 25),
    )
    by_time = Dict(record.as_of => record for record in report.records)
    book = Portfolio(equity = 1.0e6, cash = 1.0e6, as_of = last(examples).features.as_of)
    weights = zeros(Float64, length(bars))
    for (index, bar) in enumerate(bars)
        record = get(by_time, bar.timestamp, nothing)
        record === nothing && continue
        intent = decide(record.prediction, limits)
        ruling = review(
            intent, book, limits;
            annualised_volatility = annualise(std(record.prediction)),
            daily_turnover = bar.volume * bar.close,
        )
        approved(ruling) || continue
        weights[index] = intent.action === SELL ?
            -ruling.approved_weight : ruling.approved_weight
    end
    return (weights = weights, report = report)
end

# ---------------------------------------------------------------------------- runs

function compare(process, label; n_bars = 3000, seed = 20260825, costs = Costs())
    series = fixed_series(n_bars = n_bars, seed = seed, process = process)
    examples = training_set(series)
    bars = series.bars
    system = bayestrade_weights(bars, examples)

    strategies = (
        ("buy_and_hold", buy_and_hold(bars)),
        ("random", random_weights(bars)),
        ("momentum_20", momentum_weights(bars)),
        ("moving_average", moving_average_weights(bars)),
        ("bayestrade", system.weights),
    )

    println()
    println(uppercase(label))
    @printf(
        "%-16s %8s %8s %8s %9s %8s %9s %8s %8s %8s\n",
        "strategy", "cagr", "sharpe", "sortino", "maxdd", "vol", "turnover", "win", "pf",
        "exposed",
    )
    results = Dict{String, Any}()
    for (name, weights) in strategies
        outcome = metrics(run_weights(bars, weights, costs), n_bars)
        results[name] = outcome
        @printf(
            "%-16s %8.4f %8.3f %8.3f %9.4f %8.4f %9.1f %8.3f %8.3f %8.3f\n",
            name, outcome["cagr"], outcome["sharpe"], outcome["sortino"],
            outcome["max_drawdown"], outcome["volatility"], outcome["turnover"],
            outcome["win_rate"], outcome["profit_factor"],
            outcome["share_of_bars_exposed"],
        )
    end
    calibration = system.report.calibration
    results["_calibration"] = Dict{String, Any}(
        "interval_error" => rounded(interval_calibration_error(calibration)),
        "brier_score" => rounded(calibration.brier_score),
        "mean_log_score" => rounded(calibration.mean_log_score),
    )
    return (results = results, bars = bars, weights = system.weights)
end

"""
    sensitivity(bars, weights)

What the same positions are worth at half, one, two and five times the cost assumption.

A strategy whose edge survives only at the costs it was designed against has no edge, and the
brief asks for that to be flagged rather than discovered later.
"""
function sensitivity(bars, weights)
    heading("cost sensitivity")
    base = Costs()
    output = Dict{String, Any}()
    @printf("%-10s %10s %10s %10s\n", "multiple", "cagr", "sharpe", "maxdd")
    for factor in (0.5, 1.0, 2.0, 5.0)
        outcome = metrics(run_weights(bars, weights, scaled(base, factor)), length(bars))
        output[string(factor)] = outcome
        @printf(
            "%-10s %10.4f %10.3f %10.4f\n", string(factor, "x"),
            outcome["cagr"], outcome["sharpe"], outcome["max_drawdown"],
        )
    end
    return output
end

function main()
    heading("baselines on the same data, costs and metrics")
    all_results = Dict{String, Any}()
    trending = nothing
    for (label, process) in (
            ("ar1, genuine predictability", AR1Returns(phi = 0.35, annual_drift = 0.05)),
            ("gaussian, no predictability", GaussianReturns(annual_drift = 0.05)),
            ("regime switching", RegimeSwitchingReturns()),
            (
                "stochastic volatility", StochasticVolatilityReturns(
                    annual_drift = 0.05, persistence = 0.97,
                    volatility_of_volatility = 0.35,
                ),
            ),
        )
        outcome = compare(process, label)
        all_results[label] = outcome.results
        label == "ar1, genuine predictability" && (trending = outcome)
    end

    all_results["_cost_sensitivity"] = sensitivity(trending.bars, trending.weights)

    record(
        "benchmarks",
        Dict{String, Any}(
            "recorded_at" => string(now(UTC)),
            "commit" => strip(read(`git rev-parse --short HEAD`, String)),
            "costs" => Dict{String, Any}(
                "commission" => Costs().commission, "slippage" => Costs().slippage,
            ),
            "results" => all_results,
        ),
    )
    heading("written")
    println(joinpath(RESULTS, "benchmarks.json"))
    return nothing
end

main()
