#!/usr/bin/env julia
#
#   GROWW_API_KEY=... GROWW_API_SECRET=... julia --project=. examples/groww_history.jl RELIANCE
#
# Fetches real NSE daily history from Groww and replays the pipeline over it. Read only: this
# script cannot place an order, and neither can the client it uses. GROWW_READ_PATHS permits two
# paths and neither of them is an order endpoint.
#
# Credentials come from the environment and nowhere else. Do not write them into this file or
# pass them on a command line, where they land in a shell history. A secret that has been seen
# once should be regenerated.
#
# HTTP.jl is a weak dependency: the package resolves, backtests and paper trades without it, and
# the transport only exists once it is loaded. Add it to your environment first:
#
#   julia --project=. -e 'using Pkg; Pkg.add("HTTP")'

using BayesTrade, Dates, Printf, Statistics
using HTTP        # brings the transport in; without it connect_groww says so and stops

function main(symbol::String, years::Int)
    # Authenticates and hands back a GrowwSource wrapped in a retry. A rate limit and a gateway
    # timeout are retried; an unknown symbol and a rejected token are not.
    source = connect_groww()
    session = source.inner.session
    @printf("authenticated, token expires %s\n", something(session.expiry, "unknown"))

    stop = Date(now(UTC) + IST_OFFSET)
    start = stop - Year(years)
    @printf("fetching %s daily bars, %s to %s\n", symbol, start, stop)
    bars = fetch_bars(source, symbol; start = start, stop = stop, interval = "1d")
    isempty(bars) && error("no bars came back for $symbol")
    @printf(
        "%d bars, %s to %s, last close %.2f\n",
        length(bars), Date(first(bars).timestamp), Date(last(bars).timestamp),
        last(bars).close,
    )

    # A vendor series is not a clean series. Say what is wrong with it before trusting it. The
    # calendar is passed so a holiday is not reported as a hole in the download; it covers 2025
    # and 2026, and gaps outside those years fall back to weekday arithmetic.
    quality = validate_bars(bars; calendar = nse_calendar())
    println(summarise(quality))
    is_usable(quality) || error("the series failed its quality check; see the issues above")

    engine = FeatureEngine(
        InMemoryBarStore(bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, symbol; horizon_bars = 1)
    length(examples) < 260 &&
        error("only $(length(examples)) usable examples; fetch a longer window")

    replayed = replay(
        (
            () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
            () -> MarketRegimeModel(; horizon_bars = 1),
        ),
        examples;
        config = ReplayConfig(warmup = 250, refit_every = 25),
    )

    actions = Dict{String, Int}()
    limits = RiskLimits()
    for record in replayed.records
        key = slug(decide(record.prediction, limits).action)
        actions[key] = get(actions, key, 0) + 1
    end

    println("\n", "="^68)
    @printf("%s: %d scored bars\n", symbol, length(replayed.records))
    for (action, count) in sort(collect(actions), by = first)
        @printf("  %-10s %5d\n", action, count)
    end
    println("="^68)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(
        length(ARGS) >= 1 ? ARGS[1] : "RELIANCE",
        length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3,
    )
end
