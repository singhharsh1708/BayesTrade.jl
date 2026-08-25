#!/usr/bin/env julia
#
#   GROWW_API_KEY=... GROWW_API_SECRET=... julia --project=. examples/groww_history.jl RELIANCE
#
# Fetches real NSE daily history from Groww and replays the pipeline over it. Read only: this
# script cannot place an order, and neither can the client it uses, because GROWW_READ_PATHS
# permits exactly two paths and neither of them is an order endpoint.
#
# Credentials come from the environment and nowhere else. Do not paste them into this file, a
# shell history, or a message. A secret that has been seen once should be regenerated.
#
# HTTP.jl is deliberately not a dependency of BayesTrade: a trading system's dependency list is
# a liability, and the transport is injected precisely so the package does not need one. Add it
# to your own environment to run this:
#
#   julia --project=. -e 'using Pkg; Pkg.add("HTTP")'

using BayesTrade, Dates, JSON3, Printf, Statistics
using HTTP

"""
An HTTP transport for `GrowwSession`. Takes a request, returns a response, and does not raise
on a 4xx or 5xx: the client reads the status itself and sorts the failure into one that a retry
helps with and one that it does not.
"""
function http_transport(request::GrowwRequest)
    url = string(GROWW_API_ROOT, request.path)
    headers = collect(request.headers)
    response = if request.method === :GET
        HTTP.get(
            url, headers; query = request.query, status_exception = false, retry = false,
        )
    else
        HTTP.post(
            url, headers, JSON3.write(request.body);
            status_exception = false, retry = false,
        )
    end
    return GrowwResponse(response.status, String(response.body))
end

function main(symbol::String, years::Int)
    credentials = groww_credentials_from_env()
    credentials === nothing && error(
        "set GROWW_API_KEY (and GROWW_API_SECRET for an approval key) in the environment",
    )

    session = GrowwSession(credentials, http_transport)
    authenticate!(session)
    @printf("authenticated, token expires %s\n", something(session.expiry, "unknown"))

    # Wrapped in a retry: a rate limit and a gateway timeout pass, an unknown symbol does not,
    # and the wrapper knows the difference.
    source = RetryingSource(GrowwSource(session); attempts = 3)
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

    # A vendor series is not a clean series. Say what is wrong with it before trusting it.
    quality = validate_bars(bars)
    println(summarise(quality))
    is_usable(quality) || error("the series failed its quality check; see the issues above")

    engine = FeatureEngine(
        InMemoryBarStore(bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, symbol; horizon_bars = 1)
    length(examples) < 260 && error(
        "only $(length(examples)) usable examples; fetch a longer window",
    )

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
