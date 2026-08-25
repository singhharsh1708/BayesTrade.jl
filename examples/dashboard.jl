#!/usr/bin/env julia
#
# First time, from the repository root:
#
#   julia --project=examples -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
#
# Then:
#
#   julia --project=examples examples/dashboard.jl                   # synthetic history
#   julia --project=examples examples/dashboard.jl RELIANCE          # real history, needs credentials
#   julia --project=examples examples/dashboard.jl RELIANCE 3 --serve  # live, rebuilt per request
#
# Builds a dashboard for one symbol and opens it. The written page is self-contained: no build
# step, no server, no network. It opens from the filesystem and works offline.
#
# The examples get their own environment because HTTP.jl is a weak dependency of the package.
# Running `Pkg.add("HTTP")` inside the package project promotes it to a real one and rewrites
# Project.toml, taking the extension wiring with it.
#
# `--serve` rebuilds the page on every request, which is what you want while something is still
# running. It binds to the loopback interface only.

using BayesTrade, Dates, Printf

function history(symbol::AbstractString, years::Int, synthetic_ok::Bool)
    if haskey(ENV, "GROWW_API_KEY")
        @eval Main using HTTP
        source = Base.invokelatest(connect_groww)
        stop = Date(now(UTC) + IST_OFFSET)
        @printf("fetching %s from Groww, %s to %s\n", symbol, stop - Year(years), stop)
        return Base.invokelatest(
            fetch_bars, source, symbol; start = stop - Year(years), stop = stop,
            interval = "1d",
        )
    end

    # Falling back to a generator when a real ticker was asked for would draw a chart that
    # looks like this company's history and is made up. Refusing is the only safe answer:
    # a page labelled RELIANCE has to be Reliance.
    synthetic_ok || error(
        """
        $symbol needs real history and no credentials are set.

            export GROWW_API_KEY='...'
            export GROWW_API_SECRET='...'

        Run with no symbol for a synthetic demo instead.
        """,
    )
    @printf("no GROWW_API_KEY set, generating synthetic history\n")
    return generate_series(
        AR1Returns(phi = 0.45, annual_drift = 0.06);
        symbol = symbol, n_bars = 252 * years, seed = 3,
        start = Date(2026, 8, 25) - Year(years),
    ).bars
end

function build_payload(symbol::AbstractString, years::Int, synthetic_ok::Bool)
    bars = history(symbol, years, synthetic_ok)
    quality = validate_bars(bars; calendar = nse_calendar())
    println(summarise(quality))

    engine = FeatureEngine(
        InMemoryBarStore(bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, symbol; horizon_bars = 1)
    length(examples) < 300 &&
        error("only $(length(examples)) usable examples; fetch a longer window")

    report = replay(
        (
            () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
            () -> MarketRegimeModel(; horizon_bars = 1),
        ),
        examples; config = ReplayConfig(warmup = 250, refit_every = 25),
    )
    book = Portfolio(
        equity = 1.0e6, cash = 1.0e6, as_of = last(examples).features.as_of,
    )
    return dashboard_payload(
        report; book = book, generated_at = now(UTC) + IST_OFFSET,
    )
end

function main(symbol::String, years::Int, serve::Bool, synthetic_ok::Bool)
    if serve
        # Rebuilt per request, so leaving the tab open and re-running the fit shows the new one.
        @eval Main using HTTP
        server = Base.invokelatest(
            serve_dashboard, () -> build_payload(symbol, years, synthetic_ok);
            title = "$symbol review", refresh_seconds = 30,
        )
        println("serving on http://127.0.0.1:8787/  (ctrl-c to stop)")
        try
            wait(Condition())     # nothing else to do; the server runs on its own tasks
        catch error
            error isa InterruptException || rethrow()
        finally
            close(server)
        end
        return nothing
    end

    path = write_dashboard_page(
        build_payload(symbol, years, synthetic_ok), joinpath(pwd(), "dashboard.html");
        title = "$symbol review",
    )
    @printf("wrote %s (%.0f KB)\n", path, filesize(path) / 1024)
    try
        run(
            Sys.isapple() ? `open $path` :
                Sys.iswindows() ? `cmd /c start $path` : `xdg-open $path`;
            wait = false,
        )
    catch error
        error isa InterruptException && rethrow()
        println("open it yourself: ", path)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    arguments = filter(argument -> !startswith(argument, "--"), ARGS)
    main(
        isempty(arguments) ? "SYNTH" : arguments[1],
        length(arguments) >= 2 ? parse(Int, arguments[2]) : 3,
        "--serve" in ARGS,
        isempty(arguments),      # a generator stands in only when no ticker was named
    )
end
