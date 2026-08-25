# The page has one property that matters more than how it looks: it opens from the filesystem,
# offline, with no build step. Everything here is a way of checking it still does.

function page_payload(symbol = "SYNTH"; n_bars = 700)
    series = generate_series(
        AR1Returns(phi = 0.4, annual_drift = 0.05);
        symbol = symbol, n_bars = n_bars, seed = 5, start = Date(2024, 1, 2),
    )
    engine = FeatureEngine(
        InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, symbol; horizon_bars = 1)
    report = replay(
        (
            () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
        ),
        examples; config = ReplayConfig(warmup = 400, refit_every = 50),
    )
    book = Portfolio(equity = 1.0e6, cash = 1.0e6, as_of = last(examples).features.as_of)
    return dashboard_payload(
        report; book = book, generated_at = DateTime(2026, 8, 25, 14, 30),
    )
end

@testset "the dashboard page" begin
    payload = page_payload()

    @testset "it reaches nothing outside itself" begin
        # A CDN request is a page that is blank on a plane, and a page that is blank on a plane
        # is one nobody opens while debugging.
        page = dashboard_page(payload)
        for reference in ("http://", "https://", "<script src", "<link rel=\"stylesheet\"")
            @test !occursin(reference, page)
        end
        @test count("<script", page) == 2         # the payload, and the code that draws it
        @test count("</script>", page) == 2
        @test startswith(page, "<!doctype html>")
        @test occursin("</html>", page)
    end

    @testset "the payload survives the round trip" begin
        page = dashboard_page(payload)
        block = match(
            r"<script id=\"payload\" type=\"application/json\">(.*?)</script>"s, page,
        )
        @test block !== nothing
        parsed = JSON3.read(block.captures[1], Dict{String, Any})
        @test parsed["symbol"] == payload["symbol"]
        @test parsed["n_scored"] == payload["n_scored"]
        @test length(parsed["series"]) == length(payload["series"])
        @test length(parsed["decisions"]) == length(payload["decisions"])
        @test haskey(parsed["calibration"], "coverage")
    end

    @testset "a closing script tag in the data cannot end the tag" begin
        # `</script>` ends the tag wherever it appears, including inside a JSON string. The
        # browser then parses the rest of the payload as HTML.
        hostile = Dict{String, Any}(
            "schema" => 1, "symbol" => "</script><img src=x onerror=alert(1)>",
            "series" => Any[], "decisions" => Any[], "models" => String[],
            "calibration" => Dict{String, Any}("coverage" => Any[]),
            "limits" => Dict{String, Any}(), "n_scored" => 0, "n_examples" => 0,
            "generated_at" => "2026-08-25T00:00:00",
        )
        page = dashboard_page(hostile)
        block = match(
            r"<script id=\"payload\" type=\"application/json\">(.*?)</script>"s, page,
        )
        # The tag cannot be closed early. What is inside a JSON script block is inert markup
        # until something ends the block, so this is the property that matters.
        @test count("</script>", page) == 2
        @test !occursin("</script", block.captures[1])
        # And the symbol reaches the title escaped rather than as markup.
        @test !occursin("<img src=x", page[1:something(findfirst("<style>", page)).start])
        # It still parses, and the symbol survives intact rather than being mangled.
        @test JSON3.read(block.captures[1], Dict{String, Any})["symbol"] == hostile["symbol"]
    end

    @testset "the title is the symbol, and it is escaped" begin
        @test occursin("<title>SYNTH review</title>", dashboard_page(payload))
        @test occursin(
            "<title>my &lt;report&gt;</title>",
            dashboard_page(payload; title = "my <report>"),
        )
        @test occursin("&amp;lt;", dashboard_page(payload; title = "&lt;"))
    end

    @testset "a refresh is opt in" begin
        @test !occursin("http-equiv=\"refresh\"", dashboard_page(payload))
        page = dashboard_page(payload; refresh_seconds = 15)
        @test occursin("<meta http-equiv=\"refresh\" content=\"15\">", page)
        @test_throws ArgumentError dashboard_page(payload; refresh_seconds = -1)
    end

    @testset "it writes where a browser can open it" begin
        mktempdir() do dir
            path = write_dashboard_page(payload, joinpath(dir, "nested", "d.html"))
            @test isfile(path)
            @test isabspath(path)
            @test filesize(path) > 20_000
            @test occursin("SYNTH", read(path, String))
        end
    end

    @testset "the core stubs explain themselves" begin
        # These are what a caller hits with no HTTP loaded. The extension defines narrower
        # methods, so reaching the stub here means calling with a shape it does not cover, and
        # the message is the same one somebody without the extension would read. A MethodError
        # would be technically correct and useless.
        for call in (
                () -> serve_dashboard(1, 2, 3),
                () -> groww_transport(1, 2, 3),
                () -> connect_groww(1, 2, 3),
            )
            failure = try
                call()
                nothing
            catch error
                error
            end
            @test failure isa ArgumentError
            @test occursin("using HTTP", failure.msg)
            @test occursin("package extension", failure.msg)
        end
    end
end

@testset "the dashboard rules the way the live path rules" begin
    # The panel is labelled as every bar the system acted on or refused to, with the gate that
    # stopped it. A ruling computed with fewer gates than the session uses is a different
    # ruling, and a weaker one, so the panel would be quietly optimistic.
    payload = page_payload()
    @test !isempty(payload["decisions"])

    series = generate_series(
        AR1Returns(phi = 0.4, annual_drift = 0.05);
        symbol = "GATE", n_bars = 700, seed = 5, start = Date(2024, 1, 2),
    )
    engine = FeatureEngine(
        InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, "GATE"; horizon_bars = 1)
    report = replay(
        (() -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),),
        examples; config = ReplayConfig(warmup = 400, refit_every = 50),
    )
    book = Portfolio(equity = 1.0e6, cash = 1.0e6, as_of = last(examples).features.as_of)

    # A volatility ceiling tight enough to bite has to bite in the payload too.
    tight = RiskLimits(max_annualised_volatility = 0.01)
    payload = dashboard_payload(
        report; limits = tight, book = book,
        generated_at = DateTime(2026, 8, 25, 12),
    )
    actionable = [
        row for row in payload["decisions"] if row["action"] in ("buy", "sell")
    ]
    @test !isempty(actionable)
    @test all(row -> "volatility" in row["failures"], actionable)
    @test all(row -> row["approved"] == 0.0, actionable)

    # And with a limit nothing could breach, the gate is present and passing rather than
    # absent, so a reader can tell it ran.
    loose = dashboard_payload(
        report; limits = RiskLimits(max_annualised_volatility = 50.0), book = book,
        generated_at = DateTime(2026, 8, 25, 12),
    )
    relaxed = [row for row in loose["decisions"] if row["action"] in ("buy", "sell")]
    @test !isempty(relaxed)
    @test all(row -> !("volatility" in row["failures"]), relaxed)
end
