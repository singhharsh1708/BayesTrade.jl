# The dashboard payload. The contract is JSON and nothing else.

function db_report(; n_bars = 1_600, seed = 11, warmup = 800)
    series = generate_series(
        RegimeSwitchingReturns(); symbol = "RELIANCE", n_bars = n_bars, seed = seed,
        start = Date(2019, 1, 1),
    )
    engine = FeatureEngine(
        InMemoryBarStore(series.bars),
        FeatureSet(Feature[LogReturn(1), RealisedVolatility(20)]),
    )
    examples = build_training_set(engine, "RELIANCE"; horizon_bars = 1)
    return replay(
        (
            () -> BayesianReturnModel([:log_return_1]; horizon_bars = 1),
            () -> BayesianVolatilityModel(; horizon_bars = 1),
            () -> MarketRegimeModel(; horizon_bars = 1),
        ),
        examples; config = ReplayConfig(warmup = warmup, refit_every = 100),
    )
end

db_book() = Portfolio(
    equity = 1.0e6, cash = 1.0e6, as_of = DateTime(2026, 1, 2),
)

@testset "dashboard payload" begin
    report = db_report()
    payload = dashboard_payload(
        report; book = db_book(), sector = "energy",
        generated_at = DateTime(2026, 8, 25, 10),
    )

    @testset "it carries what the dashboard draws" begin
        @test payload["schema"] == DASHBOARD_SCHEMA_VERSION
        @test payload["symbol"] == "RELIANCE"
        @test payload["n_scored"] == length(report.records)
        @test length(payload["series"]) == length(report.records)
        @test length(payload["models"]) == 3
        @test sum(payload["reliabilities"]) ≈ 1.0
        @test length(payload["decisions"]) == length(report.records)
    end

    @testset "it carries the uncomfortable half" begin
        # Any dashboard can draw an equity curve. These are the numbers that say whether
        # one means anything.
        calibration = payload["calibration"]
        @test calibration["n"] == length(report.records)
        @test 0 <= calibration["interval_error"] < 1
        @test calibration["brier_score"] ≈ 0.25 atol = 0.02
        @test isfinite(calibration["mean_log_score"])
        @test calibration["overconfident"] isa Bool
        @test !isempty(calibration["coverage"])
        for point in calibration["coverage"]
            @test 0 < point["level"] < 1
            @test 0 <= point["empirical"] <= 1
            @test point["error"] ≈ point["empirical"] - point["level"]
        end

        first_point = first(payload["series"])
        @test haskey(first_point, "epistemic_share")
        @test haskey(first_point, "disagreement")
        @test 0 <= first_point["epistemic_share"] <= 1
        @test length(first_point["weights"]) == 3
        @test first_point["lower"] < first_point["mean"] < first_point["upper"]
    end

    @testset "every refusal names its gate" begin
        # Declining is a first-class outcome, so a payload that hid the reason would hide
        # the most common thing the system does.
        for decision in payload["decisions"]
            @test decision["approved"] <= decision["requested"] + 1.0e-12
            if decision["action"] == "no_trade"
                @test decision["reason"] !== nothing
            else
                @test decision["reason"] === nothing
            end
        end
        reasons = unique(
            String[d["reason"] for d in payload["decisions"] if d["reason"] !== nothing],
        )
        @test !isempty(reasons)
    end

    @testset "no Julia type crosses the line" begin
        # The whole point of the contract: the dashboard can be rewritten in anything.
        text = JSON3.write(payload)
        @test length(text) > 1_000
        restored = JSON3.read(text, Dict{String, Any})
        @test restored["symbol"] == "RELIANCE"
        @test restored["schema"] == DASHBOARD_SCHEMA_VERSION
        @test length(restored["series"]) == length(payload["series"])

        mktempdir() do dir
            path = write_dashboard(payload, joinpath(dir, "nested", "payload.json"))
            @test isfile(path)
            written = JSON3.read(read(path, String), Dict{String, Any})
            @test written["n_scored"] == payload["n_scored"]
            @test occursin('\n', read(path, String))
        end
    end

    @testset "a payload without a portfolio carries no decisions" begin
        # The decision engine needs an account to rule against. Inventing one would put a
        # ruling in the record that was never made.
        bare = dashboard_payload(report; generated_at = DateTime(2026, 8, 25, 10))
        @test isempty(bare["decisions"])
        @test !isempty(bare["series"])
        @test bare["calibration"]["n"] == length(report.records)
    end

    @testset "an empty replay is refused rather than drawn" begin
        empty = ReplayReport(
            "RELIANCE", ReplayRecord[], report.reliability, report.calibration, 0, 0,
        )
        @test_throws ArgumentError dashboard_payload(
            empty; generated_at = DateTime(2026, 8, 25, 10),
        )
    end
end
