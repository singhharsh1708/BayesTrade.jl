featurebar(index, close; high = close, symbol = "RELIANCE") = Bar(
    symbol, DateTime(2026, 1, 5, 10, 0) + Day(index),
    close, max(high, close), min(close, high) * 0.99, close, 1000.0,
)

featurewindow(closes; highs = closes) = BarWindow(
    Bar[featurebar(i - 1, closes[i]; high = highs[i]) for i in eachindex(closes)],
)

exponential(n, rate; start = 100.0) = [start * exp(rate * (i - 1)) for i in 1:n]

struct Constant <: Feature
    name::Symbol
    value::Union{Float64, Nothing}
    lookback::Int
end
BayesTrade.feature_name(feature::Constant) = feature.name
BayesTrade.lookback(feature::Constant) = feature.lookback
BayesTrade.compute(feature::Constant, ::BarWindow) = feature.value

@testset "features" begin
    @testset "bar window" begin
        @testset "construction refuses what is not a window" begin
            @test_throws ArgumentError BarWindow(Bar[])
            @test_throws ArgumentError BarWindow(
                [featurebar(0, 100.0), featurebar(1, 101.0; symbol = "TCS")],
            )
            @test_throws ArgumentError BarWindow([featurebar(1, 101.0), featurebar(0, 100.0)])
        end

        @testset "a single bar window has no returns" begin
            window = featurewindow([100.0])
            @test length(window) == 1
            @test isempty(window.log_returns)
        end

        @testset "accessors follow the bar order" begin
            window = featurewindow([100.0, 101.0, 102.0])
            @test current(window).close ≈ 102.0
            @test window_symbol(window) == "RELIANCE"
            @test window.closes == [100.0, 101.0, 102.0]
            @test length(window.log_returns) == 2
        end

        @testset "log returns are additive" begin
            window = featurewindow([100.0, 110.0, 121.0])
            @test sum(window.log_returns) ≈ log(1.21)
        end

        @testset "tail keeps the most recent bars" begin
            window = featurewindow([100.0, 101.0, 102.0, 103.0])
            @test tail(window, 2).closes == [102.0, 103.0]
            @test window_as_of(tail(window, 2)) == window_as_of(window)
            @test length(tail(window, 10)) == 4
            @test_throws ArgumentError tail(window, 0)
        end
    end

    @testset "the contract enforces warm-up, not the implementation" begin
        # Constant would happily return a value with no history.
        feature = Constant(:always, 1.0, 10)
        @test evaluate(feature, featurewindow([100.0])) === nothing
        @test evaluate(feature, featurewindow(fill(100.0, 11))) == 1.0
        @test required_bars(LogReturn(5)) == 6
        @test evaluate(LogReturn(5), featurewindow(fill(100.0, 5))) === nothing
        @test occursin("lookback=5", sprint(show, LogReturn(5)))
    end

    @testset "feature set" begin
        @testset "an empty or duplicated set is rejected" begin
            @test_throws ArgumentError FeatureSet(Feature[])
            @test_throws ArgumentError FeatureSet([SimpleReturn(5), SimpleReturn(5)])
        end

        @testset "column order is declaration order, not alphabetical" begin
            set = FeatureSet([SimpleReturn(20), LogReturn(1), SimpleReturn(5)])
            @test columns(set) == [:return_20, :log_return_1, :return_5]
        end

        @testset "the warm-up is the longest in the set" begin
            set = FeatureSet([LogReturn(1), SimpleReturn(20), LogReturn(5)])
            @test lookback(set) == 20
            @test required_bars(set) == 21
        end

        @testset "warm-up is recorded as missing, never as a zero" begin
            set = FeatureSet([LogReturn(1), SimpleReturn(20)])
            vector = compute(set, featurewindow([100.0 + i for i in 0:4]))
            @test feature_names(vector) == [:log_return_1]
            @test vector.missing_features == [:return_20]
            @test !is_complete(vector)
            @test vector.n_bars == 5
        end

        @testset "a fully warmed set is complete" begin
            set = FeatureSet([LogReturn(1), SimpleReturn(3)])
            vector = compute(set, featurewindow([100.0 + i for i in 0:9]))
            @test is_complete(vector)
            @test Set(feature_names(vector)) == Set([:log_return_1, :return_3])
        end

        @testset "subsetting and concatenation" begin
            set = FeatureSet([LogReturn(1), SimpleReturn(5), SimpleReturn(20)])
            @test columns(subset(set, [:return_5])) == [:return_5]
            @test_throws KeyError subset(set, [:nope])
            @test columns(vcat(FeatureSet([LogReturn(1)]), FeatureSet([SimpleReturn(5)]))) ==
                [:log_return_1, :return_5]
        end

        @testset "a feature returning a non-finite value fails loudly" begin
            set = FeatureSet([Constant(:broken, NaN, 0)])
            @test_throws ArgumentError compute(set, featurewindow([100.0, 101.0]))
        end
    end

    @testset "feature vector" begin
        vector() = FeatureVector(
            symbol = "RELIANCE", as_of = DateTime(2026, 1, 5),
            data_as_of = DateTime(2026, 1, 5),
            values = Dict(:return_5 => 0.02, :log_return_1 => 0.001),
            missing_features = [:return_20], n_bars = 10,
        )

        @testset "require distinguishes warming up from never requested" begin
            @test_throws KeyError require(vector(), :return_20)
            @test_throws KeyError require(vector(), :nonsense)
            @test require(vector(), :return_5) ≈ 0.02
        end

        @testset "get falls back without raising" begin
            @test get(vector(), :return_20, nothing) === nothing
            @test get(vector(), :return_20, 0.0) == 0.0
            @test haskey(vector(), :return_5)
        end

        @testset "the design row follows the requested order" begin
            @test design_row(vector(), [:log_return_1, :return_5]) == [0.001, 0.02]
            @test_throws KeyError design_row(vector(), [:return_5, :return_20])
        end

        @testset "a value cannot be both present and missing" begin
            @test_throws ArgumentError FeatureVector(
                symbol = "R", as_of = DateTime(2026, 1, 5),
                data_as_of = DateTime(2026, 1, 5),
                values = Dict(:a => 1.0), missing_features = [:a], n_bars = 1,
            )
        end

        @testset "non-finite values are rejected" begin
            @test_throws ArgumentError FeatureVector(
                symbol = "R", as_of = DateTime(2026, 1, 5),
                data_as_of = DateTime(2026, 1, 5),
                values = Dict(:a => Inf), n_bars = 1,
            )
        end

        @testset "subset keeps missing status" begin
            narrowed = subset(vector(), [:return_5, :return_20])
            @test feature_names(narrowed) == [:return_5]
            @test narrowed.missing_features == [:return_20]
        end
    end

    @testset "price features" begin
        @testset "log and simple returns match their closed forms" begin
            window = featurewindow([100.0, 110.0, 121.0])
            @test evaluate(LogReturn(1), window) ≈ 0.09531018 atol = 1.0e-7
            @test evaluate(LogReturn(2), window) ≈ 0.19062036 atol = 1.0e-7
            @test evaluate(SimpleReturn(1), window) ≈ 0.1
            @test evaluate(SimpleReturn(2), window) ≈ 0.21
        end

        @testset "log returns compose where simple ones do not" begin
            window = featurewindow([100.0, 150.0, 75.0])
            @test evaluate(LogReturn(1), featurewindow([100.0, 150.0])) +
                evaluate(LogReturn(1), window) ≈ evaluate(LogReturn(2), window)
            @test evaluate(SimpleReturn(2), window) ≈ -0.25
        end

        @testset "a zero horizon is rejected" begin
            @test_throws ArgumentError LogReturn(0)
            @test_throws ArgumentError SimpleReturn(0)
        end
    end
end
