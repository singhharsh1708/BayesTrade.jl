using BayesTrade
using Dates
using Distributions
using JSON3
using LinearAlgebra
using Random
using SpecialFunctions
using Statistics
using Test

@testset "BayesTrade" begin
    include("test_quality.jl")
    include("test_enums.jl")
    include("test_market.jl")
    include("test_probabilistic.jl")
    include("test_limits.jl")
    include("test_settings.jl")
    include("test_model_interface.jl")
    include("test_calendar.jl")
    include("test_processes.jl")
    include("test_synthetic.jl")
    include("test_store.jl")
    include("test_csv_io.jl")
    include("test_data_quality.jl")
    include("test_sources.jl")
    include("test_features.jl")
    include("test_momentum.jl")
    include("test_engine.jl")
    include("test_volatility_features.jl")
    include("test_volume_features.jl")
    include("test_labels.jl")
    include("test_linear.jl")
    include("test_variance_filter.jl")
    include("test_regime_filter.jl")
    include("test_return_model.jl")
    include("test_volatility_model.jl")
    include("test_scaled_return_model.jl")
    include("test_regime_model.jl")
    include("test_fusion.jl")
    include("test_replay.jl")
    include("test_risk.jl")
    include("test_broker.jl")
    include("test_calibration.jl")
    include("test_walk_forward.jl")
    include("test_persistence.jl")
    include("test_leakage.jl")
end
