using BayesTrade
using Dates
using Distributions
using Random
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
end
