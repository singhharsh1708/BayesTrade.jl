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
end
