# Package hygiene and static analysis. These run first: a package that does not load
# cleanly under Aqua or JET has a problem that will surface as a confusing failure
# somewhere else, and finding it here is cheaper.

using Aqua
using JET

@testset "package quality" begin
    @testset "Aqua" begin
        Aqua.test_all(BayesTrade; ambiguities = false)
        Aqua.test_ambiguities(BayesTrade)
    end

    @testset "JET" begin
        JET.test_package(BayesTrade; target_defined_modules = true)
    end
end
