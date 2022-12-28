@testset "Type stability" begin
    function ldg(x)
        -x' * x / 2, -x
    end

    # since Float64 is default in most cases,
    # I'm satisfied to test only against Float32
    T = Float32
    dims = 10
    chains = 4

    @test isequal(eltype(StepsizeConstant(fill(0.6f0, chains))), T)
    @test isequal(eltype(StepsizeDualAverage(fill(0.5f0, chains))), T)

    @test isequal(eltype(MetricOnlineMoments(fill(1.0f0, dims, chains))), T)
    @test isequal(eltype(MetricConstant(fill(1.0f0, dims, chains))), T)

    stan = Stan(dims, chains, T)
    draws, diagnostics, rngs = sample!(stan, ldg)
    @test isequal(eltype(draws), T)
    @test isequal(eltype(stan.metric), T)
    @test isequal(eltype(stan.stepsize), T)
end
