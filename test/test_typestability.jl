@testset "Type stability" begin
    function ld(x)
        return -x' * x / 2
    end
    function ldg(x)
        return ld(x), -x
    end

    # since Float64 is default in most cases,
    # I'm satisfied to test only against Float32
    T = Float32
    dims = 10
    chains = 4
    iterations = 10

    @test isequal(eltype(StepsizeConstant(fill(0.6f0, chains))), T)
    @test isequal(eltype(StepsizeDualAverage(fill(0.5f0, chains))), T)

    @test isequal(eltype(MetricOnlineMoments(fill(1.0f0, dims, chains))), T)
    @test isequal(eltype(MetricConstant(fill(1.0f0, dims, chains))), T)
    @test isequal(eltype(MetricFisherDivergence(fill(1.0f0, dims, chains))), T)

    stan = Stan(dims, chains, T)
    draws, diagnostics, rngs = sample!(stan, ldg; iterations)
    @test isequal(eltype(draws), T)
    @test isequal(eltype(stan.metric), T)
    @test isequal(eltype(stan.stepsize), T)

    meads = MEADS(dims, chains, 32, T)
    draws, diagnostics, rngs = sample!(meads, ldg; iterations)
    @test isequal(eltype(draws), T)
    @test isequal(eltype(meads.metric), T)
    @test isequal(eltype(meads.stepsize), T)

    rwm = RWM(dims, chains, T)
    draws, diagnostics, rngs = sample!(rwm, ld; iterations)
    @test isequal(eltype(draws), T)
    @test isequal(eltype(rwm.metric), T)
    @test isequal(eltype(rwm.stepsize), T)

    mala = MALA(dims, chains, T)
    draws, diagnostics, rngs = sample!(mala, ldg; iterations)
    @test isequal(eltype(draws), T)
    @test isequal(eltype(mala.metric), T)
    @test isequal(eltype(mala.stepsize), T)
end
