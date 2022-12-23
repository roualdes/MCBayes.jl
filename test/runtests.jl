using MCBayes
using Test
using Statistics

@testset "Online Moments" begin
    N, dims, chains = 100, 10, 4
    x = randn(N, dims, chains)

    m = reshape(mean(x; dims=1), (dims, chains))
    v = reshape(var(x; dims=1, corrected=false), (dims, chains))

    # separate moments for each chain of x
    om = OnlineMoments(dims, chains)

    for n in 1:N
        MCBayes.update!(om, x[n, :, :])
    end

    @test om.n[1] == N
    @test isapprox(om.m, m)
    @test isapprox(om.v, v)
    @test isapprox(MCBayes.optimum(om; regularized=false), om.v)
    w = reshape(om.n ./ (om.n .+ 5), 1, :)
    v = @. w * om.v + (1 - w) * 1e-3
    @test isapprox(MCBayes.optimum(om), v)

    MCBayes.reset!(om)
    @test iszero(om.n)
    @test iszero(om.m)
    @test iszero(om.v)

    m = reshape(mean(x; dims=(1, 3)), dims)
    v = reshape(var(x; dims=(1, 3), corrected=false), dims)

    # same moments for all chains of x
    om = OnlineMoments(dims)

    for n in 1:N
        MCBayes.update!(om, x[n, :, :])
    end

    @test om.n[1] == N * chains
    @test isapprox(om.m, m)
    @test isapprox(om.v, v)
    @test isapprox(MCBayes.optimum(om; regularized=false), om.v)
    w = reshape(om.n ./ (om.n .+ 5), 1, :)
    v = @. w * om.v + (1 - w) * 1e-3
    @test isapprox(MCBayes.optimum(om), v)
end

@testset "Phase space point" begin
    @test_throws TypeError MCBayes.PSPoint([1;], [2;])
    @test_throws ErrorException MCBayes.PSPoint([1.0; 2.0], [3.0;])

    z = MCBayes.PSPoint([1.0; 2.0; 3.0], [4.0; 5.0; 6.0])

    @test isequal(size(z), (6,))
    @test isequal(length(z), 6)

    @test isapprox(z[1], 1.0)
    @test isapprox(z[6], 6.0)

    zz = copy(z)
    @test isequal(typeof(z), typeof(zz))
    @test isapprox(z, zz)

    zzz = similar(z)
    @test isequal(typeof(z), typeof(zz))
    @test isequal(size(zzz), (6,))
    @test isequal(length(zzz), 6)

    zzz .= z
    @test isapprox(z, zzz)
end

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
end
