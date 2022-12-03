using MCBayes
using Test
using Statistics

@testset "Online Moments" begin

    N, dims, chains = 100, 10, 4
    x = randn(N, dims, chains)

    m = reshape(mean(x, dims = 1), (dims, chains))
    v = reshape(var(x, dims = 1; corrected = false), (dims, chains))

    # separate moments for each chain of x
    om = OnlineMoments(dims, chains)

    for n in 1:N
        update!(om, x[n, :, :])
    end

    @test om.n[1] == N
    @test isapprox(om.m, m)
    @test isapprox(om.v, v)
    @test isapprox(metric(om; regularized = false), om.v)
    w = om.n ./ (om.n .+ 5)
    v = @. w' * om.v + (1 - w') * 1e-3
    @test isapprox(metric(om), v)

    reset!(om)
    @test iszero(om.n)
    @test iszero(om.m)
    @test iszero(om.v)

    m = reshape(mean(x, dims = (1, 3)), dims)
    v = reshape(var(x, dims = (1, 3); corrected = false), dims)

    # same moments for all chains of x
    om = OnlineMoments(dims, 1)

    for n in 1:N
        update!(om, x[n, :, :])
    end

    @test om.n[1] == N * chains
    @test isapprox(om.m, m)
    @test isapprox(om.v, v)
    @test isapprox(metric(om; regularized = false), om.v)
    w = om.n ./ (om.n .+ 5)
    v = @. w' * om.v + (1 - w') * 1e-3
    @test isapprox(metric(om), v)

end
