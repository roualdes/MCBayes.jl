@testset "Online moments" begin
    N, dims, chains = 100, 10, 4
    x = randn(N, dims, chains)

    m = reshape(mean(x; dims=1), (dims, chains))
    v = reshape(var(x; dims=1, corrected=false), (dims, chains))

    # separate moments for each chain of x
    om = MCBayes.OnlineMoments(dims, chains)

    for n in 1:N
        MCBayes.update!(om, x[n, :, :])
    end

    @test om.n[1] == N
    @test isapprox(om.m, m)
    @test isapprox(om.v, v)

    MCBayes.reset!(om)
    @test iszero(om.n)
    @test iszero(om.m)
    @test iszero(om.v)

    # same moments for all chains of x
    m = reshape(mean(x; dims=(1, 3)), dims)
    v = reshape(var(x; dims=(1, 3), corrected=false), dims)
    om = MCBayes.OnlineMoments(dims)

    for n in 1:N
        MCBayes.update!(om, x[n, :, :])
    end

    @test om.n[1] == N * chains
    @test isapprox(om.m, m)
    @test isapprox(om.v, v)
end
