using MCBayes
using Test
using Statistics

@testset "Online moments" begin
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

@testset "ESS, Rhat, and MCSE" begin
    x = [ 0.239407691 -0.62282776  0.24356776 -1.10368191;
          -1.507900223 -0.97702220 -0.49290339 -0.04632925;
          -1.421186419 -0.73568277  0.04301316 -0.15462371;
          -0.770254789 -1.32197653 -0.71799675 -0.05134798;
          0.003243823 -0.05474649 -1.00474817  0.52814719;
          2.118660239  0.29459794 -1.15549545 -0.27780754;
          -0.548916260 -0.14347794  1.47567766  0.16420936;
          -0.847931550  0.11253483  1.66249639  2.36078238;
          0.847943703 -0.44757236 -1.08864716  0.92910897;
          0.699827332  0.50092564 -0.30425246 -0.32194831;
          -1.070388865  0.69792782 -0.31303889 -0.29153897;
          0.688964769  0.08287254  0.19910818  0.15701755;
          -0.782429086  1.67931003 -0.14829945 -0.22959858;
          -0.925853513 -0.85916121  0.04880209 -0.04997852;
          0.391888329 -0.87161484 -0.03473023  0.93634380;
          0.016131943 -1.79682200 -0.84479692 -0.98430272;
          0.276388510 -1.50788909 -1.52233697 -1.48942141;
          -0.433214020  0.57052300  0.88867629 -0.53208358;
          -0.682587678  0.41141717  0.62939699  0.04428697;
          1.798689555  0.38975660 -0.72060527  1.12196227]
    draws = reshape(x, 20, 1, 4)

    # values taken from stan-dev/posterior
    # @test isapprox(ess_bulk(draws)[1], 73.6762; atol = 1e-5)
    @test isapprox(ess_tail(draws)[1], 65.7112; atol = 1e-5)
    @test isapprox(ess_quantile(draws, 0.75)[1], 89.15154; atol = 1e-5)
    @test isapprox(ess_mean(draws)[1], 75.95232; atol = 1e-5)
    @test isapprox(ess_std(draws)[1], 68.86159; atol = 1e-5)
    # @test isapprox(rhat(draws)[1],  1.009791; atol = 1e-5)
    @test isapprox(rhat_basic(draws)[1], 0.9629473; atol = 1e-5)
    @test isapprox(mcse_mean(draws)[1], 0.1010431; atol = 1e-5)
    @test isapprox(mcse_std(draws)[1], 0.07730264; atol = 1e-5)

    # not enough draws
    @test isnan(ess_mean(draws[1:2, :, :])[1])
    @test isnan(rhat_basic(draws[1:2, :, :])[1])

    # NaN in draws
    draws[1, 1, 1] = NaN
    @test isnan(ess_mean(draws)[1])
    @test isnan(rhat_basic(draws)[1])
end
