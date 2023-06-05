@testset "Online PCA" begin
    C = [
        0.65 -0.09 1.23 0.5
        -0.09 0.19 -0.42 -0.16
        1.23 -0.42 3.18 1.36
        0.5 -0.16 1.36 0.63
    ]
    U = cholesky(C).U
    m = [5.84; 3.06; 3.78; 1.22]
    pca = [-0.88; 0.31; -0.36; -0.03]

    N = 5_000
    dims = length(m)

    function mvrandn(N, m, U::AbstractMatrix)
        D = length(m)
        x = Matrix{eltype(m)}(undef, D, N)
        for n in 1:N
            x[:, n] .= U * randn(D, 1) .+ m
        end
        return x
    end

    x = mvrandn(N, m, U)

    opca = MCBayes.OnlinePCA(dims)
    om = MCBayes.OnlineMoments(dims, 1)
    for n in axes(x, 2)
        y = reshape(x[:, n], :, 1)
        MCBayes.update!(om, y)
        MCBayes.update!(opca, y .- om.m)
    end

    @test opca.n[1] == N

    s = sign.(pca) ./ sign.(opca.pc)
    @test isapprox(s .* opca.pc ./ norm(opca.pc), pca, atol=1e-1)

    MCBayes.reset!(opca)
    @test all(opca.pc .!= 0)
    @test opca.n[1] == 0

    om = MCBayes.OnlineMoments(dims, 1)
    for n in 1:div(N, 4)
        b = 4 * (n - 1) + 1
        e = 4 * n
        idx = b:e
        y = x[:, idx]
        MCBayes.update!(om, y)
        MCBayes.update!(opca, y .- om.m)
    end

    @test opca.n[1] == N
    s = sign.(pca) ./ sign.(opca.pc)
    @test isapprox(s .* opca.pc ./ norm(opca.pc), pca, atol=1e-1)
end
