@testset "Online PCA" begin
    C =   [0.65  -0.09   1.23   0.5;
           -0.09   0.19  -0.42  -0.16;
           1.23  -0.42   3.18   1.36;
           0.5   -0.16   1.36   0.63]
    U = cholesky(C).U
    m = [5.84; 3.06; 3.78; 1.22]

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

    opca = OnlinePCA(dims);
    for n in axes(x, 2)
        update!(opca, reshape(x[:, n], :, 1))
    end

    @test opca.n[1] == N
    @test isapprox(opca.m, mean(x, dims = 2))

    pca = [-0.88; 0.31; -0.36; -0.03]
    s = sign.(pca) ./ sign.(opca.pc)
    @test isapprox(s .* opca.pc ./ norm(opca.pc), pca, atol = 1e-1)

    x = mvrandn(N, m, U)

    opca = OnlinePCA(dims);
    for n in 1:div(N, 4)
        b = 4 * (n - 1) + 1
        e = 4 * n
        idx = b:e
        update!(opca, x[:, idx])
    end

    @test opca.n[1] == N
    @test isapprox(opca.m, mean(x, dims = 2))

    pca = [-0.88; 0.31; -0.36; -0.03]
    s = sign.(pca) ./ sign.(opca.pc)
    @test isapprox(s .* opca.pc ./ norm(opca.pc), pca, atol = 1e-1)
end
