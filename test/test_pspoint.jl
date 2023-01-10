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
