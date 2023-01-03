@testset "Stan sampling" verbose=true begin
    # models and values from stan-dev/posteriordb
    expectations = Dict(
        "illconditioned_mvnormal" =>
            Dict(:true_mean => zeros(10), :true_std => sqrt.(10.0 .^ [-2:7;])),
        "studentt" => Dict(:true_mean => zeros(30), :true_std => sqrt.(fill(5 / 3, 30))),
        "arK-arK" => Dict(
            :true_mean => [
                -0.000718650251261263
                0.692163279812727
                0.439043080115602
                0.105816025140126
                -0.0354350382459401
                -0.301512065609031
                0.150566659032913
            ],
        ),
        "gp_pois_regr-gp_regr" => Dict(:true_mean => [
            6.87434817782581
            2.4423998398757
            1.82873106572797
        ]),
        "highd_mvnormal" => Dict(:true_mean => zeros(1000), :true_std => ones(1000)),
        "arma-arma11" => Dict(
            :true_mean => [
                0.00691486351224137
                0.957012888586847
                -0.0336960218471
                0.166481622433993
            ],
        ),
        "garch-garch11" => Dict(
            :true_mean => [
                5.05001794660039
                1.47075973803898
                0.567284282813872
                0.293024546082117
            ],
        ),
        "mesquite-logmesquite" => Dict(
            :true_mean => [
                5.35036438720819
                0.398570465445401
                1.1491959478628
                0.377209789143197
                0.390044355818078
                0.109251053163576
                -0.584668727994285
                0.340679741485552
            ],
        ),
    )

    modeldir = joinpath(artifact"test_models", "test_models")
    model_names = [f for f in readdir(modeldir) if isdir(joinpath(modeldir, f))]
    warmup = 2000
    iterations = 5000

    function prepare_model(model_name)
        modeldir = joinpath(artifact"test_models", "test_models", model_name)
        stan_file = joinpath(modeldir, model_name * ".stan")
        stan_data = joinpath(modeldir, model_name * ".json")
        bsm = BS.StanModel(; stan_file=stan_file, data=stan_data)

        function ldg(q)
            return try
                BS.log_density_gradient(bsm, q)
            catch
                (typemin(eltype(q)), zero(q))
            end
        end

        return bsm, ldg
    end

    @testset "arK-arK" begin
        model_name = model_names[1]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        draws, diagnostics, rngs = sample!(stan, ldg; warmup=2000, iterations=5000)

        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        @test all(m .- 5 .* err_mean .< true_m .< m .+ 5 .* err_mean)
    end

    @testset "arma-arma11" begin
        model_name = model_names[2]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        draws, diagnostics, rngs = sample!(stan, ldg; warmup=2000, iterations=5000)
        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        @test all(m .- 5 .* err_mean .< true_m .< m .+ 5 .* err_mean)
    end

    @testset "garch-garch11" begin
        model_name = model_names[3]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        draws, diagnostics, rngs = sample!(stan, ldg; warmup=2000, iterations=5000)
        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        @test all(m .- 5 .* err_mean .< true_m .< m .+ 5 .* err_mean)
    end

    @testset "gp_pois_regr-gp_pois_regr" begin
        model_name = model_names[4]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        ssda = StepsizeDualAverage(ones(4); δ=0.99)
        draws, diagnostics, rngs = sample!(
            stan, ldg; stepsize_adapter=ssda, warmup=2000, iterations=5000
        )
        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        @test all(m .- 5 .* err_mean .< true_m .< m .+ 5 .* err_mean)
    end

    @testset "highd_mvnormal" begin
        model_name = model_names[5]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        draws, diagnostics, rngs = sample!(stan, ldg; warmup=2000, iterations=5000)
        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        @test all(m .- 5 .* err_mean .< true_m .< m .+ 5 .* err_mean)

        s = std(constrained_draws; dims=(1, 3))[:]
        err_std = mcse_std(constrained_draws)[:]

        true_s = expectations[model_name][:true_std]
        @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
    end

    @testset "illconditioned_mvnormal" begin
        model_name = model_names[6]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        draws, diagnostics, rngs = sample!(stan, ldg; warmup=2000, iterations=5000)
        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        @test all(m .- 5 .* err_mean .< true_m .< m .+ 5 .* err_mean)

        s = std(constrained_draws; dims=(1, 3))[:]
        err_std = mcse_std(constrained_draws)[:]

        true_s = expectations[model_name][:true_std]
        @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
    end

    @testset "mesquite-logmesquite" begin
        model_name = model_names[7]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        ssda = StepsizeDualAverage(ones(4); δ=0.99)
        draws, diagnostics, rngs = sample!(
            stan, ldg; stepsize_adapter=ssda, warmup=2000, iterations=5000
        )
        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        @test all(m .- 5 .* err_mean .< true_m .< m .+ 5 .* err_mean)
    end

    @testset "Student-t" begin
        model_name = model_names[8]
        bsm, ldg = prepare_model(model_name)
        stan = Stan(BS.param_unc_num(bsm))
        draws, diagnostics, rngs = sample!(stan, ldg; warmup=2000, iterations=5000)
        constrained_draws = mapslices(
            q -> BS.param_constrain(bsm, q; include_tp=true), draws[2001:end, :, :]; dims=2
        )
        m = mean(constrained_draws; dims=(1, 3))[:]
        err_mean = mcse_mean(constrained_draws)[:]

        true_m = expectations[model_name][:true_mean]
        emp_m = m[(end - 29):end]
        emp_err_mean = err_mean[(end - 29):end]
        @test all(emp_m .- 5 .* emp_err_mean .< true_m .< emp_m .+ 5 .* emp_err_mean)

        s = std(constrained_draws; dims=(1, 3))[:]
        err_std = mcse_std(constrained_draws)[:]

        true_s = expectations[model_name][:true_std]
        emp_s = s[(end - 29):end]
        emp_err_std = err_std[(end - 29):end]
        @test all(emp_s .- 5 .* emp_err_std .< true_s .< emp_s .+ 5 .* emp_err_std)
    end
end
