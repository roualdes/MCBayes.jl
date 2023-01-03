@testset "samplers" begin
    modeldir = joinpath(artifact"test_models", "test_models")

    # models and values from stan-dev/posteriordb
    expectations = open(deserialize, joinpath(modeldir, "expectations.jls"))

    model_names = [f for f in readdir(modeldir) if isdir(joinpath(modeldir, f))]

    function prepare_model(model_name)
        modeldir = joinpath(artifact"test_models", "test_models", model_name)
        stan_file = joinpath(modeldir, model_name * ".stan")
        stan_data = joinpath(modeldir, model_name * ".json")
        bsm = BS.StanModel(; stan_file=stan_file, data=stan_data)
        return bsm
    end

    function prepare_log_density_gradient(bridgestan_model)
        return function ldg(q)
            return try
                BS.log_density_gradient(bridgestan_model, q)
            catch
                (typemin(eltype(q)), zero(q))
            end
        end
    end

    function prepare_log_density(bridgestan_model)
        return function ld(q)
            return try
                BS.log_density(bridgestan_model, q)
            catch
                typemin(eltype(q))
            end
        end
    end

    function constrain_draws(draws, warmup; include_tp = false)
        return mapslices(
            q -> BS.param_constrain(bsm, q; include_tp = include_tp),
            draws[warmup+1:end, :, :];
            dims=2
        )
    end

    @testset "arK-arK" begin
        model_name = model_names[1]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            mh = HM(dims)
            draws, diagnostics, rngs = sample!(mh, ld;
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end
    end

    @testset "arma-arma11" begin
        model_name = model_names[2]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               warmup=2000, iterations=5000)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            mh = MH(dims)
            draws, diagnostics, rngs = sample!(mh, ldg;
                                               warmup=2000, iterations=5000)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end
    end

    @testset "garch-garch11" begin
        model_name = model_names[3]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               warmup=2000, iterations=5000)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            mh = MH(dims)
            draws, diagnostics, rngs = sample!(mh, ldg;
                                               warmup=2000, iterations=5000)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end
    end

    @testset "gp_pois_regr-gp_pois_regr" begin
        model_name = model_names[4]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            ssda = StepsizeDualAverage(ones(4); δ=0.99)
            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               stepsize_adapter = ssda,
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            ssda = StepsizeDualAverage(ones(4); δ=0.6)
            mh = MH(dims)
            draws, diagnostics, rngs = sample!(mh, ldg;
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end
    end

    @testset "highd_mvnormal" begin
        model_name = model_names[5]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               stepsize_adapter = ssda,
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)

            s = reshape(std(constrained_draws; dims=(1, 3)), :)
            err_std = reshape(mcse_std(constrained_draws), :)

            true_s = expectations[model_name][:true_std]
            @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            mh = MH(dims)
            draws, diagnostics, rngs = sample!(mh, ldg;
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)

            s = reshape(std(constrained_draws; dims=(1, 3)), :)
            err_std = reshape(mcse_std(constrained_draws), :)

            true_s = expectations[model_name][:true_std]
            @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
        end
    end

    @testset "illconditioned_mvnormal" begin
        model_name = model_names[6]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               stepsize_adapter = ssda,
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)

            s = reshape(std(constrained_draws; dims=(1, 3)), :)
            err_std = reshape(mcse_std(constrained_draws), :)

            true_s = expectations[model_name][:true_std]
            @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            mh = MH(dims)
            draws, diagnostics, rngs = sample!(mh, ldg;
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)

            s = reshape(std(constrained_draws; dims=(1, 3)), :)
            err_std = reshape(mcse_std(constrained_draws), :)

            true_s = expectations[model_name][:true_std]
            @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
        end
    end

    @testset "mesquite-logmesquite" begin
        model_name = model_names[7]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            ssda = StepsizeDualAverage(ones(4); δ=0.99)
            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               stepsize_adapter = ssda,
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            ssda = StepsizeDualAverage(ones(4); δ=0.6)
            mh = MH(dims)
            draws, diagnostics, rngs = sample!(mh, ldg;
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)
        end
    end

    @testset "Student-t" begin
        model_name = model_names[8]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)

        true_m = expectations[model_name][:true_mean]

        @testset "Stan" begin
            ldg = prepare_log_density_gradient(bsm)
            iterations = 5_000
            warmup = 2_000

            stan = Stan(dims)
            draws, diagnostics, rngs = sample!(stan, ldg;
                                               stepsize_adapter = ssda,
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup; include_tp=true)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)

            s = reshape(std(constrained_draws; dims=(1, 3)), :)
            err_std = reshape(mcse_std(constrained_draws), :)

            true_s = expectations[model_name][:true_std]
            @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
        end

        @testset "MH" begin
            ld = prepare_log_density(bsm)
            iterations = 20_000
            warmup = 20_000

            mh = MH(dims)
            draws, diagnostics, rngs = sample!(mh, ldg;
                                               warmup=warmup, iterations=iterations)

            constrained_draws = constrain_draws(draws, warmup; include_tp=true)
            m = reshape(mean(constrained_draws; dims=(1, 3)), :)
            err_m = reshape(mcse_mean(constrained_draws), :)

            @test all(m .- 5 .* err_m .< true_m .< m .+ 5 .* err_m)

            s = reshape(std(constrained_draws; dims=(1, 3)), :)
            err_std = reshape(mcse_std(constrained_draws), :)

            true_s = expectations[model_name][:true_std]
            @test all(s .- 5 .* err_std .< true_s .< s .+ 5 .* err_std)
        end
    end
end
