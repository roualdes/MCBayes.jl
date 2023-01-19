@testset "Metropolis" begin
    iterations = 30_000
    warmup = 20_000

    @testset "arK-arK" begin
        model_name = model_names[1]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ld = prepare_log_density(bsm)

        mh = MH(dims)
        draws, diagnostics, rngs = sample!(mh, ld; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "arma-arma11" begin
        model_name = model_names[2]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ld = prepare_log_density(bsm)

        mh = MH(dims)
        draws, diagnostics, rngs = sample!(mh, ld; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "garch-garch11" begin
        model_name = model_names[3]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ld = prepare_log_density(bsm)

        mh = MH(dims)
        draws, diagnostics, rngs = sample!(mh, ld; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "gp_pois_regr-gp_pois_regr" begin
        model_name = model_names[4]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ld = prepare_log_density(bsm)

        ssda = StepsizeDualAverage(ones(4); δ=0.6, initializer=:mh)
        mh = MH(dims)
        draws, diagnostics, rngs = sample!(
            mh, ld; stepsize_adapter=ssda, warmup=warmup, iterations=iterations
        )

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    # TODO make this test more stable, without ruining test times
    # @testset "highd_mvnormal" begin
    #     model_name = model_names[5]
    #     bsm = prepare_model(model_name)
    #     dims = BS.param_unc_num(bsm)
    #     ld = prepare_log_density(bsm)

    #     ssc = StepsizeConstant(ones(4))
    #     mh = MH(dims)
    #     draws, diagnostics, rngs = sample!(mh, ld;
    #                                        stepsize_adapter = ssc,
    #                                        warmup=warmup,
    #                                        iterations=iterations)

    #     constrained_draws = constrain_draws(bsm, draws, warmup, thin = 10)
    #     true_means = expectations[model_name][:true_mean]
    #     @test check_means(constrained_draws, true_means)

    #     true_stds = expectations[model_name][:true_std]
    #     @test check_stds(constrained_draws, true_stds)
    # end

    @testset "illconditioned_mvnormal" begin
        model_name = model_names[6]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ld = prepare_log_density(bsm)

        mh = MH(dims)
        draws, diagnostics, rngs = sample!(mh, ld; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)

        true_stds = expectations[model_name][:true_std]
        @test check_stds(constrained_draws, true_stds)
    end

    @testset "mesquite-logmesquite" begin
        model_name = model_names[7]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ld = prepare_log_density(bsm)

        ssda = StepsizeDualAverage(ones(4); δ=0.6, initializer=:mh)
        mh = MH(dims)
        draws, diagnostics, rngs = sample!(
            mh, ld; stepsize_adapter=ssda, warmup=warmup, iterations=iterations
        )

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "Student-t" begin
        model_name = model_names[8]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ld = prepare_log_density(bsm)

        mh = MH(dims)
        draws, diagnostics, rngs = sample!(mh, ld; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup; include_tp=true)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)

        true_stds = expectations[model_name][:true_std]
        @test check_stds(constrained_draws, true_stds)
    end
end
