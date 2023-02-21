@testset "ChEES" begin
    iterations = 5_000
    warmup = 2_000

    @testset "arK-arK" begin
        model_name = model_names[1]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ldg = prepare_log_density_gradient(bsm)

        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(chees, ldg; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "arma-arma11" begin
        model_name = model_names[2]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ldg = prepare_log_density_gradient(bsm)

        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(chees, ldg; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "garch-garch11" begin
        model_name = model_names[3]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ldg = prepare_log_density_gradient(bsm)

        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(chees, ldg; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "gp_pois_regr-gp_pois_regr" begin
        model_name = model_names[4]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ldg = prepare_log_density_gradient(bsm)

        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(chees, ldg; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "highd_mvnormal" begin
        model_name = model_names[5]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ldg = prepare_log_density_gradient(bsm)

        ma = MetricFisherDivergence(ones(dims, 1))
        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(
            chees, ldg; metric_adapter=ma, iterations=iterations, warmup=warmup
        )

        constrained_draws = constrain_draws(bsm, draws, warmup; thin=10)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)

        true_stds = expectations[model_name][:true_std]
        @test check_stds(constrained_draws, true_stds)
    end

    @testset "illconditioned_mvnormal" begin
        model_name = model_names[6]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ldg = prepare_log_density_gradient(bsm)

        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(chees, ldg; warmup=warmup, iterations=iterations)

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
        ldg = prepare_log_density_gradient(bsm)

        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(chees, ldg; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)
    end

    @testset "Student-t" begin
        model_name = model_names[8]
        bsm = prepare_model(model_name)
        dims = BS.param_unc_num(bsm)
        ldg = prepare_log_density_gradient(bsm)

        chees = ChEES(dims)
        draws, diagnostics, rngs = sample!(chees, ldg; warmup=warmup, iterations=iterations)

        constrained_draws = constrain_draws(bsm, draws, warmup; include_tp=true)
        true_means = expectations[model_name][:true_mean]
        @test check_means(constrained_draws, true_means)

        true_stds = expectations[model_name][:true_std]
        @test check_stds(constrained_draws, true_stds)
    end
end
