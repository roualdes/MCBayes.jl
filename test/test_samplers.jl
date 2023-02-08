using BridgeStan

const BS = BridgeStan
cwd = if get(ENV, "CI", "false") == "true"
    dirname(pwd())
else
    homedir()
end
bsdir = joinpath(cwd, "bridgestan")
BS.set_bridgestan_path!(bsdir)

function prepare_model(model_name)
    modeldir = joinpath(artifact"test_models", "test_models", model_name)
    stan_file = joinpath(modeldir, model_name * ".stan")
    stan_data = joinpath(modeldir, model_name * ".json")
    bsm = BS.StanModel(;
        stan_file=stan_file, data=stan_data, make_args=["STAN_THREADS=true"]
    )
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

function constrain_draws(bridgestan_model, draws, warmup; include_tp=false, thin=1)
    return mapslices(
        q -> BS.param_constrain(bridgestan_model, q; include_tp=include_tp),
        draws[(warmup + 1):thin:end, :, :];
        dims=2,
    )
end

function check_means(constrained_draws, true_means; z=5)
    m = reshape(mean(constrained_draws; dims=(1, 3)), :)
    err_m = reshape(mcse_mean(constrained_draws), :)
    l = length(true_means)
    if length(m) == l
        return all(m .- z .* err_m .< true_means .< m .+ z .* err_m)
    else
        l -= 1
        return all(
            m[(end - l):end] .- z .* err_m[(end - l):end] .<
            true_means .<
            m[(end - l):end] .+ z .* err_m[(end - l):end],
        )
    end
end

function check_stds(constrained_draws, true_stds; z=5)
    s = reshape(std(constrained_draws; dims=(1, 3)), :)
    err_std = reshape(mcse_std(constrained_draws), :)
    l = length(true_stds)
    if length(s) == l
        return all(s .- z .* err_std .< true_stds .< s .+ z .* err_std)
    else
        l -= 1
        return all(
            s[(end - l):end] .- z .* err_std[(end - l):end] .<
            true_stds .<
            s[(end - l):end] .+ z .* err_std[(end - l):end],
        )
    end
end

modeldir = joinpath(artifact"test_models", "test_models")
# most models and values from stan-dev/posteriordb
expectations = open(deserialize, joinpath(modeldir, "expectations.jls"))
model_names = [f for f in readdir(modeldir) if isdir(joinpath(modeldir, f))]

# include("mh.jl")
# include("stan.jl")
# include("meads.jl")
include("mala.jl")
