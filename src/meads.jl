abstract type AbstractMEADS{T} <: AbstractSampler{T} end

struct MEADS{T} <: AbstractMEADS{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    momentum::Matrix{T}
    acceptprob::Vector{T}
    damping::Vector{T}          # γ
    noise::Vector{T}            # α
    drift::Vector{T}            # δ
    nru::Bool
    partition::Matrix{Int}
    dims::Int
    folds::Int
    chainsperfold::Int
    chains::Int
end

function shuffle_folds(num_chains, num_chainsperfold)
    permchains = Random.randperm(num_chains)
    partchains = Iterators.partition(permchains, num_chainsperfold)
    return reduce(hcat, collect.(partchains))
end

"""
    MEADS(dims, folds, T = Float64; kwargs...)

Initialize MEADS sampler object.  The number of dimensions `dims` and number of
chains `chains` are the only required arguments.  The type `T` of the ...
"""
function MEADS(
    dims,
    folds=4,
    chainsperfold=32,
    T=Float64;
    metric=ones(T, dims, folds),
    stepsize=ones(T, folds),
    nru=false
)
    D = convert(Int, dims)::Int
    chains = folds * chainsperfold
    momentum = randn(T, dims, chains)
    acceptprob = 2 .* rand(T, chains) .- 1
    partition = shuffle_folds(chains, chainsperfold)
    damping = 1 ./ stepsize
    noise = 1 .- exp.(-2 .* damping .* stepsize)
    drift = 0.5 .* noise
    return MEADS(
        metric,
        stepsize,
        momentum,
        acceptprob,
        damping,
        noise,
        drift,
        nru,
        partition,
        D,
        folds,
        chainsperfold,
        chains
    )
end

function sample!(
    sampler::MEADS,
    ldg;
    draws_initializer=:adam,
    stepsize_adapter=StepsizeECA(sampler.stepsize),
    metric_adapter=MetricECA(sampler.metric),
    damping_adapter=DampingECA(1 ./ sampler.stepsize),
    noise_adapter=NoiseECA(1 .- exp.(-2 .* damping_adapter.damping .* stepsize_adapter.stepsize)),
    drift_adapter=DriftECA(noise_adapter.noise),
    adaptation_schedule=EnsembleChainSchedule(),
    kwargs...,
)
    return run_sampler!(sampler,
                        ldg;
                        draws_initializer,
                        stepsize_adapter,
                        metric_adapter,
                        damping_adapter,
                        noise_adapter,
                        drift_adapter,
                        adaptation_schedule,
                        kwargs...
    )
end

function transition!(sampler::MEADS, m, ldg, draws, rngs, trace; kwargs...)
    chains = size(draws, 3)
    if m % sampler.folds == 0
        sampler.partition .= shuffle_folds(chains, sampler.chainsperfold)
    end

    skipfold = m % sampler.folds + 1
    nt = get(kwargs, :threads, Threads.nthreads())

    @sync for it in 1:nt
        Threads.@spawn for f in it:nt:(sampler.folds)
            k = (f + 1) % sampler.folds + 1
            kfold = sampler.partition[:, k]

            fold = sampler.partition[:, f]
            for chain in fold
                if f != skipfold
                    @views info = pghmc!(
                        draws[m, :, chain],
                        draws[m + 1, :, chain],
                        sampler.momentum[:, chain],
                        ldg,
                        rngs[chain],
                        sampler.dims,
                        sampler.metric[:, f],
                        sampler.stepsize[f],
                        sampler.acceptprob[chain:chain],
                        sampler.noise[f],
                        sampler.drift[f],
                        sampler.nru,
                        kwargs...,
                    )
                    record!(trace, info, m + 1, chain)
                else
                    draws[m + 1, :, chain] .= draws[m, :, chain]
                end
            end
        end
    end
end
