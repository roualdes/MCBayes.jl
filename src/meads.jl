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
    maxdeltaH::T
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
    stepsize=ones(T, folds) / 2,
    nru=true,
    maxdeltaH=convert(T, 1000),
)
    D = convert(Int, dims)::Int
    chains = folds * chainsperfold
    momentum = randn(T, dims, chains)
    acceptprob = 2 .* rand(T, chains) .- 1
    partition = shuffle_folds(chains, chainsperfold)
    damping = 1 ./ stepsize
    noise = 1 .- exp.(-2 .* damping .* stepsize)
    drift = noise ./ 2
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
        chains,
        maxdeltaH,
    )
end

function sample!(
    sampler::MEADS,
    ldg;
    draws_initializer=DrawsInitializerAdam(),
    stepsize_initializer=StepsizeInitializerMEADS(),
    stepsize_adapter=StepsizeECA(sampler.stepsize),
    metric_adapter=MetricECA(sampler.metric),
    damping_adapter=DampingECA(sampler.damping),
    noise_adapter=NoiseECA(sampler.noise),
    drift_adapter=DriftECA(sampler.drift),
    adaptation_schedule=EnsembleChainSchedule(),
    kwargs...,
)
    return run_sampler!(
        sampler,
        ldg;
        draws_initializer,
        stepsize_initializer,
        stepsize_adapter,
        metric_adapter,
        damping_adapter,
        noise_adapter,
        drift_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::MEADS, m, ldg, draws, rngs, trace; kwargs...)
    if m % sampler.folds == 0
        sampler.partition .= shuffle_folds(sampler.chains, sampler.chainsperfold)
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
                        sampler.maxdeltaH;
                        kwargs...,
                    )
                    # TODO at least consider doing "adaptation" in here for speed,
                    # since technically this algorithm doesn't "adapt"

                    # TODO not properly recording diagnostics
                    #record!(trace, info, m + 1, chain)
                else
                    draws[m + 1, :, chain] = draws[m, :, chain]
                end
            end
        end
    end
end
