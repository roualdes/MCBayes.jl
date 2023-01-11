abstract type AbstractMEADS{T} <: AbstractSampler{T} end

struct MEADS{T} <: AbstractMEADS{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    momentum::Matrix{T}
    acceptprob::Vector{T}
    damping::Vector{T}          # γ
    noise::Vector{T}            # α
    drift::Vector{T}            # δ
    permutation::Matrix{Int}
    dims::Int
    folds::Int
    chainsperfold::Int
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
)
    D = convert(Int, dims)::Int
    chains = folds * chainsperfold
    momentum = randn(T, dims, chains), acceptprob = 2 .* rand(T, chains) .- 1
    permutation = shuffle_folds(chains, chainsperfold)
    damping = 1 ./ stepsize
    noise = 1 .- exp.(-2 .* damping .* stepsize)
    drift = 0.5 .* noise,
    return Stan(
        metric,
        stepsize,
        momentum,
        acceptprob,
        permutation,
        damping,
        noise,
        drift,
        D,
        folds,
        chainsperfold,
    )
end

struct EnsembleChainSchedule end

function sample!(
    sampler::MEADS{T},
    ldg;
    draws_initializer=:adam,
    stepsize_adapter=StepsizeECA(sampler.stepsize),
    metric_adapter=MetricECA(sampler.metric),
    damping_adapter=DampingECA(1 ./ sampler.stepsize),
    noise_adapter=NoiseECA(1 .- exp.(-2)),
    drift_adapter=DriftECA(noise_adapter.noise),
    adaptation_schedule=EnsembleChainSchedule();
    kwargs...,
) where {T} end

function transition!(sampler::MEADS, m, ldg, draws, rngs, trace; kwargs...)
    chains = size(draws, 3)
    if m % sampler.folds == 0
        sampler.permutation = shuffle_folds(chains, sampler.chainsperfold)
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
                    @views info = meads_kernel!(
                        draws[m, :, chain],
                        draws[m + 1, :, chain],
                        rngs[chain],
                        sampler.dims,
                        sampler.metric[:, fold],
                        sampler.stepsize[fold],
                        sampler.momentum[:, chain],
                        sampler.acceptprob[chain:chain],
                        ldg;
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
