abstract type AbstractRWM{T} <: AbstractSampler{T} end

struct RWM{T} <: AbstractRWM{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    dims::Int
    chains::Int
end

function RWM(
    dims, chains=4, T=Float64; metric=ones(T, dims, chains), stepsize=ones(T, chains)
)
    D = convert(Int, dims)::Int
    return RWM(metric, stepsize, D, chains)
end

function sample!(
    sampler::RWM,
    ld;
    iterations=5000,
    warmup=iterations,
    draws_initializer=DrawsInitializerRWM(),
    stepsize_initializer=StepsizeInitializerRWM(),
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize; δ=0.3),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    adaptation_schedule=WindowedAdaptationSchedule(warmup),
    kwargs...,
)
    return run_sampler!(
        sampler,
        ld;
        iterations,
        warmup,
        draws_initializer,
        stepsize_initializer,
        stepsize_adapter,
        metric_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::RWM, m, ld, draws, rngs, trace; kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    @sync for it in 1:nt
        Threads.@spawn for chain in it:nt:chains
            @views info = rwm_kernel!(
                draws[m, :, chain],
                draws[m + 1, :, chain],
                rngs[chain],
                sampler.dims,
                sampler.metric[:, chain],
                sampler.stepsize[chain],
                ld;
                kwargs...,
            )
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end

function rwm_kernel!(position, position_next, rng, dims, metric, stepsize, ld; kwargs...)
    T = eltype(position)
    position_next .= position .+ randn(rng, T, dims) .* stepsize .* sqrt.(metric)
    ld_next = ld(copy(position_next); kwargs...)
    a = exp(ld_next - ld(copy(position); kwargs...))
    acceptstat = min(1, a)
    accepted = rand(rng, T) < acceptstat
    position_next .= position_next .* accepted .+ position .* (1 - accepted)
    return (; accepted, acceptstat, stepsize, ld = ld_next)
end
