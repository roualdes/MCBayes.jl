abstract type AbstractMH{T} <: AbstractSampler{T} end

struct MH{T} <: AbstractMH{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    dims::Int
    chains::Int
end

function MH(
    dims, chains=4, T=Float64; metric=ones(T, dims, chains), stepsize=ones(T, chains)
)
    D = convert(Int, dims)::Int
    return MH(metric, stepsize, D, chains)
end

function sample!(
    sampler::MH,
    ld;
    iterations=5000,
    warmup=iterations,
    draws_initializer=:mh,
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize; initializer=:mh, δ=0.3),
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
        stepsize_adapter,
        metric_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::MH, m, ld, draws, rngs, trace; kwargs...)
    for chain in axes(draws, 3) # TODO multi-thread-able
        @views info = mh_kernel!(
            draws[m, :, chain],
            draws[m + 1, :, chain],
            rngs[chain],
            sampler.dims,
            sampler.metric[:, chain],
            sampler.stepsize[chain],
            ld;
            kwargs...,
        )
        record!(trace, info, m + 1, chain)
    end
end

function mh_kernel!(position, position_next, rng, dims, metric, stepsize, ld; kwargs...)
    T = eltype(position)
    position_next .= position .+ randn(rng, T, dims) .* stepsize .* sqrt.(metric)
    a = exp(ld(copy(position_next); kwargs...) - ld(copy(position); kwargs...))
    acceptstat = min(1, a)
    accepted = rand(rng, T) < acceptstat
    position_next .= position_next .* accepted .+ position .* (1 - accepted)
    return (; accepted, acceptstat, stepsize)
end
