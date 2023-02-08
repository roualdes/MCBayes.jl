abstract type AbstractMALA{T} <: AbstractSampler{T} end

struct MALA{T} <: AbstractMALA{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    dims::Int
    chains::Int
end

function MALA(
    dims, chains=4, T=Float64; metric=ones(T, dims, chains), stepsize=ones(T, chains)
)
    D = convert(Int, dims)::Int
    return MALA(metric, stepsize, D, chains)
end

function sample!(
    sampler::MALA,
    ldg;
    iterations=2000,
    warmup=iterations,
    draws_initializer=:stan,
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize; initializer=:stan, δ=0.6),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    adaptation_schedule=WindowedAdaptationSchedule(warmup),
    kwargs...,
)
    return run_sampler!(
        sampler,
        ldg;
        iterations,
        warmup,
        draws_initializer,
        stepsize_adapter,
        metric_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::MALA, m, ldg, draws, rngs, trace; kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    @sync for it in 1:nt
        Threads.@spawn for chain in it:nt:chains
            @views info = hmc!(
                draws[m, :, chain],
                draws[m + 1, :, chain],
                ldg,
                rngs[chain],
                sampler.dims,
                sampler.metric[:, chain],
                sampler.stepsize[chain],
                1,
                1000,
                kwargs...,
            )
            record!(trace, info, m + 1, chain)
        end
    end
end
