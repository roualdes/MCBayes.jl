abstract type AbstractSGA{T} <: AbstractSampler{T} end

struct ChEES{T} <: AbstractSGA{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    trajectorylength::Vector{T}
    dims::Int
    chains::Int
end

function ChEES(
    dims, chains=10, T=Float64; metric=ones(T, dims, 1), stepsize=ones(T, 1), trajectorylength=ones(T, 1)
)
    D = convert(Int, dims)::Int
    return ChEES(metric, stepsize, trajectorylength, D, chains)
end

function sample!(
    sampler::ChEES,
    ldg;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerSGA(),
    stepsize_adapter=StepsizeAdam(sampler.stepsize; δ=0.8),
    trajectorylength_adapter=TrajectorylengthChEES(sampler.trajectorylength, sampler.dims),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    adaptation_schedule=SGAAdaptationSchedule(warmup),
    kwargs...,
)
    return run_sampler!(
        sampler,
        ldg;
        iterations,
        warmup,
        draws_initializer,
        stepsize_initializer,
        stepsize_adapter,
        trajectorylength_adapter,
        metric_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

struct SNAPER{T} <: AbstractSGA{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    trajectorylength::Vector{T}
    dims::Int
    chains::Int
end

function SNAPER(
    dims, chains=10, T=Float64; metric=ones(T, dims, 1), stepsize=ones(T, 1), trajectorylength=ones(T, 1)
)
    D = convert(Int, dims)::Int
    return SNAPER(metric, stepsize, trajectorylength, D, chains)
end

function sample!(
    sampler::SNAPER,
    ldg;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerSGA(),
    stepsize_adapter=StepsizeAdam(sampler.stepsize; δ=0.8),
    # TODO trajectorylength_adapter=Trajectorylength(),
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
        stepsize_initializer,
        stepsize_adapter,
        metric_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::AbstractSGA, m, ldg, draws, rngs, trace; kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    stepsize = sampler.stepsize[1]
    trajectorylength = sampler.trajectorylength[1]
    steps = max(1, ceil(Int, 2 * halton(m) * trajectorylength / stepsize))
    metric = sampler.metric[:, 1]
    metric ./= maximum(metric)
    @sync for it in 1:nt
        Threads.@spawn for chain in it:nt:chains
            @views info = hmc!(
                draws[m, :, chain],
                draws[m + 1, :, chain],
                ldg,
                rngs[chain],
                sampler.dims,
                metric,
                stepsize,
                steps,
                1000;
                kwargs...,
            )
            info = (; info..., trajectorylength)
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end
