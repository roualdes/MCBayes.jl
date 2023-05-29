abstract type AbstractSGA{T} <: AbstractSampler{T} end

struct ChEES{T} <: AbstractSGA{T}
    metric::Matrix{T}
    pca::Vector{T}
    stepsize::Vector{T}
    trajectorylength::Vector{T}
    dims::Int
    chains::Int
end

function ChEES(
    dims,
    chains=10,
    T=Float64;
    metric=ones(T, dims, 1),
    stepsize=ones(T, 1),
    trajectorylength=ones(T, 1),
)
    D = convert(Int, dims)::Int
    return ChEES(metric, zeros(T, dims), stepsize, trajectorylength, D, chains)
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
    pca::Vector{T}
    stepsize::Vector{T}
    trajectorylength::Vector{T}
    dims::Int
    chains::Int
end

function SNAPER(
    dims,
    chains=12,
    T=Float64;
    metric=ones(T, dims, 1),
    pca=zeros(T, dims),
    stepsize=ones(T, 1),
    trajectorylength=ones(T, 1),
)
    D = convert(Int, dims)::Int
    return SNAPER(metric, pca, stepsize, trajectorylength, D, chains)
end

function sample!(
    sampler::SNAPER,
    ldg;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerSGA(),
    stepsize_adapter=StepsizeAdam(sampler.stepsize; δ=0.8),
    trajectorylength_adapter=TrajectorylengthSNAPER(sampler.trajectorylength, sampler.dims),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    pca_adapter=PCAOnline(eltype(sampler), sampler.dims),
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
        pca_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::AbstractSGA, m, ldg, draws, rngs, trace; kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    trajectorylength = sampler.trajectorylength[1]
    stepsize = sampler.stepsize[1]
    steps = trajectorylength / stepsize
    steps = round(Int64, clamp(ifelse(isfinite(steps), steps, 1), 1, 1000))
    metric = sampler.metric[:, 1]
    metric ./= maximum(metric)
    Threads.@threads for it in 1:nt
        for chain in it:nt:chains
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
