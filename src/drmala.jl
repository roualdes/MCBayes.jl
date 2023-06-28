# adapted from https://github.com/modichirag/hmc/blob/main/src/algorithms.py#L463
abstract type AbstractDrMALA{T} <: AbstractSampler{T} end

struct DrMALA{T} <: AbstractDrMALA{T}
    momentum::Matrix{T}
    metric::Matrix{T}
    pca::Vector{T}
    trajectorylength::Vector{T}
    stepsize::Vector{T}
    damping::Vector{T}
    noise::Vector{T}
    dims::Int
    chains::Int
end

# TODO(ear) remove ability to set placeholders such as metric,
# stepsize, trajectorylength, noise, damping, etc. from all sampler
# constructors, as it conflates where I want the defaults to be set.
# I want the API to set the defaults in the adapters.
function DrMALA(
    dims,
    chains=12,
    T=Float64;
    metric=ones(T, dims, 1),
    pca=zeros(T, dims),
    stepsize=ones(T, 1),
    trajectorylength=ones(T, 1),
    )
    momentum = randn(T, dims, chains)
    D = convert(Int, dims)::Int
    damping = ones(T, 1)
    noise = ones(T, 1)
    return DrMALA(momentum, metric, pca, trajectorylength, stepsize, damping, noise, D, chains)
end

function sample!(
    sampler::DrMALA,
    ldg;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerAdam(),
    stepsize_initializer=StepsizeInitializerSGA(),
    stepsize_adapter=StepsizeAdam(sampler.stepsize, warmup; δ=0.6),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    pca_adapter=PCAOnline(eltype(sampler), sampler.dims),
    trajectorylength_adapter=TrajectorylengthLDG(
        sampler.trajectorylength, sampler.dims, warmup
    ),
    damping_adapter=DampingMALT(sampler.damping),
    noise_adapter=NoiseMALT(sampler.noise),
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
        metric_adapter,
        pca_adapter,
        trajectorylength_adapter,
        damping_adapter,
        noise_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::DrMALA, m, ldg, draws, rngs, trace; kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    stepsize = sampler.stepsize[1]
    trajectorylength = sampler.trajectorylength[1]
    steps = trajectorylength / stepsize
    steps = ifelse(isfinite(steps), steps, 1)
    steps = round(Int64, clamp(steps, 1, 1000))
    metric = sampler.metric[:, 1]
    # metric ./= maximum(metric)
    noise = sampler.noise[1]
    # TODO deal with J and reduction_factor
    J = get(kwargs, :J, 3)
    reduction_factor = get(kwargs, :reduction_factor, 4)
    Threads.@threads for it in 1:nt
        for chain in it:nt:chains
            @views info = drhmc!(
                draws[m, :, chain],
                draws[m + 1, :, chain],
                sampler.momentum[:, chain],
                ldg,
                rngs[chain],
                sampler.dims,
                metric,
                stepsize,
                31, # steps,
                noise,
                J,
                reduction_factor,
                1000;
                kwargs...,
            )
            info = (; info..., trajectorylength, damping=sampler.damping[1])
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end
