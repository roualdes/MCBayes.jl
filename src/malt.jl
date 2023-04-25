abstract type AbstractMALT{T} <: AbstractSampler{T} end

struct MALT{T} <: AbstractMALT{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    trajectorylength::Vector{T}
    damping::Vector{T}
    noise::Vector{T}
    dims::Int
    chains::Int
end

# TODO(ear) remove ability to set placeholders such as metric,
# stepsize, trajectorylength, noise, damping, etc. from all sampler
# constructors, as it conflates where I want the defaults to be set.
# I want the API to set the defaults in the adapters.
function MALT(
    dims, chains=12, T=Float64; metric=ones(T, dims, 1), stepsize=ones(T, 1), trajectorylength=ones(T, 1)
    )
    D = convert(Int, dims)::Int
    damping = ones(T, 1)
    noise = exp.(-0.5 .* damping .* stepsize)
    return MALT(metric, stepsize, trajectorylength, damping, noise, D, chains)
end

function sample!(
    sampler::MALT,
    ldg;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerSGA(),
    stepsize_adapter=StepsizeAdam(sampler.stepsize; δ=0.8),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    trajectorylength_adapter = TrajectorylengthChEES(sampler.trajectorylength, sampler.dims),
    damping_adapter = DampingMALT(sampler.damping),
    noise_adapter = NoiseMALT(sampler.noise),
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
        trajectorylength_adapter,
        damping_adapter,
        noise_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::MALT, m, ldg, draws, rngs, trace; kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    Threads.@threads for it in 1:nt
        for chain in it:nt:chains
            stepsize = sampler.stepsize[1]
            trajectorylength = sampler.trajectorylength[1]
            steps = ceil(Int64, clamp(trajectorylength / stepsize, 1, 1000))
            metric = sampler.metric[:, 1]
            noise = sampler.noise[1]
            @views info = malt!(
                draws[m, :, chain],
                draws[m + 1, :, chain],
                ldg,
                rngs[chain],
                sampler.dims,
                metric,
                stepsize,
                steps,
                noise,
                1000;
                kwargs...,
            )
            info = (; info..., trajectorylength, damping = sampler.damping[1])
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end
