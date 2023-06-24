abstract type AbstractDrMALA{T} <: AbstractSampler{T} end

struct DrMALA{T} <: AbstractDrMALA{T}
    momentum::Matrix{T}
    metric::Matrix{T}
    pca::Vector{T}
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
    )
    momentum = randn(T, dims, chains)
    D = convert(Int, dims)::Int
    damping = ones(T, 1)
    noise = ones(T, 1)
    return DrMALA(momentum, metric, pca, stepsize, damping, noise, D, chains)
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
    metric = sampler.metric[:, 1]
    metric ./= maximum(metric)
    noise = sampler.noise[1]
    warmup = get(kwargs, :warmup, div(size(draws, 1), 2))
    T = eltype(position)
    Threads.@threads for it in 1:nt
        for chain in it:nt:chains

            q = draws[m, :, chain]
            p = noise .* sampler.momentum[:, chain] .+ sqrt.(1 .- noise .^ 2) .* randn(T, dims)

            ld, gradient = ldg(q; kwargs...)
            H0 = hamiltonian(ld, p)
            isnan(H0) && (H0 = typemax(T))

            ld, gradient = leapfrog!(q, p, ldg, gradient, stepsize .* sqrt.(metric), 1; kwargs...)

            H = hamiltonian(ld, p)
            isnan(H) && (H = typemax(T))
            divergent = H - H0 > maxdeltaH

            a = min(1, exp(H0 - H))
            accepted = rand(rng, T) <= a

            if accepted
                position_next .= q
                momentum .= p
            else
                position_next .= position
                momentum .*= -1
            end

            # @views info = drhmc!(
            #     draws[m, :, chain],
            #     draws[m + 1, :, chain],
            #     sampler.momentum[:, chain],
            #     ldg,
            #     rngs[chain],
            #     sampler.dims,
            #     metric,
            #     stepsize,
            #     1,
            #     noise,
            #     1000;
            #     kwargs...,
            # )
            info = (; info..., damping=sampler.damping[1])
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end
