abstract type AbstractDRGHMC{T} <: AbstractSampler{T} end

struct DRGHMC{T} <: AbstractDRGHMC{T}
    momentum::Matrix{T}
    metric::Matrix{T}
    pca::Matrix{T}
    stepsize::Vector{T}
    reductionfactor::Vector{T}
    damping::Vector{T}
    noise::Vector{T}
    drift::Vector{T}
    acceptanceprob::Vector{T}
    dims::Int
    chains::Int
end

# TODO(ear) remove ability to set placeholders such as metric,
# stepsize, trajectorylength, noise, damping, etc. from all sampler
# constructors, as it conflates where I want the defaults to be set.
# I want the API to set the defaults in the adapters.
function DRGHMC(
    dims,
    chains=12,
    T=Float64;
    metric=ones(T, dims, 1),
    pca=randn(T, dims, 1),
    stepsize=ones(T, 1),
    reductionfactor=2 * ones(T, 1),
)
    momentum = randn(T, dims, chains)
    D = convert(Int, dims)::Int
    pca ./= mapslices(norm, pca, dims = 1)
    damping = ones(T, 1)
    noise = exp.(-2 .* damping .* stepsize)
    drift = (1 .- noise .^ 2) ./ 2
    acceptanceprob = 2 .* rand(T, chains) .- 1
    return DRGHMC(
        momentum, metric, pca, stepsize, reductionfactor, damping, noise, drift, acceptanceprob, D, chains
    )
end

function sample!(
    sampler::DRGHMC,
    ldg!;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerStan(),
    steps_adapter=StepsConstant(copy(sampler.steps)),
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize; δ=0.5),
    reductionfactor_adapter=ReductionFactorConstant(sampler.reductionfactor; reductionfactor_δ = 0.9),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    pca_adapter=PCAOnline(sampler.pca),
    damping_adapter=DampingMALT(sampler.damping),
    noise_adapter=NoiseMALT(sampler.noise),
    adaptation_schedule=WindowedAdaptationSchedule(warmup),
    kwargs...,
)
    return run_sampler!(
        sampler,
        ldg!;
        iterations,
        warmup,
        draws_initializer,
        stepsize_initializer,
        stepsize_adapter,
        reductionfactor_adapter,
        steps_adapter,
        metric_adapter,
        pca_adapter,
        damping_adapter,
        noise_adapter,
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::DRGHMC, m, ldg!, draws, rngs, trace;
                     J = 3, nonreversible_update = false,
                     kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)

    reduction_factor = sampler.reductionfactor[1]
    damping = sampler.damping[1]
    metric = sampler.metric[:, 1]
    metric ./= maximum(metric)
    noise = sampler.noise[1]
    drift = sampler.drift[1]
    stepsize = sampler.stepsize[1]

    Threads.@threads for it in 1:nt
        for chain in it:nt:chains
            acceptanceprob = sampler.acceptanceprob[chain:chain]

            @views info = drhmc!(
                draws[m, :, chain],
                draws[m + 1, :, chain],
                sampler.momentum[:, chain],
                ldg!,
                rngs[chain],
                sampler.dims,
                metric,
                stepsize,
                1,
                noise,
                drift,
                acceptanceprob,
                J,
                reduction_factor,
                nonreversible_update,
                1000;
                kwargs...,
            )

            info = (; info...,
                    damping,
                    reductionfactor = reduction_factor)
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end

function adapt!(
    sampler::DRGHMC,
    schedule::WindowedAdaptationSchedule,
    trace,
    m,
    ldg!,
    draws,
    rngs,
    metric_adapter,
    pca_adapter,
    stepsize_initializer,
    stepsize_adapter,
    reductionfactor_adapter,
    steps_adapter,
    trajectorylength_adapter,
    damping_adapter,
    noise_adapter,
    drift_adapter;
    steps_coefficient = 1,
    kwargs...,
    )

    warmup = schedule.warmup
    if m <= warmup
        accept_stats = trace.acceptstat[m + 1, :]
        update!(stepsize_adapter, mean(accept_stats); kwargs...)
        set!(sampler, stepsize_adapter; kwargs...)

        update!(noise_adapter, sampler.damping, sampler.stepsize; kwargs...)
        set!(sampler, noise_adapter; kwargs...)
        sampler.drift .= (1 .- sampler.noise .^ 2) ./ 2

        if schedule.firstwindow <= m <= schedule.lastwindow
            final_accept_stats = trace.finalacceptstat[m + 1, :]
            update!(reductionfactor_adapter, maybe_mean(final_accept_stats); kwargs...)
            set!(sampler, reductionfactor_adapter; kwargs...)

            positions = draws[m + 1, :, :]

            update!(metric_adapter, positions, ldg!; kwargs...)

            metric = sqrt.(sampler.metric)
            metric ./= maximum(metric, dims = 1)

            update!(
                pca_adapter, positions, metric_mean(metric_adapter), metric; kwargs...
                    )

            update!(damping_adapter, m + 1, lambda, sampler.stepsize; kwargs...)
        end

        if m == schedule.closewindow
            initialize_stepsize!(
                stepsize_initializer,
                stepsize_adapter,
                sampler,
                rngs,
                ldg!,
                draws[m + 1, :, :];
                kwargs...,
            )
            set!(sampler, stepsize_adapter; kwargs...)
            reset!(stepsize_adapter; kwargs...)

            set!(sampler, reductionfactor_adapter; kwargs...)
            reset!(reductionfactor_adapter; kwargs...)

            set!(sampler, metric_adapter; kwargs...)
            reset!(metric_adapter)

            set!(sampler, pca_adapter; kwargs...)
            reset!(pca_adapter; reset_pc = true)

            set!(sampler, damping_adapter; kwargs...)

            calculate_nextwindow!(schedule)
        end
    else
        set!(sampler, stepsize_adapter; smoothed=true, kwargs...)
        set!(sampler, metric_adapter)
        set!(sampler, pca_adapter)
        set!(sampler, damping_adapter)
        set!(sampler, noise_adapter)
        set!(sampler, reductionfactor_adapter; smoothed=true, kwargs...)
    end
end
