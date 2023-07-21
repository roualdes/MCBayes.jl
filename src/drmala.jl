# adapted from https://github.com/modichirag/hmc/blob/main/src/algorithms.py#L463
abstract type AbstractDrMALA{T} <: AbstractSampler{T} end

struct DrMALA{T} <: AbstractDrMALA{T}
    momentum::Matrix{T}
    metric::Matrix{T}
    pca::Matrix{T}
    steps::Vector{Int}
    stepsize::Vector{T}
    reductionfactor::Vector{T}
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
    pca=randn(T, dims, 1),
    stepsize=ones(T, chains),
    reductionfactor=5 * ones(T, 1),
    steps=10 * ones(Int, chains),
)
    momentum = randn(T, dims, chains)
    D = convert(Int, dims)::Int
    pca ./= mapslices(norm, pca, dims = 1)
    damping = ones(T, 1)
    noise = exp.(-2 .* damping .* stepsize)
    return DrMALA(
        momentum, metric, pca, steps, stepsize, reductionfactor, damping, noise, D, chains
    )
end

function sample!(
    sampler::DrMALA,
    ldg!;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerStan(),
    steps_adapter=StepsPCA(sampler.steps),
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize; δ=0.5),
    reductionfactor_adapter=ReductionFactorDualAverage(sampler.reductionfactor; reductionfactor_δ = 0.9),
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

function transition!(sampler::DrMALA, m, ldg!, draws, rngs, trace;
                     J = 3, kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    pca = sampler.pca ./ mapslices(norm, sampler.pca, dims = 1)
    reduction_factor = sampler.reductionfactor[1]
    Threads.@threads for it in 1:nt
        for chain in it:nt:chains
            stepsize = sampler.stepsize[chain]
            steps = sampler.steps[chain]
            metric = sampler.metric[:, 1]
            metric ./= maximum(metric)
            noise = sampler.noise[chain]
            damping = sampler.damping[1]

            local info
            acceptstats = zeros(steps)
            finalacceptstats = zeros(steps)
            retried = zeros(Int, steps)
            lastposition = draws[m, :, chain]
            for step in 1:steps
                @views info = drhmc!(
                    lastposition,
                    draws[m + 1, :, chain],
                    sampler.momentum[:, chain],
                    ldg!,
                    rngs[chain],
                    sampler.dims,
                    metric,
                    stepsize,
                    1,
                    noise,
                    J,
                    reduction_factor,
                    1000;
                    kwargs...,
                )
                lastposition .= draws[m + 1, :, chain]
                acceptstats[step] = info[:acceptstat]
                finalacceptstats[step] = info[:finalacceptstat]
                retried[step] = info[:retries]
            end
            info = (; info...,
                    damping,
                    reductionfactor = reduction_factor,
                    pca = pca[:, 1],
                    # previousposition = draws[m, :, chain],
                    acceptstat = mean(acceptstats),
                    finalacceptstat = maybe_mean(finalacceptstats),
                    steps = sum(retried))
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end

function adapt!(
    sampler::DrMALA,
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
    kwargs...,
    )

    warmup = schedule.warmup
    if m <= warmup
        accept_stats = trace.acceptstat[m + 1, :]
        update!(stepsize_adapter, accept_stats; kwargs...)
        set!(sampler, stepsize_adapter; kwargs...)

        update!(noise_adapter, sampler.damping, sampler.stepsize; kwargs...)
        set!(sampler, noise_adapter; kwargs...)

        if schedule.firstwindow <= m <= schedule.lastwindow
            final_accept_stats = trace.finalacceptstat[m + 1, :]
            update!(reductionfactor_adapter, maybe_mean(final_accept_stats); kwargs...)

            positions = draws[m + 1, :, :]

            update!(metric_adapter, positions, ldg!; kwargs...)

            metric = sqrt.(sampler.metric)
            metric ./= maximum(metric, dims = 1)

            update!(
                pca_adapter, positions, metric_mean(metric_adapter), metric; kwargs...
                    )

            lambda = sqrt.(lambda_max(pca_adapter))
            update!(steps_adapter, m + 1, lambda, sampler.stepsize, pca_adapter.opca.n[1]; kwargs...)

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

            set!(sampler, steps_adapter; kwargs...)

            set!(sampler, damping_adapter; kwargs...)

            calculate_nextwindow!(schedule)
        end
    else
        set!(sampler, stepsize_adapter; smoothed=true, kwargs...)
        set!(sampler, metric_adapter)
        set!(sampler, pca_adapter)
        set!(sampler, damping_adapter)
        set!(sampler, noise_adapter)
        set!(sampler, steps_adapter)
        set!(sampler, reductionfactor_adapter; smoothed=true, kwargs...)
    end
end
