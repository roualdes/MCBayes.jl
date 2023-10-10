# adapted from https://github.com/modichirag/hmc/blob/main/src/algorithms.py#L463
abstract type AbstractDRHMC{T} <: AbstractSampler{T} end

struct DRHMC{T} <: AbstractDRHMC{T}
    momentum::Matrix{T}
    metric::Matrix{T}
    pca::Matrix{T}
    steps::Vector{T}
    stepsize::Vector{T}
    reductionfactor::Vector{T}
    acceptanceprob::Vector{T}
    dims::Int
    chains::Int
end

# TODO(ear) remove ability to set placeholders such as metric,
# stepsize, trajectorylength, noise, damping, etc. from all sampler
# constructors, as it conflates where I want the defaults to be set.
# I want the API to set the defaults in the adapters.
function DRHMC(
    dims,
    chains=12,
    T=Float64;
    metric=ones(T, dims, 1),
    pca=randn(T, dims, 1),
    stepsize=ones(T, 1),
    reductionfactor=ones(T, 1),
    steps=10 * ones(T, 1),
    )
    momentum = randn(T, dims, chains)
    D = convert(Int, dims)::Int
    pca ./= mapslices(norm, pca, dims = 1)
    # damping = ones(T, 1)
    # noise = exp.(-2 .* damping .* stepsize)
    # drift = (1 .- noise .^ 2) ./ 2
    acceptanceprob = 2 .* rand(T, chains) .- 1
    return DRHMC(
        momentum, metric, pca, steps, stepsize, reductionfactor, acceptanceprob, D, chains
    )
end

function sample!(
    sampler::DRHMC,
    ldg!;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerStan(),
    steps_adapter = StepsAdamSNAPER(sampler.steps, fill(norm(randn(sampler.dims)), sampler.chains), sampler.dims, warmup, adam_schedule = :linear),
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize; δ=0.8),
    reductionfactor_adapter=ReductionFactorConstant(sampler.reductionfactor, sampler.chains; reductionfactor_δ = 0.95),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    pca_adapter=PCAOnline(sampler.pca),
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
        adaptation_schedule,
        kwargs...,
    )
end

function transition!(sampler::DRHMC, m, ldg!, draws, rngs, trace;
                     J = 3,
                     nonreversible_update = false,
                     refresh_momenta = true,
                     persist_momenta = false,
                     kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)

    nru = m > kwargs[:warmup] && nonreversible_update

    metric = sampler.metric[:, 1]
    metric ./= maximum(metric)

    reduction_factor = sampler.reductionfactor[1]
    stepsize = sampler.stepsize[1]
    steps = ceil(Int, rand() * sampler.steps[1])

    println("iteration $m")
    println("steps = $(sampler.steps)")
    # println("stepsize = $(sampler.stepsize)")
    # println("reduction factor = $(sampler.reductionfactor)")

    @sync for it in 1:nt
        Threads.@spawn for chain in it:nt:chains
            # @views info = hmc!(
            #     draws[m, :, chain],
            #     draws[m + 1, :, chain],
            #     ldg!,
            #     rngs[chain],
            #     sampler.dims,
            #     metric,
            #     stepsize,
            #     steps,
            #     1000;
            #     kwargs...,
            # )
            # info = (; info..., momentum = randn(sampler.dims),
            #         reduction_factor = 1, retries = 1, firsttry = 1,
            #         finalacceptstat = info.acceptstat)
            @views info = drhmc!(
                draws[m, :, chain],
                draws[m + 1, :, chain],
                ldg!,
                rngs[chain],
                sampler.dims,
                metric,
                stepsize,
                steps,
                1,
                sampler.acceptanceprob[chain:chain],
                J,
                4, # reduction_factor,
                nru,
                1000;
                kwargs...,
            )
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end

function adapt!(
    sampler::DRHMC,
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

        final_accept_stats = [isnan(a) ? zero(a) : a for a in trace.finalacceptstat[m + 1, :]]
        final_accept_stats .+= 1e-20
        update!(stepsize_adapter, mean(final_accept_stats); kwargs...)
        set!(sampler, stepsize_adapter; kwargs...)

        if schedule.firstwindow <= m < schedule.lastwindow

            update!(reductionfactor_adapter, trace.accepted[m + 1, :]; kwargs...)
            set!(sampler, reductionfactor_adapter; kwargs...)

            positions = draws[m + 1, :, :]

            update!(metric_adapter, positions, ldg!; kwargs...)

            metric = metric_adapter.metric
            metric ./= sqrt.(maximum(metric, dims = 1))

            update!(
                pca_adapter, positions, metric_mean(metric_adapter), metric; kwargs...
                    )

            iterations_since_refresh = m + 1 - schedule.previousclosewindow

            update!(steps_adapter,
                    iterations_since_refresh,
                    final_accept_stats,
                    draws[m, :, :],
                    trace.proposedq,
                    trace.previousmomentum,
                    trace.proposedp,
                    # sampler.stepsize[1],
                    mean(trace.stepsize[m + 1, :]), # effectively stepsize / reduction_factor
                    sampler.pca ./ mapslices(norm, sampler.pca, dims = 1),
                    ldg!;
                    kwargs...)
            set!(sampler, steps_adapter; kwargs...)
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

            reset!(steps_adapter, lambda_max(pca_adapter)[1], sampler.stepsize[1]; kwargs...)

            set!(sampler, pca_adapter; kwargs...)
            reset!(pca_adapter; reset_pc = false)

            set!(sampler, damping_adapter; kwargs...)

            calculate_nextwindow!(schedule)
        end
    else
        set!(sampler, stepsize_adapter; kwargs...)
        set!(sampler, pca_adapter; kwargs...)
        set!(sampler, metric_adapter; kwargs...)
        set!(sampler, steps_adapter; kwargs...)
    end
end
