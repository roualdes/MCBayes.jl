# adapted from https://github.com/modichirag/hmc/blob/main/src/algorithms.py#L463
abstract type AbstractDHMC{T} <: AbstractSampler{T} end

struct DHMC{T} <: AbstractDHMC{T}
    momentum::Matrix{T}
    metric::Matrix{T}
    pca::Matrix{T}
    steps::Vector{T}
    stepsize::Vector{T}
    reductionfactor::Vector{T}
    acceptanceprob::Vector{T}
    maxtreedepth::Int
    maxdeltaH::T
    proposal_mean::Matrix{T}
    dims::Int
    chains::Int
end

# TODO(ear) remove ability to set placeholders such as metric,
# stepsize, trajectorylength, noise, damping, etc. from all sampler
# constructors, as it conflates where I want the defaults to be set.
# I want the API to set the defaults in the adapters.
function DHMC(
    dims,
    chains=12,
    T=Float64;
    metric=ones(T, dims, chains),
    pca=randn(T, dims, 1),
    stepsize=ones(T, chains),
    reductionfactor=ones(T, 1),
    steps=10 * ones(T, 1),
    maxtreedepth = 10,
    maxdeltaH=convert(T, 1000)::T,
    proposal_mean = zeros(T, dims, chains)
    )
    momentum = randn(T, dims, chains)
    D = convert(Int, dims)::Int
    pca ./= mapslices(norm, pca, dims = 1)
    acceptanceprob = 2 .* rand(T, chains) .- 1
    return DHMC(
        momentum, metric, pca, steps, stepsize, reductionfactor, acceptanceprob, maxtreedepth, maxdeltaH, proposal_mean, D, chains
    )
end

function sample!(
    sampler::DHMC,
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
    proposal_adapter=MetricOnlineMoments(sampler.proposal_mean),
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
        proposal_adapter,
        kwargs...,
    )
end

function transition!(sampler::DHMC, m, ldg!, draws, rngs, trace;
                     J = 3,
                     kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)
    warmup = kwargs[:warmup]

    metric = sampler.metric ./ maximum(sampler.metric, dims = 1)

    # reduction_factor = sampler.reductionfactor[1]
    # stepsize = sampler.stepsize[1]
    steps = ceil(Int, rand() * sampler.steps[1])

    println("iteration $m")
    println("steps = $(sampler.steps)")
    println("stepsize = $(sampler.stepsize)")
    # println("reduction factor = $(sampler.reductionfactor)")

    @sync for it in 1:nt
        Threads.@spawn for chain in it:nt:chains
            if !kwargs[:adaptation_schedule].startHMC
                @views info = stan_kernel!(
                    draws[m, :, chain],
                    draws[m + 1, :, chain],
                    rngs[chain],
                    sampler.dims,
                    metric[:, chain],
                    sampler.stepsize[chain],
                    sampler.maxdeltaH,
                    sampler.maxtreedepth,
                    ldg!;
                    kwargs...,
                )
            else
                @views info = hmc!(
                    draws[m, :, chain],
                    draws[m + 1, :, chain],
                    ldg!,
                    rngs[chain],
                    sampler.dims,
                    metric[:, chain],
                    sampler.stepsize[chain],
                    steps,
                    1000;
                    kwargs...,
                )
            end
            info = (; info...,
                    pca = sampler.pca);
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end

function adapt!(
    sampler::DHMC,
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
    proposal_adatper;
    kwargs...,
    )

    warmup = schedule.warmup
    if m <= warmup

        accept_stats = trace.acceptstat[m + 1, :]
        update!(stepsize_adapter, accept_stats; kwargs...)
        set!(sampler, stepsize_adapter; kwargs...)

        if schedule.firstwindow <= m < schedule.lastwindow

            positions = draws[m + 1, :, :]
            update!(metric_adapter, positions, ldg!; kwargs...)

            scale = metric_adapter.metric
            scale ./= maximum(scale, dims = 1)
            scale .= sqrt(scale)
            location = metric_mean(metric_adapter)

            update!(
                pca_adapter, positions, location, scale; kwargs...
                    )
            if any(isnan.(pca_adapter.opca.pc))
                reset!(pca_adapter; reset_pc = true)
            end

            println("adapt trajectorylength")
            ev = pca_adapter.opca.pc
            ev ./ mapslices(norm, ev, dims = 1)
            iterations_since_refresh = m + 1 - schedule.previousclosewindow
            update!(proposal_adapter, trace.proposedq; kwargs...)

            update!(steps_adapter,
                    iterations_since_refresh,
                    accept_stats,
                    draws[m, :, :] .- location,
                    trace.proposedq .- metric_mean(proposal_adapter),
                    trace.previousmomentum,
                    trace.proposedp,
                    mean(sampler.stepsize),
                    ev,
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

            set!(sampler, metric_adapter; kwargs...)
            reset!(metric_adapter)

            set!(sampler, pca_adapter; kwargs...)
            reset!(pca_adapter; reset_pc = true)

            reset!(steps_adapter, lambda_max(pca_adapter), mean(sampler.stepsize); kwargs...)

            calculate_nextwindow!(schedule)
        end
    else
        set!(sampler, stepsize_adapter; kwargs...)
        set!(sampler, pca_adapter; kwargs...)
        set!(sampler, metric_adapter; kwargs...)
    end
end
