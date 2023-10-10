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
    drift::Vector{T}
    acceptanceprob::Vector{T}
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
    drift = (1 .- noise .^ 2) ./ 2
    acceptanceprob = 2 .* rand(T, chains) .- 1
    return DrMALA(
        momentum, metric, pca, steps, stepsize, reductionfactor, damping, noise, drift, acceptanceprob, D, chains
    )
end

function sample!(
    sampler::DrMALA,
    ldg!;
    iterations=2000,
    warmup=iterations,
    draws_initializer=DrawsInitializerStan(),
    stepsize_initializer=StepsizeInitializerStan(),
    steps_adapter=StepsPCA(copy(sampler.steps)),
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize; δ=0.5),
    reductionfactor_adapter=ReductionFactorConstant(sampler.reductionfactor; reductionfactor_δ = 0.9),
    # reductionfactor_adapter=ReductionFactorDualAverage(sampler.reductionfactor; reductionfactor_δ = 0.9),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    pca_adapter=PCAOnline(sampler.pca),
    damping_adapter=DampingMALT(sampler.damping),
    noise_adapter=NoiseMALT(sampler.noise),
    adaptation_schedule=WindowedAdaptationSchedule(warmup; windowsize = 12),
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
                     J = 3,
                     nonreversible_update = false,
                     refresh_momenta = false,
                     persist_momenta = true,
                     kwargs...)
    nt = get(kwargs, :threads, Threads.nthreads())
    chains = size(draws, 3)

    if refresh_momenta
        randn!(sampler.momentum)
    end

    nru = m > kwargs[:warmup] && nonreversible_update

    reduction_factor = sampler.reductionfactor[1]
    damping = sampler.damping[1]
    metric = sampler.metric[:, 1]
    metric ./= maximum(metric)
    drift = sampler.drift[1]

    Threads.@threads for it in 1:nt
        for chain in it:nt:chains
            stepsize = sampler.stepsize[chain]
            steps = sampler.steps[chain]
            noise = sampler.noise[chain]
            acceptanceprob = sampler.acceptanceprob[chain:chain]

            local info
            acceptstats = zeros(steps)
            finalacceptstats = zeros(steps)
            leapfrog = zeros(Int, steps)
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
                    drift,
                    acceptanceprob,
                    J,
                    reduction_factor,
                    nru,
                    persist_momenta,
                    1000;
                    kwargs...,
                )
                lastposition .= draws[m + 1, :, chain]
                acceptstats[step] = info[:acceptstat]
                finalacceptstats[step] = info[:finalacceptstat]
                leapfrog[step] = info[:leapfrog]
            end
            info = (; info...,
                    damping,
                    leapfrog = sum(leapfrog),
                    reductionfactor = reduction_factor,
                    acceptstat = mean(acceptstats),
                    finalacceptstat = maybe_mean(finalacceptstats))
            record!(sampler, trace, info, m + 1, chain)
        end
    end
end

function hmc_mixing!(m, draws, momentum, ldg!, rngs, stepsize_adapter, metric; maxdeltaH = 1000, mixing_windowsize = 20, numsignchanges = 4, max_leapfrogs = 20_000, kwargs...)
    T = eltype(draws)
    _, dims, chains = size(draws)

    gradient = zeros(T, dims)
    lds = zeros(T, mixing_windowsize, chains)

    idx = m-mixing_windowsize:m-1
    for w in 1:mixing_windowsize
        for chain in 1:chains
            lds[w, chain] = ldg!(draws[idx[w], :, chain], gradient; kwargs...)
        end
    end

    signchanges = mapslices(cummean_signchanges, lds, dims = 1)
    if all(signchanges .>= numsignchanges)
        return
    end

    @views current_draw = draws[m + 1, :, :]
    @views next_draw = draws[m + 2, :, :]
    accept_stats = zeros(T, chains)

    num_leapfrogs = 0

    for l in 1:max_leapfrogs
        for w in 1:mixing_windowsize-1
            lds[w, :] .= lds[w+1, :]
        end

        S = Iterators.cycle(1:length(stepsize_adapter.stepsize))
        for (chain, s) in zip(1:chains, S)
            stepsize = stepsize_adapter.stepsize[s]

            @views info = hmc_momentum!(current_draw[:, chain],
                                        next_draw[:, chain],
                                        momentum[:, chain],
                                        ldg!, rngs[chain],
                                        dims,
                                        metric,
                                        stepsize,
                                        1,
                                        maxdeltaH;
                                        kwargs...)

            lds[end, chain] = ldg!(next_draw[:, chain], gradient; kwargs...)

            accept_stats[chain] = info.acceptstat
            if !info.accepted
                momentum[:, chain] .= randn(rngs[chain], T, dims)
            end
        end

        signchanges = mapslices(cummean_signchanges, lds, dims = 1)
        if all(signchanges .>= numsignchanges)
            num_leapfrogs = l
            break
        end

        update!(stepsize_adapter, mean(accept_stats), m + l; kwargs...)
        current_draw .= next_draw
        num_leapfrogs = l
    end

    draws[m + 2, :, :] .= next_draw
    return num_leapfrogs
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
    steps_coefficient = 1,
    kwargs...,
    )

    warmup = schedule.warmup
    if m <= warmup
        accept_stats = trace.acceptstat[m + 1, :]
        update!(stepsize_adapter, accept_stats; kwargs...)
        set!(sampler, stepsize_adapter; kwargs...)

        update!(noise_adapter, sampler.damping, sampler.stepsize; kwargs...)
        set!(sampler, noise_adapter; kwargs...)
        sampler.drift .= (1 .- sampler.noise .^ 2) ./ 2

        if schedule.firstwindow - 1 <= m < schedule.firstwindow
            hmc_mixing!(m, draws, sampler.momentum, ldg!, rngs, stepsize_adapter;  kwargs...)

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
        end

        if schedule.firstwindow <= m <= schedule.lastwindow
            final_accept_stats = trace.finalacceptstat[m + 1, :]
            update!(reductionfactor_adapter, maybe_mean(final_accept_stats); kwargs...)
            set!(sampler, reductionfactor_adapter; kwargs...)

            positions = draws[m + 1, :, :]

            update!(metric_adapter, positions, ldg!; kwargs...)

            metric = sqrt.(metric_adapter.metric)
            metric ./= maximum(metric, dims = 1)

            update!(
                pca_adapter, positions, metric_mean(metric_adapter), metric; kwargs...
                    )

            lambda = sqrt.(lambda_max(pca_adapter))
            update!(steps_adapter, m + 1, steps_coefficient * lambda, maximum(sampler.stepsize); kwargs...)
            set!(sampler, steps_adapter; kwargs...)

            update!(damping_adapter, m + 1, lambda, sampler.stepsize; kwargs...)
            set!(sampler, damping_adapter; kwargs...)
        end

        if m == schedule.closewindow
            calculate_nextwindow!(schedule)

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
