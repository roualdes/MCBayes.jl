mutable struct WindowedAdaptationSchedule
    closewindow::Int
    windowsize::Int
    const warmup::Int
    const firstwindow::Int
    const lastwindow::Int
end

function WindowedAdaptationSchedule(warmup; initbuffer=75, termbuffer=50, windowsize=25)
    # TODO probably want some reasonable checks on warmup values
    return WindowedAdaptationSchedule(
        initbuffer + windowsize, windowsize, warmup, initbuffer, warmup - termbuffer
    )
end

function calculate_nextwindow!(ws::WindowedAdaptationSchedule)
    ws.windowsize *= 2
    nextclosewindow = ws.closewindow + ws.windowsize
    return ws.closewindow = if ws.closewindow + 2 * ws.windowsize > ws.lastwindow
        ws.lastwindow
    else
        min(nextclosewindow, ws.lastwindow)
    end
end

function adapt!(
    sampler,
    schedule::WindowedAdaptationSchedule,
    trace,
    m,
    ldg,
    draws,
    rngs,
    metric_adapter,
    stepsize_initializer,
    stepsize_adapter,
    trajectorylength_adapter,
    damping_adapter,
    noise_adapter,
    drift_adapter;
    kwargs...,
)
    warmup = schedule.warmup
    if m <= warmup
        accept_stats = trace.acceptstat[m, :]
        update!(stepsize_adapter, accept_stats; warmup, kwargs...)
        set!(sampler, stepsize_adapter; kwargs...)

        # TODO(ear) this is attempting to plan ahead;
        # to actually use update!() will require
        # more arguments, for additional information on which
        # the trajectorylength could be learned; re SGA methods
        update!(trajectorylength_adapter; kwargs...)
        set!(sampler, trajectorylength_adapter; kwargs...)

        if schedule.firstwindow <= m <= schedule.lastwindow
            @views update!(metric_adapter, draws[m + 1, :, :], ldg; kwargs...)
        end

        if m == schedule.closewindow
            @views initialize_stepsize!(
                stepsize_initializer,
                stepsize_adapter,
                sampler,
                rngs,
                ldg,
                draws[m + 1, :, :];
                kwargs...,
            )
            set!(sampler, stepsize_adapter; kwargs...)
            reset!(stepsize_adapter; kwargs...)

            set!(sampler, metric_adapter; kwargs...)
            reset!(metric_adapter)

            calculate_nextwindow!(schedule)
        end
    else
        set!(sampler, stepsize_adapter; smoothed=true, kwargs...)
    end
end

struct EnsembleChainSchedule end

function adapt!(
    sampler,
    schedule::EnsembleChainSchedule,
    trace,
    m,
    ldg,
    draws,
    rngs,
    metric_adapter,
    stepsize_initializer,
    stepsize_adapter,
    trajectorylength_adapter,
    damping_adapter,
    noise_adapter,
    drift_adapter;
    kwargs...,
)
    nt = get(kwargs, :threads, Threads.nthreads())

    @sync for it in 1:nt
        Threads.@spawn for f in it:nt:(sampler.folds)
            k = (f + 1) % sampler.folds + 1
            kfold = sampler.partition[:, k]

            positions = draws[m + 1, :, kfold]
            z_positions, sigma = standardize_draws(positions)

            update!(metric_adapter, sigma .^ 2, f; kwargs...)
            set!(sampler, metric_adapter, f; kwargs...)

            update!(stepsize_adapter, ldg, positions, sigma, f; kwargs...)
            set!(sampler, stepsize_adapter, f; kwargs...)

            update!(damping_adapter, m, z_positions, sampler.stepsize, f; kwargs...)
            set!(sampler, damping_adapter, f; kwargs...)

            update!(noise_adapter, sampler.damping, f; kwargs...)
            set!(sampler, noise_adapter, f; kwargs...)

            update!(drift_adapter, sampler.noise, f; kwargs...)
            set!(sampler, drift_adapter, f; kwargs...)
        end
    end
end
