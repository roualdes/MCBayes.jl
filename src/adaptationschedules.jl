mutable struct WindowedAdaptationSchedule
    closewindow::Int
    startHMC::Bool
    previousclosewindow::Int
    windowsize::Int
    const warmup::Int
    const firstwindow::Int
    const lastwindow::Int
end

function WindowedAdaptationSchedule(warmup; initbuffer=75, termbuffer=100, windowsize=25)
    # TODO probably want some reasonable checks on warmup values
    return WindowedAdaptationSchedule(
        initbuffer + windowsize, false, initbuffer, windowsize, warmup, initbuffer, warmup - termbuffer
    )
end

function calculate_nextwindow!(ws::WindowedAdaptationSchedule)
    ws.windowsize *= 2
    nextclosewindow = ws.closewindow + ws.windowsize
    ws.previousclosewindow = ws.closewindow
    if ws.closewindow + 2 * ws.windowsize > ws.lastwindow
        ws.closewindow = ws.lastwindow
    else
        ws.closewindow = min(nextclosewindow, ws.lastwindow)
        if nextclosewindow + 4 * ws.windowsize > ws.lastwindow
            ws.startHMC = true
        end
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
        update!(stepsize_adapter, accept_stats, m + 1; warmup, kwargs...)
        set!(sampler, stepsize_adapter; kwargs...)

        if schedule.firstwindow <= m <= schedule.lastwindow
            update!(metric_adapter, draws[m + 1, :, :], ldg; kwargs...)
        end

        if m == schedule.closewindow
            initialize_stepsize!(
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
            reset!(metric_adapter; kwargs...)

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

struct SGAAdaptationSchedule
    warmup::Int
end

function adapt!(
    sampler,
    schedule::SGAAdaptationSchedule,
    trace,
    m,
    ldg,
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
    stepsize_delay=0,
    trajectorylength_delay=100,
    kwargs...,
)
    warmup = schedule.warmup
    if m <= warmup

        positions = draws[m + 1, :, :]
        update!(metric_adapter, positions, ldg; kwargs...)
        set!(sampler, metric_adapter; kwargs...)

        if :pca in fieldnames(typeof(sampler))
            metric = sqrt.(sampler.metric[:, 1])
            mn, mx = extrema(metric)
            metric ./= mx #.*= mn ./ mx #

            update!(
                pca_adapter, (positions .- metric_mean(metric_adapter)) ./ metric; kwargs...
            )
            set!(sampler, pca_adapter; kwargs...)
        end

        if :damping in fieldnames(typeof(sampler))
            update!(damping_adapter, m + 1, sqrt(norm(sampler.pca)); kwargs...)
            set!(sampler, damping_adapter; kwargs...)

            update!(noise_adapter, sampler.damping, sampler.stepsize; kwargs...)
            set!(sampler, noise_adapter; kwargs...)
        end

        T = eltype(trace.acceptstat)
        accept_stats = [isnan(as) ? zero(T) : as for as in trace.acceptstat[m + 1, :]]
        accept_stats .+= 1e-20

        if m > stepsize_delay
            abar = mean(accept_stats) # inv(mean(inv, accept_stats)) #
            update!(stepsize_adapter, abar, m + 1; warmup, kwargs...)
            set!(sampler, stepsize_adapter; kwargs...)
        end

        if :trajectorylength in fieldnames(typeof(sampler))
            if m > trajectorylength_delay
                update!(
                    trajectorylength_adapter,
                    m + 1,
                    accept_stats,
                    draws[m, :, :],
                    trace.previousmomentum,
                    trace.proposedp,
                    trace.proposedq,
                    sampler.stepsize[1],
                    sampler.pca ./ norm(sampler.pca),
                    ldg;
                    kwargs...,
                )
                set!(sampler, trajectorylength_adapter; kwargs...)
            else
                trajectorylength_adapter.trajectorylength[1] = sampler.stepsize[1]
                set!(sampler, trajectorylength_adapter; kwargs...)
            end
        end
    else
        set!(sampler, stepsize_adapter; smoothed=true, kwargs...)
        set!(sampler, trajectorylength_adapter; smoothed=true, kwargs...)
    end
end

struct NoAdaptationSchedule end

function adapt!(
    sampler,
    schedule::NoAdaptationSchedule,
    trace,
    m,
    ldg,
    draws,
    rngs,
    metric_adapter,
    stepsize_initializer,
    stepsize_adapter,
    reductionfactor_adapter,
    steps_adapter,
    trajectorylength_adapter,
    damping_adapter,
    noise_adapter,
    drift_adapter;
    kwargs...,
) end
