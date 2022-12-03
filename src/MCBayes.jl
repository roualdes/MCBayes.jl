module MCBayes

abstract type AbstractAdaptationSchedule end
abstract type AbstractAdapter end

include("onlinemoments.jl")

function adapt!(sampler, schedule::WindowedAdaptationSchedule,
                metric_adapter, stepsize_adapter, trajectorylength_adapter,
                rng, i, ldg, draws; kwargs...)
    if i <= schedule.warmup
        adapt_stepsize!(sampler, schedule, stepsize_adapter, i, draws; kwargs...)
        set_stepsize!(sampler, schedule, stepsize_adapter, i, draws; kwargs...)

        adapt_metric!(sampler, schedule, metric_adapter, i, draws; kwargs...)
        set_metric!(sampler, schedule, metric_adapter, i, draws; kwargs...)

        adapt_trajectorylength!(sampler, schedule, trajectorylength_adapter, i, draws; kwargs...)
        set_trajectorylength!(sampler, schedule, trajectorylength_adapter, i, draws; kwargs...)

        if i == schedule.closewindow
            initialize_stepsize!(stepsize_adapter,
                                 metric(metric_adapter),
                                 rng,
                                 ldg,
                                 draws,
                                 gradient;
                                 get(kwargs, integrator, :leapfrog),
                                 kwargs...)
            reset!(stepsize_adapter)
            reset!(metric_adapter)
            calculate_nextwindow!(schedule)
        end
    else
        set_stepsize!(sampler, schedule, stepsize_adapter, i, ldg, draws;
                      weighted_average = true, kwargs...)
    end
end


export
    OnlineMoments,
    update!,
    reset!,
    metric,

    initialize_draws!,
    initialize_stepsize!,
    #initialize_trajectorylength!,

    transition,
    adapt!,
    adapt_stepsize!,
    set_stepsize!,
    adapt_metric!,
    set_metric!,
    adapt_trajectorylength!,
    set_trajectorylength!

end
