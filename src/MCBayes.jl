module MCBayes

using Random
using LinearAlgebra
using Statistics

abstract type AbstractSampler{T<:AbstractFloat} end

Base.eltype(::AbstractSampler{T}) where {T} = T

include("windowedadaptation.jl")

include("dualaverage.jl")
include("adam.jl")
include("onlinemoments.jl")

include("stepsize_adapter.jl")
include("trajectorylength_adapter.jl")
include("metric_adapter.jl")
include("damping_adapter.jl")
include("drift_adapter.jl")
include("noise_adapter.jl")

include("initialize_draws.jl")
include("initialize_stepsize.jl")

include("stan.jl")
include("mh.jl")

include("tools.jl")
include("integrator.jl")
include("pspoint.jl")
include("trace.jl")

include("convergence.jl")

export Stan,
    MH,
    OnlineMoments,
    MetricOnlineMoments,
    MetricConstant,
    StepsizeAdam,
    StepsizeDualAverage,
    StepsizeConstant,
    TrajectorylengthAdam,
    TrajectorylengthConstant,
    DampingECA,
    DampingConstant,
    DriftECA,
    DriftConstant,
    NoiseECA,
    NoiseConstant,
    sample!,
    # ess_bulk, # wait until https://github.com/JuliaLang/julia/pull/47040
    ess_tail,
    ess_quantile,
    ess_mean,
    ess_sq,
    ess_std,
    # rhat,  # wait until https://github.com/JuliaLang/julia/pull/47040
    rhat_basic,
    mcse_mean,
    mcse_std

"""
    run_sampler!(sampler, ldg)

Sample from `sampler` object.  User must provide a function `ldg(position;
kwargs...)` which accepts `position::Vector` and returns a tuple containing
the evaluation of the joint log density function and a vector of the gradient,
each evaluated at the argument `position`.
"""
function run_sampler!(
    sampler::AbstractSampler{T},
    ldg;
    iterations=1000,
    warmup=iterations,
    rngs=Random.Xoshiro.(rand(1:typemax(Int), sampler.chains)),
    draws_initializer=:stan,
    stepsize_adapter=StepsizeConstant(sampler.stepsize),
    trajectorylength_adapter=TrajectorylengthConstant(zeros(sampler.chains)),
    metric_adapter=MetricConstant(sampler.metric),
    damping_adapter=DampingConstant(1 ./ sampler.stepsize),
    noise_adapter=NoiseConstant(
        1 .- exp.(-2 .* damping_adapter.damping .* stepsize_adapter.stepsize)
    ),
    drift_adapter=DriftConstant(noise_adapter.noise ./ 2),
    adaptation_schedule=WindowedAdaptationSchedule(warmup),
    kwargs...,
) where {T<:AbstractFloat}
    M = iterations + warmup
    draws = Array{T,3}(undef, M + 1, sampler.dims, sampler.chains)
    diagnostics = trace(sampler, M + 1)

    initialize_draws!(draws_initializer, draws, rngs, ldg; kwargs...)

    @views initialize_stepsize!(
        stepsize_adapter, sampler.metric, rngs, ldg, draws[1, :, :]; kwargs...
    )
    set_stepsize!(sampler, stepsize_adapter; kwargs...)

    for m in 1:M
        transition!(sampler, m, ldg, draws, rngs, diagnostics; kwargs...)

        adapt!(
            sampler,
            adaptation_schedule,
            diagnostics,
            m,
            ldg,
            draws,
            rngs,
            metric_adapter,
            stepsize_adapter,
            trajectorylength_adapter;
            kwargs...,
        )
    end
    return draws, diagnostics, rngs
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
    stepsize_adapter,
    trajectorylength_adapter;
    kwargs...,
)
    warmup = schedule.warmup
    if m <= warmup
        accept_stats = trace.acceptstat[m, :]
        update!(stepsize_adapter, accept_stats; warmup, kwargs...)
        set_stepsize!(sampler, stepsize_adapter; kwargs...)

        # TODO(ear) this is attempting to plan ahead;
        # to actually use update!() will require
        # more arguments, for additional information on which
        # the trajectorylength could be learned; re SGA methods
        update!(trajectorylength_adapter; kwargs...)
        set_trajectorylength!(sampler, trajectorylength_adapter; kwargs...)

        if schedule.firstwindow <= m <= schedule.lastwindow
            @views update!(metric_adapter, draws[m + 1, :, :]; kwargs...)
        end

        if m == schedule.closewindow
            @views initialize_stepsize!(
                stepsize_adapter,
                optimum(metric_adapter),
                rngs,
                ldg,
                draws[m + 1, :, :];
                kwargs...,
            )
            set_stepsize!(sampler, stepsize_adapter; kwargs...)
            reset!(stepsize_adapter; kwargs...)

            set_metric!(sampler, metric_adapter; kwargs...)
            reset!(metric_adapter)

            calculate_nextwindow!(schedule)
        end
    else
        set_stepsize!(sampler, stepsize_adapter; smoothed=true, kwargs...)
    end
end

# precompile
function ldg(x)
    -x' * x / 2, -x
end

stan = Stan(10, 4)
draws, diagnostics, rngs = sample!(stan, ldg)
ess_mean(draws)
rhat_basic(draws)

end
