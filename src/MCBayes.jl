module MCBayes

using Random
using LinearAlgebra
using Statistics

abstract type AbstractSampler{T<:AbstractFloat} end

Base.eltype(::AbstractSampler{T}) where {T} = T

struct EnsembleChainSchedule end

include("windowedadaptation.jl")
include("dualaverage.jl")
include("adam.jl")
include("onlinemoments.jl")
include("adapt.jl")

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
include("meads.jl")

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
            trajectorylength_adapter,
            damping_adapter,
            noise_adapter,
            drift_adapter;
            kwargs...,
        )
    end
    return draws, diagnostics, rngs
end

# precompile
function ldg(x)
    -x' * x / 2, -x
end

stan = Stan(10, 4)
draws, diagnostics, rngs = sample!(stan, ldg)
ess_mean(draws)
rhat_basic(draws)

mead = MEADS(10)
draws, diagnostics, rngs = sample!(mead, ldg)

end
