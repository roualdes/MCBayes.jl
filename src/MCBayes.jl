module MCBayes

using Random
using LinearAlgebra
using Statistics

abstract type AbstractSampler{T<:AbstractFloat} end

Base.eltype(::AbstractSampler{T}) where {T} = T

include("adaptationschedules.jl")
include("dualaverage.jl")
include("adam.jl")
include("onlinemoments.jl")
include("onlinepca.jl")

include("stepsize_adapter.jl")
include("trajectorylength_adapter.jl")
include("metric_adapter.jl")
include("pca_adapter.jl")
include("damping_adapter.jl")
include("drift_adapter.jl")
include("noise_adapter.jl")

include("sampler_initializer.jl")
include("draws_initializer.jl")
include("stepsize_initializer.jl")

include("stan.jl")
include("rwm.jl")
include("meads.jl")
include("mala.jl")
include("malt.jl")
include("sga.jl")
include("xhmc.jl")
include("drmala.jl")

include("tools.jl")
include("integrator.jl")
include("pspoint.jl")
include("trace.jl")

include("convergence.jl")

export Stan,
    RWM,
    MEADS,
    MALA,
    MALT,
    ChEES,
    SNAPER,
    XHMC,
    DrMALA,
    WindowedAdaptationSchedule,
    NoAdaptationSchedule,
    SGAAdaptationSchedule,
    StepsizeInitializer,
    StepsizeInitializerStan,
    StepsizeInitializerMEADS,
    StepsizeInitializerRWM,
    StepsizeInitializerSGA,
    DrawsInitializer,
    DrawsInitializerStan,
    DrawsInitializerRWM,
    DrawsInitializerAdam,
    PCAOnline,
    MetricOnlineMoments,
    MetricConstant,
    MetricECA,
    MetricFisherDivergence,
    StepsizeAdam,
    StepsizeDualAverage,
    StepsizeConstant,
    StepsizeECA,
    TrajectorylengthAdam,
    TrajectorylengthConstant,
    TrajectorylengthChEES,
    TrajectorylengthSNAPER,
    TrajectorylengthLDG,
    DampingECA,
    DampingMALT,
    DampingConstant,
    DriftECA,
    DriftConstant,
    NoiseECA,
    NoiseMALT,
    NoiseConstant,
    sample!,
    # ess_bulk, # TODO wait until https://github.com/JuliaLang/julia/pull/47040
    ess_tail,
    ess_quantile,
    ess_mean,
    ess_sq,
    ess_std,
    # rhat,  # TODO wait until https://github.com/JuliaLang/julia/pull/47040
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
    rngs=Random.Xoshiro.(
        rand(1:typemax(Int), hasfield(typeof(sampler), :chains) ? sampler.chains : 4)
    ),
    draws_initializer=DrawsInitializer(),
    stepsize_initializer=StepsizeInitializer(),
    stepsize_adapter=StepsizeConstant(
        hasfield(typeof(sampler), :stepsize) ? sampler.stepsize : ones(1)
    ),
    trajectorylength_adapter=TrajectorylengthConstant(
        hasfield(typeof(sampler), :trajectorylength) ? sampler.trajectorylength : ones(1)
    ),
    metric_adapter=MetricConstant(
        hasfield(typeof(sampler), :metric) ? sampler.metric : ones(1)
    ),
    pca_adapter=PCAConstant(hasfield(typeof(sampler), :pca) ? sampler.pca : zeros(1)),
    damping_adapter=DampingConstant(
        hasfield(typeof(sampler), :damping) ? sampler.damping : zeros(1)
    ),
    noise_adapter=NoiseConstant(
        hasfield(typeof(sampler), :noise) ? sampler.noise : zeros(1)
    ),
    drift_adapter=DriftConstant(
        hasfield(typeof(sampler), :drift) ? sampler.drift : zeros(1)
    ),
    adaptation_schedule=WindowedAdaptationSchedule(warmup),
    kwargs...,
) where {T<:AbstractFloat}
    M = iterations + warmup
    draws = Array{T,3}(undef, M + 1, sampler.dims, sampler.chains)
    diagnostics = trace(sampler, M + 1)

    initialize_sampler!(
        sampler;
        stepsize_adapter,
        trajectorylength_adapter,
        metric_adapter,
        damping_adapter,
        noise_adapter,
        drift_adapter,
    )

    initialize_draws!(draws_initializer, draws, rngs, ldg; kwargs...)

    initialize_stepsize!(
        stepsize_initializer,
        stepsize_adapter,
        sampler,
        rngs,
        ldg,
        draws[1, :, :];
        kwargs...,
    )

    for m in 1:M
        transition!(sampler, m, ldg, draws, rngs, diagnostics; warmup, kwargs...)

        # TODO adaptations effectively should be unique to each algorithm
        # adaptation schedules don't generalize well
        adapt!(
            sampler,
            adaptation_schedule,
            diagnostics,
            m,
            ldg,
            draws,
            rngs,
            metric_adapter,
            pca_adapter,
            stepsize_initializer,
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
function ldg(x; kwargs...)
    -x' * x / 2, -x
end

stan = Stan(10)
draws, diagnostics, rngs = sample!(stan, ldg)
ess_mean(draws)
rhat_basic(draws)

meads = MEADS(10)
draws, diagnostics, rngs = sample!(meads, ldg)

mala = MALA(10)
draws, diagnostics, rngs = sample!(mala, ldg)

chees = ChEES(10)
draws, diagnostics, rngs = sample!(chees, ldg)

end
