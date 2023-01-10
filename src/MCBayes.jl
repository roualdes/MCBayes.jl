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

function ldg(x)
    -x' * x / 2, -x
end

stan = Stan(10, 4)
draws, diagnostics, rngs = sample!(stan, ldg)
ess_mean(draws)
rhat_basic(draws)

end
