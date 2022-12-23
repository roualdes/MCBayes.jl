module MCBayes

using Random
using LinearAlgebra
using Statistics

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
include("tools.jl")
include("integrator.jl")
include("pspoint.jl")
include("trace.jl")
include("convergence.jl")

export Stan,
    Adam,
    OnlineMoments,
    StepsizeAdam,
    StepsizeDualAverage,
    StepsizeConstant,
    MetricOnlineMoments,
    MetricConstant,
    TrajectorylengthAdam,
    TrajectorylengthConstant,
    sample!,
    optimum,
    ess,
    ess_bulk,
    ess_tail,
    ess_quantile,
    ess_mean,
    ess_sq,
    ess_std,
    ess_f,
    rhat,
    rhat_basic,
    mcse_mean,
    mcse_std

function ldg(x)
    -x' * x / 2, -x
end

stan = Stan(10, 4)
draws, diagnostics = sample!(stan, ldg)

end
