module MCBayes

using Random
using LinearAlgebra

include("windowedadaptation.jl")
include("dualaverage.jl")
include("adam.jl")
include("onlinemoments.jl")
include("initialize_draws.jl")
include("initialize_stepsize.jl")
include("stan.jl")
include("tools.jl")
include("integrator.jl")
include("pspoint.jl")

export
    Adam,
    OnlineMoments,

    optimum,
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
    set_trajectorylength!,

    hmc!,
    rand_momenta,
    hamiltonian,

    integrate!

end
