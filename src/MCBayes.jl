module MCBayes

using Random

include("windowedadaptation.jl")
include("dualaverage.jl")
include("onlinemoments.jl")
include("initialize_draws.jl")
include("initialize_stepsize.jl")
include("stan.jl")

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
