module MCBayes

abstract type AbstractAdaptationSchedule end
abstract type AbstractAdapter end

include("onlinemoments.jl")

export
    OnlineMoments,
    update!,
    reset!,
    metric

end
