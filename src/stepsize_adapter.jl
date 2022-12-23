abstract type AbstractStepsizeAdapter end

struct StepsizeAdam{T<:AbstractFloat} <: AbstractStepsizeAdapter
    adam::Adam{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    initializer::Symbol
end

function StepsizeAdam(initial_stepsize::Vector{T}; initializer=:sga, kwargs...) where {T}
    chains = length(initial_stepsize)
    adam = Adam(chains; kwargs...)
    return StepsizeAdam(adam, initial_stepsize, zeros(T, chains), initializer)
end

"""
Adam update on log-scale.
"""
# TODO(ear) should smoothing go into Adam, like for DualAveraging?
function update!(ssa::StepsizeAdam, αs; γ=-0.6, kwargs...)
    x = update!(ssa.adam, αs; kwargs...)
    @. ssa.stepsize *= exp(x)
    w = t^γ
    @. ssa.stepsize_bar = exp(
        w * log(ssa.stepsize) + (1 - w) * log(1e-10 + ssa.stepsize_bar)
    )
end

function optimum(ssa::AbstractStepsizeAdapter; kwargs...)
    return ssa.stepsize_bar
end

# TODO(ear) move smoothed into optimum(; kwargs...)
function set_stepsize!(sampler, ssa::AbstractStepsizeAdapter; smoothed=false, kwargs...)
    sampler.stepsize .= smoothed ? optimum(ssa) : ssa.stepsize
end

function reset!(ssa::StepsizeAdam; kwargs...)
    ss = ssa.stepsize_bar
    reset!(ssa.adam; initial_stepsize=ss, kwargs...)
end

struct StepsizeDualAverage{T<:AbstractFloat} <: AbstractStepsizeAdapter
    da::DualAverage{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    initializer::Symbol
end

function StepsizeDualAverage(
    initial_stepsize::Vector{T}; initializer=:stan, kwargs...
) where {T<:AbstractFloat}
    chains = length(initial_stepsize)
    da = DualAverage(chains; kwargs...)
    return StepsizeDualAverage(da, initial_stepsize, zeros(T, chains), initializer)
end

function update!(ssa::StepsizeDualAverage, αs; kwargs...)
    ss, ssbar = update!(ssa.da, αs; kwargs...)
    ssa.stepsize .= ss
    ssa.stepsize_bar .= ssbar
end

function reset!(ssa::StepsizeDualAverage; kwargs...)
    ss = ssa.stepsize_bar
    reset!(ssa.da; initial_stepsize=ss, kwargs...)
end

struct StepsizeConstant{T<:AbstractFloat} <: AbstractStepsizeAdapter
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    initializer::Symbol
end

function StepsizeConstant(
    initial_stepsize::Vector{T}; initializer=:none, kwargs...
) where {T<:AbstractFloat}
    chains = length(initial_stepsize)
    return StepsizeConstant(initial_stepsize, initial_stepsize, initializer)
end

function update!(ssc::StepsizeConstant, αs; kwargs...) end

function reset!(ssc::StepsizeConstant; kwargs...) end
