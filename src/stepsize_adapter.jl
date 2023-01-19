abstract type AbstractStepsizeAdapter{T} end

Base.eltype(::AbstractStepsizeAdapter{T}) where {T} = T

function optimum(ssa::AbstractStepsizeAdapter; kwargs...)
    return ssa.stepsize_bar
end

# TODO(ear) move smoothed into optimum(; kwargs...)
function set_stepsize!(sampler, ssa::AbstractStepsizeAdapter; smoothed=false, kwargs...)
    sampler.stepsize .= smoothed ? optimum(ssa) : ssa.stepsize
end

struct StepsizeAdam{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    adam::Adam{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    initializer::Symbol
end

function StepsizeAdam(
    initial_stepsize::AbstractVector{T}; initializer=:sga, kwargs...
) where {T}
    chains = length(initial_stepsize)
    adam = Adam(chains, T; kwargs...)
    return StepsizeAdam(adam, initial_stepsize, zero(initial_stepsize), initializer)
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

function reset!(ssa::StepsizeAdam; kwargs...)
    reset!(ssa.adam; initial_stepsize=ssa.stepsize, kwargs...)
end

struct StepsizeDualAverage{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    da::DualAverage{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    initializer::Symbol
end

"""
    StepsizeDualAverage(initial_stepsize::Vector)

Construct a stepsize adapter using the dual averaging method by [Nesterov 2009](https://link.springer.com/article/10.1007/s10107-007-0149-x), as used in [Stan](https://mc-stan.org/docs/reference-manual/hmc-algorithm-parameters.html#ref-Nesterov:2009).  The length of `initial_stepsize::Vector` must be appropriate for the sampling algorithm for which this stepsize adapter will be used.
"""
function StepsizeDualAverage(
    initial_stepsize::AbstractVector{T}; initializer=:stan, kwargs...
) where {T<:AbstractFloat}
    chains = length(initial_stepsize)
    da = DualAverage(chains, T; kwargs...)
    return StepsizeDualAverage(da, initial_stepsize, zero(initial_stepsize), initializer)
end

function update!(ssa::StepsizeDualAverage, αs; kwargs...)
    ss, ssbar = update!(ssa.da, αs; kwargs...)
    ssa.stepsize .= ss
    ssa.stepsize_bar .= ssbar
end

function reset!(ssa::StepsizeDualAverage; kwargs...)
    reset!(ssa.da; initial_stepsize=ssa.stepsize_bar, kwargs...)
end

struct StepsizeConstant{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    initializer::Symbol
end

"""
    StepsizeConstant(initial_stepsize::Vector)

Construct a stepsize adapter for which the stepsize is fixed at it's initial value.
"""
function StepsizeConstant(
    initial_stepsize::AbstractVector{T}; initializer=:none, kwargs...
) where {T<:AbstractFloat}
    chains = length(initial_stepsize)
    return StepsizeConstant(initial_stepsize, initial_stepsize, initializer)
end

function update!(ssc::StepsizeConstant, args...; kwargs...) end

function reset!(ssc::StepsizeConstant, args...; kwargs...) end

function set_stepsize!(sampler, ssc::StepsizeConstant, args...; kwargs...)
    sampler.stepsize .= ssc.stepsize
end

struct StepsizeECA{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    initializer::Symbol
end

function StepsizeECA(initial_stepsize::AbstractVector{T}; initializer=:meads, kwargs...) where {T<:AbstractFloat}
    return StepsizeECA(initial_stepsize, initial_stepsize, initializer)
end

function update!(seca::StepsizeECA, ldg, positions, scale, idx; kwargs...)
    dims, chains = size(positions)
    gradients = similar(positions)
    for chain in axes(positions, 2)
        q = positions[:, chain]
        _, gradients[:, chain] = ldg(q; kwargs...)
    end
    scaled_gradients = gradients .* scale
    seca.stepsize[idx] =  min(1, 0.5 / sqrt(max_eigenvalue(scaled_gradients)))
    seca.stepsize_bar[idx] = seca.stepsize[idx]
end

function reset!(seca::StepsizeECA; kwargs...) end

function set_stepsize!(sampler, seca::StepsizeECA, idx; kwargs...)
    sampler.stepsize[idx] = seca.stepsize[idx]
end
