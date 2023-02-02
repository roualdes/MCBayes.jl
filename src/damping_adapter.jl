abstract type AbstractDampingAdapter{T} end

Base.eltype(::AbstractDampingAdapter{T}) where {T} = T

function optimum(da::AbstractDampingAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? da.damping_bar : da.damping
end

function set!(sampler, da::AbstractDampingAdapter, args...; kwargs...)
    sampler.damping .= optimum(da)
end

struct DampingECA{T<:AbstractFloat} <: AbstractDampingAdapter{T}
    damping::Vector{T}
    damping_bar::Vector{T}
end

function DampingECA(initial_damping::AbstractVector{T}; kwargs...) where {T}
    return DampingECA(initial_damping, zero(initial_damping))
end

function update!(deca::DampingECA, m, zpositions, stepsize, idx, args...; kwargs...)
    deca.damping[idx] = max(1 / m, stepsize[idx] / sqrt(max_eigenvalue(zpositions)))
    deca.damping_bar[idx] = deca.damping[idx]
end

function reset!(deca::DampingECA, args...; kwargs...)
    deca.damping .= 0
    deca.damping_bar .= 0
end

function set!(sampler, deca::DampingECA, idx, args...; kwargs...)
    sampler.damping[idx] = deca.damping[idx]
end

struct DampingConstant{T<:AbstractFloat} <: AbstractDampingAdapter{T}
    damping::Vector{T}
    damping_bar::Vector{T}
end

function DampingConstant(initial_drift::AbstractVector; kwargs...)
    return DampingConstant(initial_drift, initial_drift)
end

function update!(dc::DampingConstant, args...; kwargs...) end

function reset!(dc::DampingConstant, args...; kwargs...) end
