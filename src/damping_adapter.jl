abstract type AbstractDampingAdapter{T} end

Base.eltype(::AbstractDampingAdapter{T}) where {T} = T

function optimum(deca::AbstractDampingAdapter; kwargs...)
    return deca.damping_bar
end

# TODO(ear) move smoothed into optimum(; kwargs...)
function set_stepsize!(sampler, deca::AbstractDampingAdapter; smoothed=false, kwargs...)
    sampler.damping .= smoothed ? optimum(deca) : deca.damping
end

struct DampingECA{T<:AbstractFloat} <: AbstractDampingAdapter{T}
    damping::Vector{T}
    damping_bar::Vector{T}
end

function DampingECA(initial_damping::AbstractVector{T}; kwargs...) where {T}
    return DampingECA(initial_damping, zero(initial_damping))
end

function update!(deca::DampingECA, stepsize; kwargs...)
    deca.damping .= 1 ./ stepsize
    deca.damping_bar .= deca.damping
end

function reset!(deca::DampingECA; kwargs...)
    deca.damping_bar .= 0
end

struct DampingConstant{T<:AbstractFloat} <: AbstractDampingAdapter{T}
    damping::Vector{T}
    damping_bar::Vector{T}
end

function DampingConstant(initial_drift::AbstractVector; kwargs...)
    return DampingConstant(initial_drift, initial_drift)
end

function update!(dc::DampingConstant, stepsize; kwargs...) end

function reset!(dc::DampingConstant; kwargs...) end
