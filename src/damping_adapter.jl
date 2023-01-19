abstract type AbstractDampingAdapter{T} end

Base.eltype(::AbstractDampingAdapter{T}) where {T} = T

function optimum(deca::AbstractDampingAdapter; kwargs...)
    return deca.damping_bar
end

# TODO(ear) move smoothed into optimum(; kwargs...)
function set_damping!(sampler, deca::AbstractDampingAdapter; smoothed=false, kwargs...)
    sampler.damping .= smoothed ? optimum(deca) : deca.damping
end

struct DampingECA{T<:AbstractFloat} <: AbstractDampingAdapter{T}
    damping::Vector{T}
    damping_bar::Vector{T}
end

function DampingECA(initial_damping::AbstractVector{T}; kwargs...) where {T}
    return DampingECA(initial_damping, zero(initial_damping))
end

function update!(deca::DampingECA, m, zpositions, stepsize, idx; kwargs...)
    deca.damping[idx] = max(1 / m, stepsize[idx] / sqrt(max_eigenvalue(zpositions)))
    deca.damping_bar[idx] = deca.damping[idx]
end

function reset!(deca::DampingECA; kwargs...)
    deca.damping_bar .= 0
end

function set_damping!(sampler, deca::DampingECA, idx; kwargs...)
    sampler.damping[idx] = deca.damping[idx]
end

struct DampingConstant{T<:AbstractFloat} <: AbstractDampingAdapter{T}
    damping::Vector{T}
    damping_bar::Vector{T}
end

function DampingConstant(initial_drift::AbstractVector; kwargs...)
    return DampingConstant(initial_drift, initial_drift)
end

function update!(dc::DampingConstant, m, zpositions, stepsize, idx; kwargs...) end

function reset!(dc::DampingConstant; kwargs...) end
