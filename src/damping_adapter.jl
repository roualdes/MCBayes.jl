abstract type AbstractDampingAdapter{T} end

Base.eltype(::AbstractDampingAdapter{T}) where {T} = T

function optimum(da::AbstractDampingAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? da.damping_bar : da.damping
end

function set!(sampler, da::AbstractDampingAdapter, args...; kwargs...)
    if :damping in fieldnames(typeof(sampler))
        sampler.damping .= optimum(da)
    end
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

function DampingConstant(initial_damping::AbstractVector; kwargs...)
    return DampingConstant(initial_damping, initial_damping)
end

function update!(dc::DampingConstant, args...; kwargs...) end

function reset!(dc::DampingConstant, args...; kwargs...) end


struct DampingMALT{T<:AbstractFloat} <: AbstractDampingAdapter{T}
    damping::Vector{T}
    damping_bar::Vector{T}
end

function DampingMALT(initial_damping::AbstractVector, args...; kwargs...)
    return DampingMALT(initial_damping, initial_damping)
end

function update!(dmalt::DampingMALT, m, gamma, args...; damping_coefficient = 1, kwargs...)
    dmalt.damping .= damping_coefficient ./ (1e-10 .+ sqrt(gamma))
    dmalt.damping_bar .= dmalt.damping
end

function set!(sampler, dmalt::DampingMALT, args...; kwargs...)
    sampler.damping .= dmalt.damping
end

# TODO(ear) move reset into AbstractDampingAdapter
function reset!(dmalt::DampingMALT, args...; kwargs...)
    dmalt.damping .= 0
    dmalt.damping_bar .= 0
end
