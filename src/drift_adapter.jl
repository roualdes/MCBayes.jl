abstract type AbstractDriftAdapter{T} end

Base.eltype(::AbstractDriftAdapter{T}) where {T} = T

function optimum(da::AbstractDriftAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? da.drift_bar : da.drift
end

function set!(sampler, da::AbstractDriftAdapter, args...; kwargs...)
    sampler.drift .= optimum(da)
end

struct DriftECA{T<:AbstractFloat} <: AbstractDriftAdapter{T}
    drift::Vector{T}
    drift_bar::Vector{T}
end

function DriftECA(initial_drift::AbstractVector; kwargs...)
    return DriftECA(initial_drift, zero(initial_drift))
end

function update!(deca::DriftECA, noise, args...; kwargs...)
    deca.drift .= noise .^ 2 ./ 2
    deca.drift_bar .= deca.drift
end

function update!(deca::DriftECA, noise, idx, args...; kwargs...)
    deca.drift[idx] = noise[idx]^2 / 2
    deca.drift_bar[idx] = deca.drift[idx]
end

function reset!(deca::DriftECA, args...; kwargs...)
    deca.drift_bar .= 0
end

function set!(sampler, deca::DriftECA, idx, args...; kwargs...)
    sampler.drift[idx] = deca.drift[idx]
end

struct DriftConstant{T<:AbstractFloat} <: AbstractDriftAdapter{T}
    drift::Vector{T}
    drift_bar::Vector{T}
end

function DriftConstant(initial_drift::AbstractVector; kwargs...)
    return DriftConstant(initial_drift, initial_drift)
end

function update!(dc::DriftConstant, args...; kwargs...) end

function reset!(dc::DriftConstant, args...; kwargs...) end
