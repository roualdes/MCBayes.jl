abstract type AbstractDriftAdapter{T} end

Base.eltype(::AbstractDriftAdapter{T}) where {T} = T

function optimum(deca::AbstractDriftAdapter; kwargs...)
    return deca.drift_bar
end

# TODO(ear) move smoothed into optimum(; kwargs...)
function set_stepsize!(sampler, deca::AbstractDriftAdapter; smoothed=false, kwargs...)
    sampler.drift .= smoothed ? optimum(deca) : deca.drift
end

struct DriftECA{T<:AbstractFloat} <: AbstractDriftAdapter{T}
    drift::Vector{T}
    drift_bar::Vector{T}
end

function DriftECA(initial_drift::AbstractVector; kwargs...)
    return DriftECA(initial_drift, zero(initial_drift))
end

function update!(deca::DriftECA, noise; kwargs...)
    deca.drift .= noise ./ 2
    deca.drift_bar .= deca.drift
end

function reset!(deca::DriftECA; kwargs...)
    deca.drift_bar .= 0
end

struct DriftConstant{T<:AbstractFloat} <: AbstractDriftAdapter{T}
    drift::Vector{T}
    drift_bar::Vector{T}
end

function DriftConstant(initial_drift::AbstractVector; kwargs...)
    return DriftConstant(initial_drift, initial_drift)
end

function update!(dc::DriftConstant, noise; kwargs...) end

function reset!(dc::DriftConstant; kwargs...) end