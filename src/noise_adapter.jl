abstract type AbstractNoiseAdapter{T} end

Base.eltype(::AbstractNoiseAdapter{T}) where {T} = T

function optimum(neca::AbstractNoiseAdapter; kwargs...)
    return neca.noise_bar
end

# TODO(ear) move smoothed into optimum(; kwargs...)
function set_stepsize!(sampler, neca::AbstractNoiseAdapter; smoothed=false, kwargs...)
    sampler.noise .= smoothed ? optimum(neca) : neca.noise
end

struct NoiseECA{T<:AbstractFloat} <: AbstractNoiseAdapter{T}
    noise::Vector{T}
    noise_bar::Vector{T}
end

function NoiseECA(initial_noise::AbstractVector{T}; kwargs...) where {T}
    return NoiseECA(initial_noise, zero(initial_noise))
end

function update!(neca::NoiseECA, damping, stepsize; kwargs...)
    neca.noise .= 1 .- exp.(-2 .* damping .* stepsize)
    neca.noise_bar .= neca.noise
end

function reset!(neca::NoiseECA; kwargs...)
    neca.noise_bar .= 0
end

struct NoiseConstant{T<:AbstractFloat} <: AbstractNoiseAdapter{T}
    noise::Vector{T}
    noise_bar::Vector{T}
end

function NoiseConstant(initial_drift::AbstractVector; kwargs...)
    return NoiseConstant(initial_drift, initial_drift)
end

function update!(nc::NoiseConstant, damping, stepsize; kwargs...) end

function reset!(nc::NoiseConstant; kwargs...) end
