abstract type AbstractNoiseAdapter{T} end

Base.eltype(::AbstractNoiseAdapter{T}) where {T} = T

function optimum(na::AbstractNoiseAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? na.noise_bar : na.noise
end

function set!(sampler, na::AbstractNoiseAdapter, args...; kwargs...)
    sampler.noise .= optimum(na)
end

struct NoiseECA{T<:AbstractFloat} <: AbstractNoiseAdapter{T}
    noise::Vector{T}
    noise_bar::Vector{T}
end

function NoiseECA(initial_noise::AbstractVector{T}; kwargs...) where {T}
    return NoiseECA(initial_noise, zero(initial_noise))
end

function update!(neca::NoiseECA, damping, args...; kwargs...)
    neca.noise .= sqrt(1 .- exp.(-2 .* damping))
    neca.noise_bar .= neca.noise
end

function update!(neca::NoiseECA, damping, idx, args...; kwargs...)
    neca.noise[idx] = sqrt(1 - exp(-2 * damping[idx]))
    neca.noise_bar[idx] = neca.noise[idx]
end

function reset!(neca::NoiseECA, args...; kwargs...)
    neca.noise .= 0
    neca.noise_bar .= 0
end

function set!(sampler, neca::NoiseECA, idx, args...; kwargs...)
    sampler.noise[idx] = neca.noise[idx]
end

struct NoiseConstant{T<:AbstractFloat} <: AbstractNoiseAdapter{T}
    noise::Vector{T}
    noise_bar::Vector{T}
end

function NoiseConstant(initial_drift::AbstractVector; kwargs...)
    return NoiseConstant(initial_drift, initial_drift)
end

function set!(sampler, nc::NoiseConstant, args...; kwargs...) end

function update!(nc::NoiseConstant, args...; kwargs...) end

function reset!(nc::NoiseConstant, args...; kwargs...) end
