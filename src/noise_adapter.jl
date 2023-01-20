abstract type AbstractNoiseAdapter{T} end

Base.eltype(::AbstractNoiseAdapter{T}) where {T} = T

function optimum(neca::AbstractNoiseAdapter; kwargs...)
    return neca.noise_bar
end

# TODO(ear) move smoothed into optimum(; kwargs...)
function set!(sampler, neca::AbstractNoiseAdapter; smoothed=false, kwargs...)
    sampler.noise .= smoothed ? optimum(neca) : neca.noise
end

struct NoiseECA{T<:AbstractFloat} <: AbstractNoiseAdapter{T}
    noise::Vector{T}
    noise_bar::Vector{T}
end

function NoiseECA(initial_noise::AbstractVector{T}; kwargs...) where {T}
    return NoiseECA(initial_noise, zero(initial_noise))
end

function update!(neca::NoiseECA, damping; kwargs...)
    neca.noise .= sqrt(1 .- exp.(-2 .* damping))
    neca.noise_bar .= neca.noise
end

function update!(neca::NoiseECA, damping, idx; kwargs...)
    neca.noise[idx] = sqrt(1 - exp(-2 * damping[idx]))
    neca.noise_bar[idx] = neca.noise[idx]
end

function reset!(neca::NoiseECA; kwargs...)
    neca.noise_bar .= 0
end

function set!(sampler, neca::NoiseECA, idx; kwargs...)
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

function reset!(nc::NoiseConstant; kwargs...) end
