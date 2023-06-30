abstract type AbstractNoiseAdapter{T} end

Base.eltype(::AbstractNoiseAdapter{T}) where {T} = T

function optimum(na::AbstractNoiseAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? na.noise_bar : na.noise
end

function set!(sampler, na::AbstractNoiseAdapter, args...; kwargs...)
    if :noise in fieldnames(typeof(sampler))
        sampler.noise .= optimum(na)
    end
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

function NoiseConstant(initial_noise::AbstractVector; kwargs...)
    return NoiseConstant(initial_noise, initial_noise)
end

# function set!(sampler, nc::NoiseConstant, args...; kwargs...) end

function update!(nc::NoiseConstant, args...; kwargs...) end

function reset!(nc::NoiseConstant, args...; kwargs...) end

struct NoiseMALT{T<:AbstractFloat} <: AbstractNoiseAdapter{T}
    noise::Vector{T}
    noise_bar::Vector{T}
end

function NoiseMALT(initial_noise::AbstractVector; kwargs...)
    return NoiseMALT(initial_noise, initial_noise)
end

function update!(nmalt::NoiseMALT, damping, stepsize, args...; kwargs...)

    nmalt.noise .= exp.(-0.5 .* stepsize .* damping) .- 1e-2
    nmalt.noise_bar .= nmalt.noise
end

# TODO(ear) move reset! into AbstractNoiseAdapter
function reset!(nmalt::NoiseMALT, args...; kwargs...)
    nmalt.noise .= 0
    nmalt.noise_bar .= 0
end

function set!(sampler, nmalt::NoiseMALT, args...; kwargs...)
    sampler.noise .= nmalt.noise
end
