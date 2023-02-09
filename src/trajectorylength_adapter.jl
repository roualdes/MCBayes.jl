abstract type AbstractTrajectorylengthAdapter{T} end

Base.eltype(::AbstractTrajectorylengthAdapter{T}) where {T} = T

function optimum(tla::AbstractTrajectorylengthAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? tla.trajectorylength_bar : tla.trajectorylength
end

function set!(sampler, tla::AbstractTrajectorylengthAdapter, args...; kwargs...)
    if :trajectorylength in fieldnames(typeof(sampler))
        sampler.trajectorylength .= optimum(tla; kwargs...)
    end
end

struct TrajectorylengthChEES{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthChEES(
    initial_trajectorylength::AbstractVector{T}; maxleapfrogsteps = 1000, kwargs...) where {T}
    adam = Adam(1, T; kwargs...)
    return TrajectorylengthChEES(adam, initial_trajectorylength, zeros(T, 1), zeros(T, 1), maxleapfrogsteps)
end

function update!(tlc::TrajectorylengthChEES, m, αs, draws, stepsize, args...; γ=-0.6, kwargs...)
   ghats = trajectorylength_gradient(m, )
end


struct TrajectorylengthConstant{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
end

function TrajectorylengthConstant(
    initial_trajectorylength::AbstractVector{T}; kwargs...) where {T<:AbstractFloat}
    return TrajectorylengthConstant(initial_trajectorylength, initial_trajectorylength)
end

function update!(tlc::TrajectorylengthConstant, args...; kwargs...) end

function reset!(tlc::TrajectorylengthConstant, args...; kwargs...) end
