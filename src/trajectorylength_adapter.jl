abstract type AbstractTrajectorylengthAdapter{T} end

Base.eltype(::AbstractTrajectorylengthAdapter{T}) where {T} = T

struct TrajectorylengthAdam{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    initial_trajectorylength::Vector{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    learningrate::Symbol
end

function TrajectorylengthAdam(
    initial_stepsize::AbstractVector{T}; initializer=:none, kwargs...
) where {T}
    chains = length(initial_stepsize)
    adam = Adam(chains, T; kwargs...)
    return TrajectorylengthAdam(
        adam, initial_trajectorylength, zeros(T, chains), zeros(T, chains), initializer
    )
end

function optimum(tla::AbstractTrajectorylengthAdapter)
    return tla.trajectorylength_bar
end

function set_trajectorylength!(sampler, tla::AbstractTrajectorylengthAdapter; kwargs...)
    if :trajectorylength in fieldnames(typeof(sampler))
        sampler.trajectorylength .= optimum(tla)
    end
end

# TODO update, reset

struct TrajectorylengthConstant{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    trajectorylength_bar::Vector{T}
    initializer::Symbol
end

function TrajectorylengthConstant(
    initial_trajectorylength::AbstractVector{T}; initializer=:none, kwargs...
) where {T<:AbstractFloat}
    return TrajectorylengthConstant(initial_trajectorylength, initializer)
end

# TODO update, reset; args will need to match update and reset above

function update!(tlc::TrajectorylengthConstant; kwargs...) end
