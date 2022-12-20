abstract type AbstractTrajectorylengthAdapter end

struct TrajectorylengthAdam{T <: AbstractFloat} <: AbstractTrajectorylengthAdapter
    adam::Adam{T}
    initial_trajectorylength::Vector{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    learningrate::Symbol
end

function TrajectorylengthAdam(initial_stepsize::Vector{T};
                             initializer = :none, kwargs...) where {T}
    chains = length(initial_stepsize)
    adam = Adam(chains; kwargs...)
    return TrajectorylengthAdam(adam,
                                initial_trajectorylength,
                                zeros(T, chains),
                                zeros(T, chains),
                                initializer)
end

function optimum(tla::AbstractTrajectorylengthAdapter)
    return tla.trajectorylength_bar
end

function set_trajectorylength!(sampler, tla::AbstractTrajectorylengthAdapter)
    if :trajectorylength in fieldnames(typeof(sampler))
        sampler.trajectorylength .= optimum(tla)
    end
end

# TODO update, reset

struct TrajectorylengthConstant{T <: AbstractFloat} <: AbstractTrajectorylengthAdapter
    trajectorylength_bar::Vector{T}
    initializer::Symbol
end

function TrajectorylengthConstant(initial_trajectorylength::Vector{T};
                                  initializer = :none, kwargs...) where {T <: AbstractFloat}
    return TrajectorylengthConstant(initial_trajectorylength, initializer)
end

# TODO update, reset; args will need to match update and reset above

function update!(tlc::TrajectorylengthConstant; kwargs...)
end
