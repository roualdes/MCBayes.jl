abstract type AbstractPCAAdapter{T} end

Base.eltype(::AbstractPCAAdapter{T}) where {T} = T

function set!(sampler, pca::AbstractPCAAdapter, args...; kwargs...)
    sampler.pca .= pca.pc
end

function lambda_max(pca::AbstractPCAAdapter{T}) where {T}
    chains = size(pca.pc, 2)
    lambda_max = Vector{T}(undef, chains)
    @views for chain in 1:chains
        lambda_max[chain] = norm(pca.pc[:, chain])
    end
    return lambda_max
end

struct PCAOnline{T<:AbstractFloat} <: AbstractPCAAdapter{T}
    opca::OnlinePCA{T}
    pc::Matrix{T}
end

function PCAOnline(initial_pca::AbstractMatrix{T}; l = 2, kwargs...) where {T}
    dims, pcas = size(initial_pca)
    opca = OnlinePCA(T, dims, pcas, convert(T, l)::T)
    return PCAOnline(opca, copy(initial_pca))
end

PCAOnline(dims; kwargs...) = PCAOnline(Float64, dims; kwargs...)

# TODO not convinced this should be smoothed
# updating the averaging is smoothing, why would we smooth a mean?
function update!(
    pca::PCAOnline{T}, x::AbstractMatrix, args...; kwargs...
) where {T}
    update!(pca.opca, x; kwargs...)
    pca.pc .= pca.opca.pc
end

function update!(
    pca::PCAOnline{T}, x::AbstractMatrix, location::AbstractMatrix, scale::AbstractMatrix, args...; kwargs...
        ) where {T}
    update!(pca.opca, x, location, scale; kwargs...)
    pca.pc .= pca.opca.pc
end

function reset!(pca::PCAOnline, args...; kwargs...)
    reset!(pca.opca; kwargs...)
end

struct PCAConstant{T<:AbstractFloat} <: AbstractPCAAdapter{T}
    pc::Vector{T}
end

function PCAConstant(initial_pca::AbstractMatrix, args...; kwargs...)
    return PCAConstant(copy(initial_pca))
end

function update!(pca::PCAConstant, args...; kwargs...) end

function reset!(pca::PCAConstant, args...; kwargs...) end
