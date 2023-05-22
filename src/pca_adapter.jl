abstract type AbstractPCAAdapter{T} end

Base.eltype(::AbstractPCAAdapter{T}) where {T} = T

function set!(sampler, pca::AbstractPCAAdapter, args...; kwargs...)
    sampler.pca .= pca.pc
end

function optimum(pca::AbstractPCAAdapter, args...; kwargs...)
    return pca.pc
end

struct PCAOnline{T<:AbstractFloat} <: AbstractPCAAdapter{T}
    opca::OnlinePCA{T}
    pc::Vector{T}
end

function PCAOnline(T, dims; l = 2.0, kwargs...)
    opca = OnlinePCA(T, dims, l)
    return PCAOnline(opca, zeros(T, dims))
end

PCAOnline(dims; l = 2.0, kwargs...) = PCAOnline(Float64, dims; l = l, kwargs...)

function update!(pca::PCAOnline, x::AbstractMatrix, args...; kwargs...)
    update!(pca.opca, x; kwargs...)
end

function reset!(pca::PCAOnline, args...; kwargs...)
    reset!(pca.opca; kwargs...)
end

function optimum(pca::PCAOnline, args...; kwargs...)
    return pca.opca.pc
end

struct PCAConstant{T<:AbstractFloat} <: AbstractPCAAdapter{T}
    pc::Vector{T}
end

function PCAConstant(initial_pca::AbstractMatrix; kwargs...)
    return PCAConstant(initial_pca)
end

function update!(pca::PCAConstant, args...; kwargs...) end

function reset!(pca::PCAConstant, args...; kwargs...) end
