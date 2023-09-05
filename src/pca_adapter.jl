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
    pc_bar::Matrix{T}
    alpha::T
end

function PCAOnline(initial_pca::AbstractMatrix{T}; l = 2, pca_smoothing_factor = 1 - 8/9, kwargs...) where {T}
    dims, pcas = size(initial_pca)
    opca = OnlinePCA(T, dims, pcas, convert(T, l)::T)
    return PCAOnline(opca, copy(initial_pca), copy(initial_pca), convert(T, pca_smoothing_factor)::T)
end

# TODO not convinced this should be smoothed
# updating the averaging is smoothing, why would we smooth a mean?
function update!(
    pca::PCAOnline{T}, x::AbstractMatrix, args...; kwargs...
) where {T}
    update!(pca.opca, x; kwargs...)
    pca.pc .= pca.opca.pc
    pca.pc_bar .= pca.alpha .* pca.pc .+ (1 - pca.alpha) .* pca.pc_bar
end

function update!(
    pca::PCAOnline{T}, x::AbstractMatrix, location::AbstractMatrix, scale::AbstractMatrix, args...; kwargs...
        ) where {T}
    update!(pca.opca, x, location, scale; kwargs...)
    pca.pc .= pca.opca.pc
    pca.pc .= pca.opca.pc
    pca.pc_bar .= pca.alpha .* pca.pc .+ (1 - pca.alpha) .* pca.pc_bar
end

function reset!(pca::PCAOnline, args...; kwargs...)
    reset!(pca.opca; kwargs...)
    pca.pc_bar .= 0
end

struct PCAConstant{T<:AbstractFloat} <: AbstractPCAAdapter{T}
    pc::Matrix{T}
end

function update!(pca::PCAConstant, args...; kwargs...) end

function reset!(pca::PCAConstant, args...; kwargs...) end
