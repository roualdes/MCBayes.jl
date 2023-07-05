abstract type AbstractPCAAdapter{T} end

Base.eltype(::AbstractPCAAdapter{T}) where {T} = T

function set!(sampler, pca::AbstractPCAAdapter, args...; kwargs...)
    sampler.pca .= pca.pc
end

struct PCAOnline{T<:AbstractFloat} <: AbstractPCAAdapter{T}
    opca::OnlinePCA{T}
    pc::Vector{T}
    alpha::T
end

function PCAOnline(T, dims; pca_smoothing_factor=1 - 3 / 4, l = 2, kwargs...)
    opca = OnlinePCA(T, dims, convert(T, l)::T)
    return PCAOnline(opca, zeros(T, dims), convert(T, pca_smoothing_factor)::T)
end

PCAOnline(dims; kwargs...) = PCAOnline(Float64, dims; kwargs...)

# TODO not convinced this should be smoothed
# updating the averaging is smoothing, why would we smooth a mean?
function update!(
    pca::PCAOnline{T}, x::AbstractMatrix, args...; pca_smooth=true, kwargs...
) where {T}
    update!(pca.opca, x; kwargs...)
    w = pca.alpha + (1 - pca_smooth) * (1 - pca.alpha)
    pca.pc .= w .* pca.opca.pc .+ (1 - w) .* pca.pc
end

function reset!(pca::PCAOnline, args...; kwargs...)
    reset!(pca.opca; kwargs...)
end

struct PCAConstant{T<:AbstractFloat} <: AbstractPCAAdapter{T}
    pc::Vector{T}
end

function PCAConstant(initial_pca::AbstractMatrix, args...; kwargs...)
    return PCAConstant(initial_pca)
end

function update!(pca::PCAConstant, args...; kwargs...) end

function reset!(pca::PCAConstant, args...; kwargs...) end
