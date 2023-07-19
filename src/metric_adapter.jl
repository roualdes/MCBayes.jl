abstract type AbstractMetricAdapter{T} end

Base.eltype(::AbstractMetricAdapter{T}) where {T} = T

function set!(sampler, ma::AbstractMetricAdapter{T}, args...; kwargs...) where {T}
    sampler.metric .= ma.metric
end

# TODO just add method to
# Base.mean or whatever, maybe Statistics.mean
# via import Statistics.mean
function metric_mean(ma::AbstractMetricAdapter, args...; kwargs...)
    return ma.om.m
end

struct MetricOnlineMoments{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    om::OnlineMoments{T}
    metric::Matrix{T}
end

function MetricOnlineMoments(
    initial_metric::AbstractMatrix{T}, args...; kwargs...
) where {T}
    dims, metrics = size(initial_metric)
    om = OnlineMoments(T, dims, metrics)
    return MetricOnlineMoments(om, initial_metric)
end

function update!(
    mom::MetricOnlineMoments{T},
    x::AbstractMatrix,
    args...;
    metric_regularize=true,
    kwargs...,
) where {T}
    update!(mom.om, x; kwargs...)
    if metric_regularize
        w = reshape(convert.(T, mom.om.n ./ (mom.om.n .+ 5)), 1, :)
        mom.metric .= w .* mom.om.v .+ (1 .- w) .* convert(T, 1e-3)::T
    else
        mom.metric .= mom.om.v
    end
end

function reset!(mom::MetricOnlineMoments, args...; kwargs...)
    reset!(mom.om; kwargs...)
end

struct MetricConstant{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    metric::Matrix{T}
end

function MetricConstant(initial_metric::AbstractMatrix, args...; kwargs...)
    return MetricConstant(initial_metric)
end

function metric_mean(mc::MetricConstant, args...; kwargs...)
    T = eltype(mc.metric)
    dims = size(mc.metric, 1)
    return zeros(T, dims)
end

function update!(mc::MetricConstant, args...; kwargs...) end

function reset!(mc::MetricConstant, args...; kwargs...) end

struct MetricECA{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    metric::Matrix{T}
end

function MetricECA(initial_metric::AbstractMatrix, args...; kwargs...)
    return MetricECA(initial_metric)
end

function update!(meca::MetricECA, sigma, idx, args...; kwargs...)
    meca.metric[:, idx] .= sigma
end

function set!(sampler, meca::MetricECA{T}, idx, args...; kwargs...) where {T}
    sampler.metric[:, idx] .= meca.metric[:, idx]
end

struct MetricFisherDivergence{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    om::OnlineMoments{T}
    og::OnlineMoments{T}
    metric::Matrix{T}
end

function MetricFisherDivergence(
    initial_metric::AbstractMatrix{T}, args...; kwargs...
) where {T}
    dims, metrics = size(initial_metric)
    om = OnlineMoments(T, dims, metrics)
    og = OnlineMoments(T, dims, metrics)
    return MetricFisherDivergence(om, og, initial_metric)
end

function update!(
    mfd::MetricFisherDivergence{T},
    x::AbstractMatrix,
    ldg!,
    args...;
    metric_regularize=true,
    kwargs...,
    ) where {T}
    dims, chains = size(x)
    grads = [zeros(dims) for chain in 1:chains]
    for chain in axes(x, 2)
        ldg!(x[:, chain], grads[chain]; kwargs...)
    end
    update!(mfd.om, x; kwargs...)
    gradients = reduce(hcat, grads)
    update!(mfd.og, gradients; kwargs...)
    V = sqrt.(mfd.om.v ./ (mfd.og.v .+ 1e-10))
    if metric_regularize
        w = reshape(convert.(T, mfd.om.n ./ (mfd.om.n .+ 5)), 1, :)
        mfd.metric .= w .* V .+ (1 .- w) .* convert(T, 1e-3)::T
    else
        mfd.metric .= V
    end
end

function reset!(mfd::MetricFisherDivergence, args...; kwargs...)
    reset!(mfd.om; kwargs...)
    reset!(mfd.og; kwargs...)
end
