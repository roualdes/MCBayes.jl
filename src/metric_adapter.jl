abstract type AbstractMetricAdapter{T} end

Base.eltype(::AbstractMetricAdapter{T}) where {T} = T

function set!(sampler, ma::AbstractMetricAdapter, args...; kwargs...)
    if ma.om.n[1] > 3
        sampler.metric .= ma.metric
    else
        sampler.metric .= 1
    end
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

function set!(sampler, meca::MetricECA, idx, args...; kwargs...)
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
    mfd::MetricFisherDivergence,
    x::AbstractMatrix,
    ldg,
    args...;
    metric_smooth=true,
    kwargs...,
    )
    grads = similar(x)
    for c in axes(grads, 2)
        _, grads[:, c] = ldg(x[:, c]; kwargs...)
    end
    update!(mfd.om, x; kwargs...)
    update!(mfd.og, grads; kwargs...)
    # TODO potential issues: divide by zero, infinity, nan
    mfd.metric .= sqrt.(mfd.om.v ./ mfd.og.v)
    mfd.metric .= clamp.(mfd.metric, 1e-10, 1e10)
end

function reset!(mfd::MetricFisherDivergence, args...; kwargs...)
    reset!(mfd.om; kwargs...)
    reset!(mfd.og; kwargs...)
end
