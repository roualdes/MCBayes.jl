abstract type AbstractMetricAdapter{T} end

Base.eltype(::AbstractMetricAdapter{T}) where {T} = T

function set!(sampler, ma::AbstractMetricAdapter, args...; kwargs...)
    sampler.metric .= ma.metric
end

struct MetricOnlineMoments{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    om::OnlineMoments{T}
    metric::Matrix{T}
    alpha::T
end

function MetricOnlineMoments(initial_metric::AbstractMatrix{T}, args...;
                             metric_smoothing_factor = 1 - 8/9, kwargs...) where {T}
    dims, metrics = size(initial_metric)
    om = OnlineMoments(T, dims, metrics)
    return MetricOnlineMoments(om, initial_metric, metric_smoothing_factor)
end

function update!(mom::MetricOnlineMoments{T}, x::AbstractMatrix, args...;
                 metric_regularize = true, metric_smooth = true, kwargs...) where {T}
    update!(mom.om, x; kwargs...)
    if metric_regularize
        w = reshape(convert.(T, mom.om.n ./ (mom.om.n .+ 5)), 1, :)
        mom.metric .= w .* mom.om.v .+ (1 .- w) .* convert(T, 1e-3)::T
    end
    w = mom.alpha + (1 - metric_smooth) * (1 - mom.alpha)
    mom.metric .= w .* mom.om.v .+ (1 - w) .* mom.metric
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
    alpha::T
end

function MetricFisherDivergence(initial_metric::AbstractMatrix{T}, args...;
                                metric_smoothing_factor = 1 - 8/9, kwargs...) where {T}
    dims, metrics = size(initial_metric)
    om = OnlineMoments(T, dims, metrics)
    og = OnlineMoments(T, dims, metrics)
    return MetricFisherDivergence(om, og, initial_metric)
end

function update!(mfd::MetricFisherDivergence, x::AbstractMatrix, ldg, args...;
                 metric_smooth = true, kwargs...)
    grads = similar(x)
    for c in axes(grads, 2)
        _, grads[:, c] = ldg(x[:, c]; kwargs...)
    end
    update!(mfd.om, x; kwargs...)
    update!(mfd.og, grads; kwargs...)
    w = mfd.alpha + (1 - metric_smooth) * (1 - mfd.alpha)
    mfd.metric .= w .* sqrt.(mfd.om.v ./ mfd.og.v) .+ (1 - w) .* mfd.metric
end

function reset!(mfd::MetricFisherDivergence, args...; kwargs...)
    reset!(mfd.om; kwargs...)
    reset!(mfd.og; kwargs...)
end
