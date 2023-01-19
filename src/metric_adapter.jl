abstract type AbstractMetricAdapter{T} end

Base.eltype(::AbstractMetricAdapter{T}) where {T} = T

function set_metric!(sampler, ma::AbstractMetricAdapter; kwargs...)
    sampler.metric .= optimum(ma; kwargs...)
end

struct MetricOnlineMoments{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    om::OnlineMoments{T}
    metric::Matrix{T}
end

function MetricOnlineMoments(initial_metric::AbstractMatrix{T}; kwargs...) where {T}
    dims, metrics = size(initial_metric)
    om = OnlineMoments(T, dims, metrics)
    return MetricOnlineMoments(om, initial_metric)
end

function update!(mom::MetricOnlineMoments, x::AbstractMatrix; kwargs...)
    update!(mom.om, x; kwargs...)
end

function optimum(mom::MetricOnlineMoments; kwargs...)
    return optimum(mom.om; kwargs...)
end

function reset!(mom::MetricOnlineMoments; kwargs...)
    reset!(mom.om; kwargs...)
end

struct MetricConstant{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    metric::Matrix{T}
end

function MetricConstant(initial_metric::AbstractMatrix; kwargs...)
    return MetricConstant(initial_metric)
end

function update!(mc::MetricConstant, args...; kwargs...) end

function optimum(mc::MetricConstant; kwargs...)
    return mc.metric
end

function reset!(mc::MetricConstant; kwargs...) end

function set_metric!(sampler, mc::MetricConstant, args...; kwargs...)
    sampler.metric .= mc.metric
end

struct MetricECA{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    metric::Matrix{T}
end

function MetricECA(initial_metric::AbstractMatrix; kwargs...)
    return MetricECA(initial_metric)
end

function update!(meca::MetricECA, sigma, idx; kwargs...)
    meca.metric[:, idx] .= sigma
end

function set_metric!(sampler, meca::MetricECA, idx; kwargs...)
    sampler.metric[:, idx] .= meca.metric[:, idx]
end
