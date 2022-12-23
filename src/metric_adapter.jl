abstract type AbstractMetricAdapter{T} end

Base.eltype(::AbstractMetricAdapter{T}) where {T} = T

function set_metric!(sampler, ma::AbstractMetricAdapter; kwargs...)
    return sampler.metric .= optimum(ma; kwargs...)
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
    return update!(mom.om, x; kwargs...)
end

function optimum(mom::MetricOnlineMoments; kwargs...)
    return optimum(mom.om; kwargs...)
end

function reset!(mom::MetricOnlineMoments; kwargs...)
    return reset!(mom.om; kwargs...)
end

struct MetricConstant{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    metric::Matrix{T}
end

function MetricConstant(initial_metric::AbstractMatrix{T}; kwargs...) where {T}
    return MetricConstant(initial_metric)
end

function update!(mc::MetricConstant, x::AbstractMatrix; kwargs...) end

function optimum(mc::MetricConstant; kwargs...)
    return mc.metric
end

function reset!(mc::MetricConstant; kwargs...) end
