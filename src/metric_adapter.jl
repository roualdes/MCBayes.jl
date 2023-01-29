abstract type AbstractMetricAdapter{T} end

Base.eltype(::AbstractMetricAdapter{T}) where {T} = T

function set!(sampler, ma::AbstractMetricAdapter; kwargs...)
    sampler.metric .= optimum(ma; kwargs...)
end

function optimum(ma::AbstractMetricAdapter; kwargs...)
    return ma.metric
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

function update!(mom::MetricOnlineMoments, x::AbstractMatrix, args...; kwargs...)
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

function reset!(mc::MetricConstant; kwargs...) end

function set!(sampler, mc::MetricConstant, args...; kwargs...)
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

function set!(sampler, meca::MetricECA, idx; kwargs...)
    sampler.metric[:, idx] .= meca.metric[:, idx]
end

struct MetricFisherDivergence{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    om::OnlineMoments{T}
    og::OnlineMoments{T}
    metric::Matrix{T}
end

function MetricFisherDivergence(initial_metric::AbstractMatrix{T}; kwargs...) where {T}
    dims, metrics = size(initial_metric)
    om = OnlineMoments(T, dims, metrics)
    og = OnlineMoments(T, dims, metrics)
    return MetricFisherDivergence(om, og, initial_metric)
end

function update!(mfd::MetricFisherDivergence, x::AbstractMatrix, g::AbstractMatrix; kwargs...)
    update!(mfd.om, x; kwargs...)
    update!(mfd.og, g; kwargs...)
end

function optimum(mfd::MetricFisherDivergence; kwargs...)
    T = eltype(mfd.om.v)
    if mfd.om.n[1] > 1
        return sqrt.(mfd.om.v ./ mfd.og.v)
    else
        return ones(T, size(mfd.om.v))
    end
end

function reset!(mfd::MetricFisherDivergence; kwargs...)
    reset!(mfd.om; kwargs...)
    reset!(mfd.og; kwargs...)
end
