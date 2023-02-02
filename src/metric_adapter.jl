abstract type AbstractMetricAdapter{T} end

Base.eltype(::AbstractMetricAdapter{T}) where {T} = T

function set!(sampler, ma::AbstractMetricAdapter, args...; kwargs...)
    sampler.metric .= optimum(ma; kwargs...)
end

function optimum(ma::AbstractMetricAdapter, args...; kwargs...)
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

function optimum(mom::MetricOnlineMoments, args...; regularized=true, kwargs...)
    T = eltype(mom.om.v)
    if mom.om.n[1] > 1
        v = if regularized
            w = reshape(convert.(T, mom.om.n ./ (mom.om.n .+ 5)), 1, :)
            @. w * mom.om.v + (1 - w) * convert(T, 1e-3)
        else
            mom.om.v
        end
        return v
    else
        return ones(T, size(mom.om.v))
    end
end

function reset!(mom::MetricOnlineMoments, args...; kwargs...)
    reset!(mom.om; kwargs...)
end

struct MetricConstant{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    metric::Matrix{T}
end

function MetricConstant(initial_metric::AbstractMatrix; kwargs...)
    return MetricConstant(initial_metric)
end

function update!(mc::MetricConstant, args...; kwargs...) end

function reset!(mc::MetricConstant, args...; kwargs...) end

struct MetricECA{T<:AbstractFloat} <: AbstractMetricAdapter{T}
    metric::Matrix{T}
end

function MetricECA(initial_metric::AbstractMatrix; kwargs...)
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

function MetricFisherDivergence(initial_metric::AbstractMatrix{T}; kwargs...) where {T}
    dims, metrics = size(initial_metric)
    om = OnlineMoments(T, dims, metrics)
    og = OnlineMoments(T, dims, metrics)
    return MetricFisherDivergence(om, og, initial_metric)
end

function update!(mfd::MetricFisherDivergence, x::AbstractMatrix, ldg, args...; kwargs...)
    grads = similar(mfd.metric)
    for c in axes(grads, 2)
        _, grads[:, c] = ldg(x[:, c]; kwargs...)
    end
    update!(mfd.om, x; kwargs...)
    update!(mfd.og, grads; kwargs...)
end

function optimum(mfd::MetricFisherDivergence, args...; kwargs...)
    return sqrt.(optimum(mfd.om) ./ mfd.og.v)
end

function reset!(mfd::MetricFisherDivergence, args...; kwargs...)
    reset!(mfd.om; kwargs...)
    reset!(mfd.og; kwargs...)
end
