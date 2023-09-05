abstract type AbstractReductionFactorAdapter{T} end

Base.eltype(::AbstractReductionFactorAdapter{T}) where {T} = T

function optimum(sa::AbstractReductionFactorAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? sa.reductionfactor_bar : sa.reductionfactor
end

function set!(sampler, sa::AbstractReductionFactorAdapter, args...; kwargs...)
    sampler.reductionfactor .= optimum(sa; kwargs...)
end

struct ReductionFactorDualAverage{T <: AbstractFloat} <: AbstractReductionFactorAdapter{T}
    da::DualAverage{T}
    m::Vector{T}
    N::Vector{Int}
    reductionfactor::Vector{T}
    reductionfactor_bar::Vector{T}
    δ::Vector{T}
end

function ReductionFactorDualAverage(initial_reductionfactor::AbstractVector{T}, chains, args...; reductionfactor_δ = 0.95, kwargs...) where {T}
    da = DualAverage(1, T; μ = log(reductionfactor_δ))
    m = zeros(T, chains)
    N = zeros(Int, 1)
    return ReductionFactorDualAverage(da, m, N, copy(initial_reductionfactor), copy(initial_reductionfactor), fill(convert(T, reductionfactor_δ)::T, 1))
end

function update!(sa::ReductionFactorDualAverage, α, args...; kwargs...)
    # transform to keep in [1, 10]
    # https://mc-stan.org/docs/reference-manual/logit-transform-jacobian.html#lower-and-upper-bounds-inverse-transform
    sa.N[1] += 1
    @. sa.m += (α - sa.m) / sa.N[1]
    y = update!(sa.da, mean(sa.m) .- sa.δ; kwargs...)
    sa.reductionfactor .= 9 .* (y ./ (1 .+ y)) .+ 1
    y_bar = optimum(sa.da)
    sa.reductionfactor_bar .= 9 .* (y_bar ./ (1 .+ y_bar)) .+ 1
end

function reset!(sa::ReductionFactorDualAverage, args...; kwargs...)
    sa.N .= 0
    sa.m .= 0
    reset!(sa.da; μ = sa.δ[1])
end

struct ReductionFactorConstant{T<:AbstractFloat} <: AbstractReductionFactorAdapter{T}
    reductionfactor::Vector{T}
    reductionfactor_bar::Vector{T}
end

function ReductionFactorConstant(initial_reductionfactor::AbstractVector, args...; kwargs...)
    return ReductionFactorConstant(copy(initial_reductionfactor), copy(initial_reductionfactor))
end

function update!(sc::ReductionFactorConstant, args...; kwargs...)
end

function reset!(sc::ReductionFactorConstant, args...; kwargs...)
end
