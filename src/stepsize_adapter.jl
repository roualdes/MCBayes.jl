abstract type AbstractStepsizeAdapter{T} end

Base.eltype(::AbstractStepsizeAdapter{T}) where {T} = T

function optimum(ssa::AbstractStepsizeAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? ssa.stepsize_bar : ssa.stepsize
end

function set!(sampler, ssa::AbstractStepsizeAdapter, args...; kwargs...)
    sampler.stepsize .= optimum(ssa; kwargs...)
end

struct StepsizeAdam{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    adam::Adam{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    δ::T
    alpha::T
end

function StepsizeAdam(
    initial_stepsize::AbstractVector{T},
    warmup;
    δ=0.8,
    stepsize_smoothing_factor=1 - 8 / 9,
    kwargs...,
) where {T<:AbstractFloat}
    chains = length(initial_stepsize)
    adam = Adam(chains, warmup, T; kwargs...)
    return StepsizeAdam(
        adam,
        copy(initial_stepsize),
        copy(initial_stepsize),
        convert(T, δ)::T,
        convert(T, stepsize_smoothing_factor)::T,
    )
end

"""
Adam update on log-scale.
"""
function update!(ssa::StepsizeAdam, abar, m, args...; stepsize_smooth=true, kwargs...)
    x = update!(ssa.adam, abar - ssa.δ, m; kwargs...)
    @. ssa.stepsize *= exp(x)
    w = ssa.alpha + (1 - stepsize_smooth) * (1 - ssa.alpha)
    @. ssa.stepsize_bar = exp(
        w * log(ssa.stepsize) + (1 - w) * log(1e-10 + ssa.stepsize_bar)
    )
end

function reset!(ssa::StepsizeAdam, args...; kwargs...)
    reset!(ssa.adam; initial_stepsize=copy(ssa.stepsize), kwargs...)
end

struct StepsizeDualAverage{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    da::DualAverage{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    δ::Vector{T}
end

"""
    StepsizeDualAverage(initial_stepsize::Vector)

Construct a stepsize adapter using the dual averaging method by [Nesterov 2009](https://link.springer.com/article/10.1007/s10107-007-0149-x), as used in [Stan](https://mc-stan.org/docs/reference-manual/hmc-algorithm-parameters.html#ref-Nesterov:2009).  The length of `initial_stepsize::Vector` must be appropriate for the sampling algorithm for which this stepsize adapter will be used.
"""
function StepsizeDualAverage(
    initial_stepsize::AbstractVector{T}; δ = 0.6, kwargs...
) where {T<:AbstractFloat}
    chains = length(initial_stepsize)
    da = DualAverage(chains, T; kwargs...)
    return StepsizeDualAverage(da,
                               copy(initial_stepsize),
                               copy(initial_stepsize),
                               fill(convert(T, δ)::T, 1)
)
end

function update!(ssa::StepsizeDualAverage, αs, args...; kwargs...)
    ssa.stepsize .= update!(ssa.da, ssa.δ .- αs; kwargs...)
    ssa.stepsize_bar .= optimum(ssa.da)
end

function reset!(ssa::StepsizeDualAverage, args...; kwargs...)
    reset!(ssa.da; μ = 10 .* ssa.stepsize, kwargs...)
end

struct StepsizeConstant{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
end

"""
    StepsizeConstant(initial_stepsize::Vector)

Construct a stepsize adapter for which the stepsize is fixed at it's initial value.
"""
function StepsizeConstant(
    initial_stepsize::AbstractVector{T}; kwargs...
) where {T<:AbstractFloat}
    chains = length(initial_stepsize)
    return StepsizeConstant(copy(initial_stepsize), copy(initial_stepsize))
end

function update!(ssc::StepsizeConstant, args...; kwargs...) end

function reset!(ssc::StepsizeConstant, args...; kwargs...) end

# TODO not convinced this is necessary
# function set!(sampler, ssc::StepsizeConstant, args...; kwargs...)
#     sampler.stepsize .= ssc.stepsize
# end

struct StepsizeECA{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
end

function StepsizeECA(
    initial_stepsize::AbstractVector{T}; kwargs...
) where {T<:AbstractFloat}
    return StepsizeECA(copy(initial_stepsize), copy(initial_stepsize))
end

function update!(seca::StepsizeECA, ldg!, positions, scale, idx, args...; kwargs...)
    dims, chains = size(positions)
    grads = [zeros(dims) for chain in 1:chains]
    for chain in axes(positions, 2)
        q = positions[:, chain]
        ldg!(q, grads[chain]; kwargs...)
    end
    gradients = reduce(hcat, grads)
    scaled_gradients = gradients .* scale
    seca.stepsize[idx] = min(1, 0.5 / sqrt(max_eigenvalue(scaled_gradients)))
    seca.stepsize_bar[idx] = seca.stepsize[idx]
end

function reset!(seca::StepsizeECA, args...; kwargs...) end

function set!(sampler, seca::StepsizeECA, idx, args...; kwargs...)
    sampler.stepsize[idx] = seca.stepsize[idx]
end

struct StepsizeGradientPCA{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    opca::OnlinePCA{T}
    alpha::T
end

function StepsizeGradientPCA(
    initial_stepsize::AbstractVector{T}, dims; stepsize_smoothing_factor=1 - 3 / 4, l = 2.0, kwargs...
        ) where {T<:AbstractFloat}
    opca = OnlinePCA(T, dims, 1, convert(T, l)::T)
    return StepsizeGradientPCA(copy(initial_stepsize), copy(initial_stepsize), opca, stepsize_smoothing_factor)
end

function update!(ssg::StepsizeGradientPCA, αs, positions, ldg!, scale, args...; stepsize_factor = 0.5, stepsize_smooth=true, kwargs...)
    dims, chains = size(positions)
    T = eltype(positions)

    gradients = [zeros(dims) for chain in 1:chains]
    @views for chain in 1:chains
        q = positions[:, chain]
        ldg!(q, gradients[chain]; kwargs...)
    end

    grads = reduce(hcat, gradients)

    u = Vector{T}(undef, dims)
    f = Vector{T}(undef, dims)
    g = Vector{T}(undef, dims)

    m = zeros(T, dims)
    s = ones(T, dims)

    n = ssg.opca.n[1]
    l = ssg.opca.l
    for chain in 1:chains
        n += 1
        g .= grads[:, chain] .* scale # .+ 1e-10?
        update!(ssg.opca.pc, g, m, s, n, l, u, f)
    end

    ssg.opca.n[1] = n

    lambda_max = norm(ssg.opca.pc)
    # println("lambda_max = $(lambda_max)")
    if isnan(lambda_max)
        lambda_max = 1
    end

    ssg.stepsize .= 0.5 / (lambda_max + 1e-10)
    ssg.stepsize .= minimum.(1, ss)
    # println("stepsize $(ssg.stepsize)")

    w = ssg.alpha + (1 - stepsize_smooth) * (1 - ssg.alpha)
    ssg.stepsize_bar .= w .* ssg.stepsize .+ (1 - w) .* ssg.stepsize_bar
end

function reset!(ssg::StepsizeGradientPCA, args...; kwargs...)
    ssg.stepsize .= 1
    ssg.stepsize_bar .= 1
    reset!(ssg.opca; reset_pc = true, kwargs...)
end


struct StepsizeChEES{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    tla::TrajectorylengthChEES{T}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    alpha::T
end

function StepsizeChEES(initial_stepsize::AbstractVector{T}, dims, warmup, args...; stepsize_smoothing_factor=1 - 8 / 9, kwargs...) where {T}
    tla = TrajectorylengthChEES(copy(initial_stepsize), dims, warmup; kwargs...)
    alpha = convert(T, stepsize_smoothing_factor)::T
    return StepsizeChEES(tla, copy(initial_stepsize), copy(initial_stepsize), alpha)
end

function update!(ss::StepsizeChEES, m, αs, previouspositions, previousmomentum, proposedmomentum, proposedpositions, stepsize, pca, ldg!, args...; max_steps = 1000, stepsize_smooth = true, kwargs...)
    update!(ss.tla, m, αs, previouspositions, previousmomentum, proposedmomentum, proposedpositions, stepsize, pca, ldg!; kwargs...)
    τ = optimum(ss.tla, smoothed = false)
    ss.stepsize .= τ / 1
    w = ss.alpha + (1 - stepsize_smooth) * (1 - ss.alpha)
    ss.stepsize_bar .= w .* ss.stepsize .+ (1 - w) .* ss.stepsize_bar
end

function reset!(ss::StepsizeChEES, stepsize, decay_steps, args...; kwargs...)
    reset!(ss.tla, stepsize, decay_steps; kwargs...)
end


struct StepsizeFirstTryDualAverage{T<:AbstractFloat} <: AbstractStepsizeAdapter{T}
    da::DualAverage{T}
    m::Vector{T}
    N::Vector{Int}
    stepsize::Vector{T}
    stepsize_bar::Vector{T}
    δ::Vector{T}
end

function StepsizeFirstTryDualAverage(
    initial_stepsize::AbstractVector{T}, chains; δ = 0.5, kwargs...
) where {T<:AbstractFloat}
    da = DualAverage(1, T; kwargs...)
    m = zeros(T, chains)
    N = zeros(Int, 1)
    return StepsizeFirstTryDualAverage(da,
                               m,
                               N,
                               copy(initial_stepsize),
                               copy(initial_stepsize),
                               fill(convert(T, δ)::T, 1)
)
end

function update!(ssa::StepsizeFirstTryDualAverage, firsttries, args...; kwargs...)
    ssa.N[1] += 1
    @. ssa.m += (firsttries - ssa.m) / ssa.N[1]
    println("cumulative mean first tries: $(ssa.m)")
    ssa.stepsize .= update!(ssa.da, ssa.δ .- inv(mean(inv, ssa.m .+ 1e-10)); kwargs...)
    ssa.stepsize_bar .= optimum(ssa.da)
end

function reset!(ssa::StepsizeFirstTryDualAverage, args...; kwargs...)
    ssa.N .= 0
    ssa.m .= 0
    reset!(ssa.da; μ = 10 .* ssa.stepsize, kwargs...)
end
