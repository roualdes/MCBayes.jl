abstract type AbstractStepsAdapter{T} end

Base.eltype(::AbstractStepsAdapter{T}) where {T} = T

function set!(sampler, sa::AbstractStepsAdapter, args...; kwargs...)
    sampler.steps .= sa.steps
end

struct StepsPCA{T <: Int} <: AbstractStepsAdapter{T}
    steps::Vector{T}
end

function update!(sa::StepsPCA, m, lambda_max, stepsize::AbstractVector, args...; max_steps = 1000, kwargs...)
    L = Iterators.cycle(1:length(lambda_max))
    for (l, i) in zip(L, 1:length(stepsize))
        step = lambda_max[l] / stepsize[i]
        step = ifelse(isfinite(step), step, 10)
        step = clamp(step, 1, max_steps)
        step = min(m, step)
        sa.steps[i] = round(Int, step)
    end
end

function update!(sa::StepsPCA, m, lambda_max, stepsize::T, args...; max_steps = 1000, kwargs...) where {T}
    step = lambda_max / stepsize
    step .= ifelse.(isfinite.(step), step, 10)
    step .= clamp.(step, 1, max_steps)
    # step .= min.(m, step)
    sa.steps .= round.(Int, step)
end

function reset!(sa::StepsPCA, args...; kwargs...)
    sa.steps .= 1
end


struct StepsConstant{T<:Integer} <: AbstractStepsAdapter{T}
    steps::Vector{T}
end

function StepsConstant(initial_steps::AbstractVector, args...; kwargs...)
    return StepsConstant(copy(initial_steps))
end

function update!(sc::StepsConstant, args...; kwargs...)
end

function reset!(sc::StepsConstant, args...; kwargs...)
end

struct StepsAdamSNAPER{T<:AbstractFloat} <: AbstractStepsAdapter{T}
    steps::Vector{T}
    das::AdamSNAPER{T}
end

function StepsAdamSNAPER(initial_steps::AbstractVector{T}, initial_trajectorylength::AbstractVector{T}, dims, warmup, args...; kwargs...) where {T}
    # das = DualAverageSNAPER(copy(initial_trajectorylength), dims; kwargs...)
    das = AdamSNAPER(copy(initial_trajectorylength), dims, warmup; kwargs...)
    return StepsAdamSNAPER(copy(initial_steps), das)
end

function update!(sa::StepsAdamSNAPER, m, αs, previous_positions, proposed_positions, previous_momentum, proposed_momentum, stepsize, pca, ldg!, args...; max_steps = 1_000, kwargs...)

    update!(sa.das, m, αs, previous_positions, previous_momentum, proposed_momentum, proposed_positions, stepsize, pca, ldg!; kwargs...)

    τ = optimum(sa.das; kwargs...)[1]
    step = τ / stepsize
    step = ifelse(isfinite(step), step, 10)
    w = 1 / m
    sa.steps .= sa.steps .* w .+ (1 - w) .* clamp(step, 1, max_steps)
end

function reset!(sa::StepsAdamSNAPER, lambda, stepsize; max_steps = 1_000, kwargs...)
    sa.steps .= ifelse.(isfinite.(lambda), clamp.(lambda, 1, max_steps), 10)
    reset!(sa.das, stepsize; kwargs...)
end
