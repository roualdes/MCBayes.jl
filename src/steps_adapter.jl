abstract type AbstractStepsAdapter{T} end

Base.eltype(::AbstractStepsAdapter{T}) where {T} = T

function set!(sampler, sa::AbstractStepsAdapter, args...; kwargs...)
    sampler.steps .= sa.steps
end

struct StepsPCA{T <: Int} <: AbstractStepsAdapter{T}
    steps::Vector{T}
end

function update!(sa::StepsPCA, m, lambda_max, stepsize::AbstractVector, n, args...; max_steps = 1000, kwargs...)
    w = n / (n + 5)
    L = Iterators.cycle(1:length(lambda_max))
    for (l, i) in zip(L, 1:length(stepsize))
        step = lambda_max[l] / stepsize[i]
        step = ifelse(isfinite(step), step, 10)
        step = w * step + (1 - w) * 10
        step = clamp(step, 1, max_steps)
        step = round(Int, min(m, step))
        sa.steps[i] = step
    end
end

function update!(sa::StepsPCA, m, lambda_max, stepsize::T, args...; max_steps = 1000, kwargs...) where {T}
    step = lambda_max / stepsize
    step = ifelse(isfinite(step), step, 1)
    step = round(Int, clamp(step, 1, max_steps))
    sa.steps .= min(m, step)
end

function reset!(sa::StepsPCA, args...; kwargs...)
    sa.steps .= 1
end


struct StepsConstant{T<:Integer} <: AbstractStepsAdapter{T}
    steps::Vector{T}
end

function StepsConstant(initial_steps::AbstractVector, args...; kwargs...)
    return StepsConstant(initial_steps)
end

function update!(sc::StepsConstant, args...; kwargs...)
end

function reset!(sc::StepsConstant, args...; kwargs...)
end

struct StepsTrajectorylengthDualAverage{T<:AbstractFloat} <: AbstractStepsAdapter{T}
    steps::Vector{Int}
    tla::TrajectorylengthDualAverageLDG{T}
end

function StepsTrajectorylengthDualAverage(initial_steps::AbstractVector{Int}, stepsize, args...; kwargs...)
    tla = TrajectorylengthDualAverageLDG(stepsize)
    return StepsTrajectorylengthDualAverage(initial_steps, tla)
end

function update!(sa::StepsTrajectorylengthDualAverage, m, αs, previouspositions, proposedpositions, proposedmomentum, stepsize, ldg!, args...; max_steps = 1000, kwargs...)
    update!(sa.tla, αs, previouspositions, proposedpositions, proposedmomentum, stepsize, ldg!; kwargs...)
    τ = optimum(sa.tla, smoothed = false)[1]
    for (si, ss) in pairs(stepsize)
        step = τ / ss
        step = ifelse(isfinite(step), step, 1)
        # step = 2 * rand() * step
        # step = w * step + (1 - w) * 10
        step = round(Int, clamp(step, 1, max_steps))
        step = min(m, step)
        sa.steps[si] = step
    end
end

function reset!(sa::StepsTrajectorylengthDualAverage, args...; μ = 1, kwargs...)
    reset!(tla, μ = 0.5 * sa.tla.trajectorylength[1])
end
