# consider the following
abstract type AbstractAdaptationSchedule end

mutable struct WindowedAdaptation <: AbstractAdaptationSchedule
    closewindow::Int
    windowsize::Int
    const firstwindow::Int
    const lastwindow::Int
end

mutable struct Adam{T <: AbstractFloat, S <: AbstractSchedule} <: AbstractAdapter
    x::T
    m::T
    v::T
    schedule::S
    const α::T
    const β1::T
    const β2::T
    const ι::T
    const update::Bool
end

# old
mutable struct Adam{T <: AbstractFloat} <: AbstractAdapter
    x::T
    m::T
    v::T
    const α::T
    const β1::T
    const β2::T
    const ι::T
    const update::Bool
end

function Adam(; α = 0.05, β1 = 0.0, β2 = 0.5, ι = 1e-8)
    T = eltype(α)
    return Adam(
        zero(T),
        zero(T),
        convert(T, α),
        convert(T, β1),
        convert(T, β2),
        convert(T, ι))
end

function update!(adm::Adam, g, t)
    if adm.update
        amd.m = adm.β1 * adm.m + (1 - adm.β1) * g
        adm.v = adm.β2 * adm.v + (1 - adm.β2) * g ^ 2
        a = adm.α * sqrt(1 - adm.β2 ^ t) / (1 - adm.β1 ^ t)
        adm.x += a * adm.m / (sqrt(adm.v) + adm.ι)
    end
end

function reset!(adm)
    T = eltype(adm.x)
    m = zero(T)
    v = zero(T)
end
