mutable struct Adam{T <: AbstractFloat} <: AbstractAdapter
    x::T
    xbar::T
    m::T
    v::T
    α::T
    # learning_schedule::S        # TODO implement a la optax learning rate schedules
    const β1::T
    const β2::T
    const ι::T
    const initializer::Symbol
    const update::Bool
end

function Adam(;
              α = 0.05,
              β1 = 0.0,
              β2 = 0.5,
              ι = 1e-8,
              initializer = :sga,
              update = true)
    T = eltype(α)
    return Adam(
        zero(T),
        zero(T),
        convert(T, α),
        convert(T, β1),
        convert(T, β2),
        convert(T, ι),
        initializer,
        update)
end

"""
Update Adam's x on log scale.
"""
function update!(adm::Adam, g, t; γ = -0.6)
    if adm.update
        amd.m = adm.β1 * adm.m + (1 - adm.β1) * g
        adm.v = adm.β2 * adm.v + (1 - adm.β2) * g ^ 2
        a = adm.α * sqrt(1 - adm.β2 ^ t) / (1 - adm.β1 ^ t)
        adm.x *= exp(a * adm.m / (sqrt(adm.v) + adm.ι))
        # TODO wouldn't a learning schedule remove the need for this
        # long run averaging?
        w = t ^ γ
        adm.xbar = exp(w * log(adm.x) + (1 - w) * log(1e-10 + adm.xbar))
    end
end

function reset!(adm)
    T = eltype(adm.x)
    m = zero(T)
    v = zero(T)
end

function optimum(adm::Adam)
    return adm.xbar
end
