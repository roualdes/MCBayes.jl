struct Adam{T<:AbstractFloat}
    decaysteps::Int
    m::Vector{T}
    v::Vector{T}
    α::T
    β1::T
    β2::T
    ι::T
    schedule::Symbol
end

function Adam(
    dims, decaysteps, T=Float64; α=0.05, β1=0.0, β2=0.95, ι=1e-8, adam_schedule = :linear, kwargs...
)
    return Adam(
        decaysteps,
        zeros(T, dims),
        zeros(T, dims),
        convert(T, α)::T,
        convert(T, β1)::T,
        convert(T, β2)::T,
        convert(T, ι)::T,
        adam_schedule,
    )
end

function learningrate(adm::Adam, t)
    frac = min(t, adm.decaysteps) / adm.decaysteps
    decay = if adm.schedule == :linear
        1 - frac
    elseif adm.schedule == :cosine
        0.5 * (1 + cos(pi * frac))
    elseif adm.schedule == :none
        1
    end
    return decay * adm.α
end

"""
Adam update.
"""
function update!(adm::Adam, g, t; kwargs...)
    @. adm.m = adm.β1 * adm.m + (1 - adm.β1) * g
    @. adm.v = adm.β2 * adm.v + (1 - adm.β2) * g^2
    lr = learningrate(adm, t)
    a = lr * sqrt(1 - adm.β2^t) / (1 - adm.β1^t)
    return a .* adm.m ./ (sqrt.(adm.v) .+ adm.ι)
    # TODO implement Adamw? adm.λ = 0.0001, x = previous state
    # return a .* (adm.m ./ (sqrt.(adm.v) .+ adm.ι) .+ adm.λ .* x)
end

function reset!(adm::Adam; initial_stepsize=1, kwargs...)
    adm.m .= 0
    adm.v .= 0
end
