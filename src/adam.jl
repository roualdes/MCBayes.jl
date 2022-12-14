struct Adam{T<:AbstractFloat}
    m::Vector{T}
    v::Vector{T}
    α::Vector{T}
    β1::Vector{T}
    β2::Vector{T}
    ι::Vector{T}
    schedule::Symbol
end

function Adam(chains, T=Float64; α=0.05, β1=0.0, β2=0.5, ι=1e-8, schedule=:constant)
    return Adam(
        zeros(T, chains),
        zeros(T, chains),
        fill(convert(T, α)::T, chains),
        fill(convert(T, β1)::T, chains),
        fill(convert(T, β2)::T, chains),
        fill(convert(T, ι)::T, chains),
        schedule,
    )
end

"""
Adam update.
"""
function update!(adm::Adam, g, t; kwargs...)
    @. adm.m = adm.β1 * adm.m + (1 - adm.β1) * g
    @. adm.v = adm.β2 * adm.v + (1 - adm.β2) * g^2

    warmup = get(kwargs, :warmup, 1000)
    lr = learningrate(adm.schedule, t, adm.α, warmup)
    a = lr * sqrt(1 - adm.β2^t) / (1 - adm.β1^t)
    return a * adm.m / (sqrt(adm.v) + adm.ι)
end

function reset!(adm::Adam; initial_stepsize=1, kwargs...)
    adm.m .= 0
    adm.v .= 0
end

function learningrate(schedule::Symbol, i, initialvalue, decaysteps)
    return learningrate(Val{schedule}(), i, initialvalue, decaysteps)
end

function learningrate(::Val{:constant}, i, initialvalue, decaysteps)
    return initialvalue
end

function learningrate(::Val{:cosine}, i, initialvalue, decaysteps)
    count = min(i, decaysteps)
    decay = 0.5 * (1 + cos(pi * count / decaysteps))
    return initialvalue * decay
end
