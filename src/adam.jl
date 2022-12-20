struct Adam{T <: AbstractFloat}
    m::Vector{T}
    v::Vector{T}
    α::T
    β1::T
    β2::T
    ι::T
    schedule::Symbol
end

function Adam(chains; α = 0.05, β1 = 0.0, β2 = 0.5, ι = 1e-8, schedule = :constant)
    T = eltype(α)
    return Adam(
        zeros(T, chains),
        zeros(T, chains),
        α,
        convert(T, β1),
        convert(T, β2),
        convert(T, ι),
        schedule)
end

"""
Adam update.
"""
function update!(adm::Adam, g, t; kwargs...)
    @. adm.m = adm.β1 * adm.m + (1 - adm.β1) * g
    @. adm.v = adm.β2 * adm.v + (1 - adm.β2) * g ^ 2

    warmup = get(kwargs, :warmup, 1000)
    lr = learningrate(adm.schedule, i, adm.α, warmup)
    a = lr * sqrt(1 - adm.β2 ^ t) / (1 - adm.β1 ^ t)
    return a * adm.m / (sqrt(adm.v) + adm.ι)
end

function reset!(adm::Adam; initial_stepsize = 1)
    adm.m .= 0
    adm.v .= 0
end

function learningrate(schedule::Symbol, i, initialvalue, decaysteps)
    learningrate(Val{schedule}(), i, initialvalue, decaysteps)
end

function learningrate(::Val{:constant}, i, initialvalue, decaysteps)
    return initialvalue
end

function learningrate(::Val{:cosine}, i, initialvalue, decaysteps)
    count = min(i, decaysteps)
    decay = 0.5 * (1 + cos(pi * count / decaysteps))
    return initialvalue * decay
end
