struct Adam{T <: AbstractFloat}
    x::Vector{T}
    xbar::Vector{T}
    m::Vector{T}
    v::Vector{T}
    α::T
    β1::T
    β2::T
    ι::T
    initializer::Symbol
    schedule::Symbol
    update::Bool
    smooth::Bool
end

function Adam(chains;
              α = 0.05,
              β1 = 0.0,
              β2 = 0.5,
              ι = 1e-8,
              initializer = :sga,
              schedule = :constant,
              update = true,
              smooth = true)
    T = eltype(α)
    return Adam(
        zeros(T, chains),
        zeros(T, chains),
        zeros(T, chains),
        zeros(T, chains),
        α,
        convert(T, β1),
        convert(T, β2),
        convert(T, ι),
        initializer,
        schedule,
        update,
        smooth)
end

"""
Update Adam's x on log scale.
"""
function update!(adm::Adam, g, t; γ = -0.6, kwargs...)
    if adm.update
        @. adm.m = adm.β1 * adm.m + (1 - adm.β1) * g
        @. adm.v = adm.β2 * adm.v + (1 - adm.β2) * g ^ 2

        warmup = get(kwargs, :warmup, 1000)
        lr = learningrate(adm.schedule, i, adm.α, warmup)
        a = lr * sqrt(1 - adm.β2 ^ t) / (1 - adm.β1 ^ t)

        @. adm.x *= exp(a * adm.m / (sqrt(adm.v) + adm.ι))
        # TODO wouldn't a learning schedule remove the need for this
        # long run averaging?
        if adm.smooth
            w = t ^ γ
            @. adm.xbar = exp(w * log(adm.x) + (1 - w) * log(1e-10 + adm.xbar))
        else
            adm.xbar .= adm.x
        end
    end
end

"""
Noop update.
"""
function update!(x::NamedTuple, g, t; kwargs...)
end

function reset!(adm::Adam)
    adm.m .= 0
    adm.v .= 0
    adm.x .= 0
    adm.xbar .= 0
end

function optimum(adm::Adam)
    return adm.xbar
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
