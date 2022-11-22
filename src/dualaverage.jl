# TODO will want a copy method; for why?

mutable struct DualAverage{T <: AbstractFloat} <: AbstractAdapter
    ε::T
    μ::T
    εbar::T
    sbar::T
    xbar::T
    counter::Int
    const γ::T
    const δ::T
    const κ::T
    const t0::T
    const update::Bool
end

function DualAverage(;
                     ε = 1.0
                     μ = log(10 * ε),
                     δ = 0.8,
                     γ = 0.05,
                     t0 = 10.0
                     κ = 0.75,
                     update = true)
    T = eltype(ε)
    return DualAverage(ε,
                       convert(T, μ),
                       zero(T),
                       zero(T),
                       zero(T),
                       0,
                       convert(T, γ),
                       convert(T, δ),
                       convert(T, κ),
                       convert(T, t0),
                       update)
end

function update!(da::DualAverage, α)
    if da.update
        da.counter += 1
        α = α > 1 ? 1 : α
        eta = 1 / (da.counter + da.t0)
        da.sbar = (1 - eta) * da.sbar + eta * (da.δ - α)
        x = da.μ - da.sbar * sqrt(da.counter) / da.γ
        xeta = da.counter ^ -da.κ
        da.xbar = xeta * x + (1 - xeta) * da.xbar
        da.ε = exp(x)
        da.εbar = exp(da.xbar)
    end
end

function reset!(da::DualAverage)
    da.μ = log(10 * da.ε)
    T = eltype(da.sbar)
    da.sbar = zero(T)
    da.xbar = zero(T)
    da.counter = zero(da.counter)
end
