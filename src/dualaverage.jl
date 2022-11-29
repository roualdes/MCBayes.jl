# TODO will want a copy method; for why?
# TODO finish converting to handle multiple chains within one struct
# TODO add methods which extract the pieces of most interest: stepsize, ...?
# and then the Adam optimized stepsize will need a stepsize method too
# TODO need a method to set ε, from initialize_stepsize

struct DualAverage{T <: AbstractFloat} <: AbstractAdapter
    ε::Vector{T}
    μ::Vector{T}
    εbar::Vector{T}
    sbar::Vector{T}
    xbar::Vector{T}
    γ::Vector{T}
    δ::Vector{T}
    κ::Vector{T}
    t0::Vector{T}
    counter::Vector{Int}
    initializer::Symbol
    update::Bool
end

function DualAverage(chains;
                     ε = 1.0
                     μ = log(10 * ε),
                     δ = 0.8,
                     γ = 0.05,
                     t0 = 10.0
                     κ = 0.75,
                     initializer = :stan,
                     update = true)
    T = eltype(ε)
    return DualAverage([ε for _ in 1:chains],
                       [μ for _ in 1:chains],
                       zeros(T, chains),
                       zeros(T, chains),
                       zeros(T, chains),
                       [oftype(ε, γ)  for _ in 1:chains],
                       [oftype(ε, δ)  for _ in 1:chains],
                       [oftype(ε, κ)  for _ in 1:chains],
                       [oftype(ε, t0) for _ in 1:chains],
                       zeros(Int, chains),
                       initializer,
                       update)
end

function _update_one!(da::DualAverage, αs; kwargs...)
    if da.update
        for chain in eachindex(αs)
            da.counter[chain] += 1
            α = αs[chain] > 1 ? 1 : αs[chain]
            eta = 1 / (da.counter[chain] + da.t0[chain])
            da.sbar[chain] = (1 - eta) * da.sbar[chain] + eta * (da.δ[chain] - α)
            x = da.μ[chain] - da.sbar[chain] * sqrt(da.counter[chain]) / da.γ[chain]
            xeta = da.counter[chain] ^ -da.κ[chain]
            da.xbar[chain] = xeta * x + (1 - xeta) * da.xbar[chain]
            da.ε[chain] = exp(x)
            da.εbar[chain] = exp(da.xbar[chain])
        end
    end
end

function reset!(da::DualAverage)
    for chain in eachindex(da.ε)
        da.μ[chain] = log(10 * da.ε[chain])
        da.sbar = zero(da.sbar)
        da.xbar = zero(da.xbar)
        da.counter = zero(da.counter)
    end
end


function stepsize(da::DualAverage, chain; weighted_average = false, kwargs...)
    stepsize = da.ε[chain]
    if weighted_average
        stepsize = da.εbar[chain]
    end
    return stepsize
end
