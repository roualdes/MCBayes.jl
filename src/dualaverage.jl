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

function update!(da::DualAverage, αs; kwargs...)
    if da.update
        da.counter .+= 1
        α = [a > 1 ? 1 : a for a in αs]
        eta = 1 ./ (da.counter .+ da.t0)
        @. da.sbar = (1 - eta) * da.sbar + eta * (da.δ - a)
        x = da.μ .- da.sbar .* sqrt.(da.counter) ./ da.γ
        xeta = da.counter .^ -da.κ
        @. da.xbar = xeta * x + (1 - xeta) * da.xbar
        @. da.ε = exp(x)
        @. da.εbar = exp(da.xbar)
    end
end

function reset!(da::DualAverage)
    @. da.μ = log(10 * da.ε)
    da.sbar .= 0
    da.xbar .= 0
    da.counter .= 0
end


function stepsize(da::DualAverage; weighted_average = false, kwargs...)
    stepsize = da.ε
    if weighted_average
        stepsize = da.εbar
    end
    return stepsize
end
