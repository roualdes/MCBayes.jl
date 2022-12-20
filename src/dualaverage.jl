struct DualAverage{T <: AbstractFloat}
    sbar::Vector{T}
    xbar::Vector{T}
    μ::Vector{T}
    γ::Vector{T}
    δ::Vector{T}
    κ::Vector{T}
    t0::Vector{T}
    counter::Vector{Int}
end

function DualAverage(chains;
                     μ = log(10 * 1.0),
                     δ = 0.8,
                     γ = 0.05,
                     t0 = 10.0,
                     κ = 0.75)
    T = eltype(μ)
    return DualAverage(zeros(T, chains),
                       zeros(T, chains),
                       μ .* ones(T, chains),
                       γ .* ones(T, chains),
                       δ .* ones(T, chains),
                       κ .* ones(T, chains),
                       t0 .* ones(T, chains),
                       zeros(Int, chains))
end

function update!(da::DualAverage, αs; kwargs...)
    da.counter .+= 1
    α = [a > 1 ? 1 : a for a in αs]
    eta = 1 ./ (da.counter .+ da.t0)
    @. da.sbar = (1 - eta) * da.sbar + eta * (da.δ - α)
    x = da.μ .- da.sbar .* sqrt.(da.counter) ./ da.γ
    xeta = da.counter .^ -da.κ
    @. da.xbar = xeta * x + (1 - xeta) * da.xbar
    return exp.(x), exp.(da.xbar)
end

function reset!(da::DualAverage; initial_stepsize = 1)
    @. da.μ = log(10 * oftype(da.μ, initial_stepsize))
    da.sbar .= 0
    da.xbar .= 0
    da.counter .= 0
end

function optimum(da::DualAverage)
    return da.εbar
end
