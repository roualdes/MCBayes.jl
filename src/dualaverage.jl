struct DualAverage{T<:AbstractFloat}
    sbar::Vector{T}
    xbar::Vector{T}
    μ::Vector{T}
    γ::Vector{T}
    δ::Vector{T}
    κ::Vector{T}
    t0::Vector{T}
    counter::Vector{Int}
end

# TODO want ensure_vector or some such function
# TODO enable μ to be set as log(10 * initial_stepsize)
function DualAverage(chains, T=Float64; μ=log(10), δ=0.8, γ=0.05, t0=10.0, κ=0.75)
    return DualAverage(
        zeros(T, chains),
        zeros(T, chains),
        fill(convert(T, μ)::T, chains),
        fill(convert(T, γ)::T, chains),
        fill(convert(T, δ)::T, chains),
        fill(convert(T, κ)::T, chains),
        fill(convert(T, t0)::T, chains),
        zeros(Int, chains),
    )
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

function reset!(da::DualAverage; initial_stepsize=1)
    @. da.μ = log(10 * oftype(da.μ, initial_stepsize))
    da.sbar .= 0
    da.xbar .= 0
    da.counter .= 0
end

function optimum(da::DualAverage)
    return da.εbar
end
