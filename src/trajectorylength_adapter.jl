abstract type AbstractTrajectorylengthAdapter{T} end

Base.eltype(::AbstractTrajectorylengthAdapter{T}) where {T} = T

function optimum(tla::AbstractTrajectorylengthAdapter, args...; smoothed=false, kwargs...)
    return smoothed ? tla.trajectorylength_bar : tla.trajectorylength
end

function set!(sampler, tla::AbstractTrajectorylengthAdapter, args...; kwargs...)
    if :trajectorylength in fieldnames(typeof(sampler))
        sampler.trajectorylength .= optimum(tla; kwargs...)
    end
end

"""
# Adapted from
# [1] https://arxiv.org/pdf/2110.11576.pdf
# [2] https://proceedings.mlr.press/v130/hoffman21a.html
# [3] https://github.com/tensorflow/probability/blob/c678caa1b8e94ab3677a37e581e8b19e68e59248/tensorflow_probability/python/experimental/mcmc/gradient_based_trajectory_length_adaptation.py
# [4] https://github.com/tensorflow/probability/blob/5ebcdf1f32ecc340dece4f21694790ea95c6c1e2/spinoffs/fun_mc/fun_mc/sga_hmc.py
# [5] https://github.com/tensorflow/probability/blob/5ebcdf1f32ecc340dece4f21694790ea95c6c1e2/spinoffs/fun_mc/fun_mc/fun_mc_lib.py
"""
struct TrajectorylengthChEES{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    om::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthChEES(
    initial_trajectorylength::AbstractVector{T}, dims; maxleapfrogsteps = 1000, kwargs...) where {T}
    adam = Adam(1, T; kwargs...)
    om = OnlineMoments(T, dims, 1)
    return TrajectorylengthChEES(adam, om, initial_trajectorylength, initial_trajectorylength, maxleapfrogsteps)
end

function update!(tlc::TrajectorylengthChEES, m, αs, draws, ps, qs, stepsize, args...; γ=-0.6, kwargs...)
    update!(tlc.om, draws[m, :, :]; kwargs...)
    ghats = trajectorylength_gradient(tlc, m, αs, draws, ps, qs, stepsize)

    αbar = inv(mean(inv, αs))
    αbar < 1e-4 && (ghats .= zero(ghats)) # [3]#L733
    !all(isfinite.(ghats)) && (ghats .= zero(ghats))

    ghat = weighted_mean(ghats, αs)
    as = update!(tlc.adam, ghat, m)[1]

    logupdate = clamp(as, -0.35, 0.35)               # [3]#L759
    T = tlc.trajectorylength[1] * exp(logupdate)     # [3]#L761
    T = clamp(T, 0, stepsize * tlc.maxleapfrogsteps) # [3]#L773

    tlc.trajectorylength[1] = T
    aw = m ^ γ
    tlc.trajectorylength_bar[1] = exp(aw * log(T) + (1 - aw) * log(1e-10 + tlc.trajectorylength_bar[1]))
end

function trajectorylength_gradient(tla::AbstractTrajectorylengthAdapter, m, αs, draws, ps, qs, stepsize)
    _, dims, chains = size(draws)

    meanθ = zeros(dims)
    meanq = zeros(dims)
    v = zero(eltype(draws))

    for chain in 1:chains
        @. meanθ += (draws[m, :, chain] - meanθ) / chain
        a = αs[chain]
        v += a
        @. meanq += a * (qs[:, chain] - meanq) / v
    end

    mw = m / (m + chains)
    @. meanθ = mw * tla.om.m + (1 - mw) * meanθ
    @. meanq = mw * tla.om.m + (1 - mw) * meanq

    ghats = sampler_trajectorylength_gradient(tla, m, draws, ps, qs, stepsize, meanθ, meanq)
    return ghats
end

function sampler_trajectorylength_gradient(tlc::TrajectorylengthChEES, m, draws, ps, qs, stepsize, mθ, mq)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(draws)
    _, dims, chains = size(draws)
    ghats = zeros(chains)
    for chain in 1:chains
        q = qs[:, chain]
        dsq = centered_cumsum(abs2, q, mq) - centered_cumsum(abs2, draws[m, :, chain], mθ)
        ghats[chain] = 4 * dsq * centered_dot(q, mq, ps[:, chain]) - dsq ^ 2 / t
        ghats[chain] *= h
    end
    return ghats
end

struct TrajectorylengthConstant{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
end

function TrajectorylengthConstant(
    initial_trajectorylength::AbstractVector{T}; kwargs...) where {T<:AbstractFloat}
    return TrajectorylengthConstant(initial_trajectorylength, initial_trajectorylength)
end

function update!(tlc::TrajectorylengthConstant, args...; kwargs...) end

function reset!(tlc::TrajectorylengthConstant, args...; kwargs...) end
