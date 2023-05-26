# Adapted from
# [1] https://arxiv.org/pdf/2110.11576.pdf
# [2] https://proceedings.mlr.press/v130/hoffman21a.html
# [3] https://github.com/tensorflow/probability/blob/c678caa1b8e94ab3677a37e581e8b19e68e59248/tensorflow_probability/python/experimental/mcmc/gradient_based_trajectory_length_adaptation.py
# [4] https://github.com/tensorflow/probability/blob/5ebcdf1f32ecc340dece4f21694790ea95c6c1e2/spinoffs/fun_mc/fun_mc/sga_hmc.py
# [5] https://github.com/tensorflow/probability/blob/5ebcdf1f32ecc340dece4f21694790ea95c6c1e2/spinoffs/fun_mc/fun_mc/fun_mc_lib.py

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

function update!(
    tla::AbstractTrajectorylengthAdapter,
    m,
    αs,
    positions,
    ps,
    qs,
    stepsize,
    r,
    ldg,
    args...;
    γ=-0.6,
    kwargs...,
)
    update!(tla.om, positions; kwargs...)
    ghats = trajectorylength_gradient(tla, m, αs, positions, ps, qs, stepsize, r, ldg)

    αbar = inv(mean(inv, αs))
    if αbar < 1e-4 || !all(isfinite.(ghats)) # [3]#L733
        ghats .= zero(ghats)
    end

    ghat = weighted_mean(ghats, αs)
    as = update!(tla.adam, ghat, m; kwargs...)[1]

    logupdate = clamp(as, -0.35, 0.35)           # [3]#L759
    T = tla.trajectorylength[1] * exp(logupdate) # [3]#L761
    T = min(T, stepsize * tla.maxleapfrogsteps)  # [3]#L773

    tla.trajectorylength[1] = T
    aw = 1 - 8/9                # TODO add this as an argument to the constructor
    # TODO and then re-design like OnlineMoments
    tla.trajectorylength_bar[1] = exp(
        aw * log(T) + (1 - aw) * log(1e-10 + tla.trajectorylength_bar[1])
    )
end

function trajectorylength_gradient(
    tla::AbstractTrajectorylengthAdapter, m, αs, positions, ps, qs, stepsize, r, ldg
)
    dims, chains = size(positions)

    meanθ = zeros(dims)
    meanq = zeros(dims)
    v = zero(eltype(positions))

    @views for chain in 1:chains
        @. meanθ += (positions[:, chain] - meanθ) / chain
        a = αs[chain]
        v += a
        @. meanq += a * (qs[:, chain] - meanq) / v
    end

    N = tla.om.n[1]
    mw = N / (N + chains)

    @. meanθ = mw * tla.om.m + (1 - mw) * meanθ
    @. meanq = mw * tla.om.m + (1 - mw) * meanq

    ghats = sampler_trajectorylength_gradient(
        tla, m, positions, ps, qs, stepsize, meanθ, meanq, r, ldg
    )
    return ghats
end

struct TrajectorylengthChEES{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    om::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthChEES(
    initial_trajectorylength::AbstractVector{T}, dims; maxleapfrogsteps=1000, kwargs...
) where {T}
    adam = Adam(1, T; α = 0.01, kwargs...)
    om = OnlineMoments(T, dims, 1)
    return TrajectorylengthChEES(
        adam, om, initial_trajectorylength, initial_trajectorylength, maxleapfrogsteps
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthChEES, m, positions, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        q = qs[:, chain]
        dsq = centered_sum(abs2, q, mq) - centered_sum(abs2, positions[:, chain], mθ)
        fd = 2 * dsq * centered_dot(q, mq, ps[:, chain])
        fd2 = 2 * -dsq * centered_dot(q, mq, -ps[:, chain])
        ghats[chain] = 2 * (fd + fd2)  - dsq ^ 2 / t
        ghats[chain] *= h
    end
    return ghats
end

struct TrajectorylengthMALT{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    om::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthMALT(
    initial_trajectorylength::AbstractVector{T}, dims; maxleapfrogsteps=1000, kwargs...
) where {T}
    adam = Adam(1, T; α = 0.01, kwargs...)
    om = OnlineMoments(T, dims, 1)
    return TrajectorylengthMALT(
        adam, om, initial_trajectorylength, initial_trajectorylength, maxleapfrogsteps
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthMALT, m, positions, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        # ChEES
        # q = qs[:, chain]
        # dsq = centered_sum(abs2, q, mq) - centered_sum(abs2, positions[:, chain], mθ)
        # fd = 2 * dsq * centered_dot(q, mq, ps[:, chain])
        # fd2 = 2 * -dsq * centered_dot(q, mq, -ps[:, chain])
        # ghats[chain] = 2 * (fd + fd2)  - dsq ^ 2 / t
        # SNAPER
        q = qs[:, chain]
        tmp = centered_dot(q, mq, r)
        dsq = tmp ^ 2 - centered_dot(positions[:, chain], mθ, r) ^ 2
        fd = 2 * tmp * dsq * dot(ps[:, chain], r)
        fd2 = 2 * tmp * -dsq * dot(ps[:, chain], -r)
        ghats[chain] = 2 * (fd + fd2) - dsq^2 / t

    end
    return ghats
end

struct TrajectorylengthSNAPER{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    om::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthSNAPER(
    initial_trajectorylength::AbstractVector{T}, dims; maxleapfrogsteps=1000, kwargs...
) where {T}
    adam = Adam(1, T; α = 0.01, kwargs...)
    om = OnlineMoments(T, dims, 1)
    return TrajectorylengthMALT(
        adam, om, initial_trajectorylength, initial_trajectorylength, maxleapfrogsteps
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthSNAPER, m, positions, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        q = qs[:, chain]
        tmp = centered_dot(q, mq, r)
        dsq = tmp ^ 2 - centered_dot(positions[:, chain], mθ, r) ^ 2
        fd = 2 * tmp * dsq * dot(ps[:, chain], r)
        fd2 = 2 * tmp * -dsq * dot(ps[:, chain], -r)
        ghats[chain] = 2 * (fd + fd2) - dsq^2 / t
        ghats[chain] *= h
    end
    return ghats
end

struct TrajectorylengthLDG{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    om::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthLDG(
    initial_trajectorylength::AbstractVector{T}, dims; maxleapfrogsteps=1000, kwargs...
) where {T}
    adam = Adam(1, T; kwargs...)
    om = OnlineMoments(T, dims, 1)
    return TrajectorylengthMALT(
        adam, om, initial_trajectorylength, initial_trajectorylength, maxleapfrogsteps
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthLDG, m, positions, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        q = qs[:, chain]
        ldq, gradient = ldg(q)
        ldp, gradient = ldg(positions[:, chain])
        dsq = ldq ^ 2 - ldp ^ 2
        ghats[chain] = 4 * dsq * dot(gradient, ps[:, chain]) - dsq^2 / t
        ghats[chain] *= h
    end
    return ghats
end

struct TrajectorylengthConstant{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
end

function TrajectorylengthConstant(initial_trajectorylength::AbstractVector; kwargs...)
    return TrajectorylengthConstant(initial_trajectorylength, initial_trajectorylength)
end

function update!(tlc::TrajectorylengthConstant, args...; kwargs...) end

function reset!(tlc::TrajectorylengthConstant, args...; kwargs...) end
