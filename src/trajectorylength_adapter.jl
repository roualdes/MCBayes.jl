# Adapted from
# [1] https://arxiv.org/pdf/2110.11576.pdf
# [2] https://proceedings.mlr.press/v130/hoffman21a.html
# [3] https://github.com/tensorflow/probability/blob/c678caa1b8e94ab3677a37e581e8b19e68e59248/tensorflow_probability/python/experimental/mcmc/gradient_based_trajectory_length_adaptation.py
# [4] https://github.com/tensorflow/probability/blob/5ebcdf1f32ecc340dece4f21694790ea95c6c1e2/spinoffs/fun_mc/fun_mc/sga_hmc.py
# [5] https://github.com/tensorflow/probability/blob/5ebcdf1f32ecc340dece4f21694790ea95c6c1e2/spinoffs/fun_mc/fun_mc/fun_mc_lib.py
# [6] https://arxiv.org/pdf/2210.12200.pdf
# [7] https://github.com/tensorflow/probability/blob/main/discussion/adaptive_malt/adaptive_malt.py#L537

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
    previousps,
    ps,
    qs,
    stepsize,
    r,
    ldg,
    args...;
    γ=-0.6, # TODO replace by aw below, or needs to be renamed
    kwargs...,
)
    update!(tla.omstates, positions; kwargs...)
    update!(tla.omproposals, qs; kwargs...)
    ghats = trajectorylength_gradient(
        tla, m, αs, positions, previousps, ps, qs, stepsize, r, ldg
    )

    for (i, (ai, gi)) in enumerate(zip(αs, ghats)) # [7]#L214
        if !isfinite(gi) || ai < 1e-4
            αs[i] = 1e-20
        end
    end

    ghat = weighted_mean(ghats, αs)
    as = update!(tla.adam, ghat, m; kwargs...)[1]

    logupdate = clamp(as, -0.35, 0.35)           # [3]#L759
    T = tla.trajectorylength[1] * exp(logupdate) # [3]#L761
    T = min(T, stepsize * tla.maxleapfrogsteps)  # [3]#L773

    tla.trajectorylength[1] = T
    aw = 1 - 8 / 9                # TODO add this as an argument to the constructor
    # TODO and then re-design like OnlineMoments
    tla.trajectorylength_bar[1] = exp(
        aw * log(T) + (1 - aw) * log(1e-10 + tla.trajectorylength_bar[1])
    )
end

function trajectorylength_gradient(
    tla::AbstractTrajectorylengthAdapter,
    m,
    αs,
    positions,
    previousps,
    ps,
    qs,
    stepsize,
    r,
    ldg,
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

    N = tla.omstates.n[1]
    mw = N / (N + chains)

    @. meanθ = mw * tla.omstates.m + (1 - mw) * meanθ
    @. meanq = mw * tla.omproposals.m + (1 - mw) * meanq

    ghats = sampler_trajectorylength_gradient(
        tla, m, positions, previousps, ps, qs, stepsize, meanθ, meanq, r, ldg
    )
    return ghats
end

struct TrajectorylengthChEES{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    omstates::OnlineMoments{T}
    omproposals::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthChEES(
    initial_trajectorylength::AbstractVector{T},
    dims,
    warmup;
    maxleapfrogsteps=1000,
    kwargs...,
) where {T}
    adam = Adam(1, warmup, T; α=0.01, kwargs...)
    omstates = OnlineMoments(T, dims, 1)
    omproposals = OnlineMoments(T, dims, 1)
    return TrajectorylengthChEES(
        adam,
        omstates,
        omproposals,
        initial_trajectorylength,
        initial_trajectorylength,
        maxleapfrogsteps,
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthChEES, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        q = qs[:, chain]
        dsq = centered_sum(abs2, q, mq) - centered_sum(abs2, positions[:, chain], mθ)
        fd = dsq * centered_dot(q, mq, ps[:, chain])
        fd2 = -dsq * centered_dot(positions[:, chain], mθ, -previousps[:, chain])
        ghats[chain] = 2 * (fd + fd2) - dsq^2 / t
        ghats[chain] *= h
    end
    return ghats
end

struct TrajectorylengthMALT{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    omstates::OnlineMoments{T}
    omproposals::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthMALT(
    initial_trajectorylength::AbstractVector{T},
    dims,
    warmup;
    maxleapfrogsteps=1000,
    kwargs...,
) where {T}
    adam = Adam(1, warmup, T; α=0.01, kwargs...)
    omstates = OnlineMoments(T, dims, 1)
    omproposals = OnlineMoments(T, dims, 1)
    return TrajectorylengthMALT(
        adam,
        omstates,
        omproposals,
        initial_trajectorylength,
        initial_trajectorylength,
        maxleapfrogsteps,
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthMALT, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        q = qs[:, chain]
        tmp = centered_dot(q, mq, r)
        dsq = tmp^2 - centered_dot(positions[:, chain], mθ, r)^2
        fd = dsq * tmp * dot(ps[:, chain], r)
        tmp2 = centered_dot(positions[:, chain], mθ, r)
        fd2 = -dsq * tmp2 * dot(-previousps[:, chain], r)
        ghats[chain] = 2 * (fd + fd2) - dsq^2 / t
    end
    return ghats
end

struct TrajectorylengthSNAPER{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    omstates::OnlineMoments{T}
    omproposals::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

function TrajectorylengthSNAPER(
    initial_trajectorylength::AbstractVector{T},
    dims,
    warmup;
    maxleapfrogsteps=1000,
    kwargs...,
) where {T}
    adam = Adam(1, warmup, T; α=0.01, kwargs...)
    omstates = OnlineMoments(T, dims, 1)
    omproposals = OnlineMoments(T, dims, 1)
    return TrajectorylengthMALT(
        adam,
        omstates,
        omproposals,
        initial_trajectorylength,
        initial_trajectorylength,
        maxleapfrogsteps,
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthSNAPER, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        q = qs[:, chain]
        tmp = centered_dot(q, mq, r)
        dsq = tmp^2 - centered_dot(positions[:, chain], mθ, r)^2
        fd = dsq * tmp * dot(ps[:, chain], r)
        tmp2 = centered_dot(positions[:, chain], mθ, r)
        fd2 = -dsq * tmp2 * dot(-previousps[:, chain], r)
        ghats[chain] = 2 * (fd + fd2) - dsq^2 / t
        ghats[chain] *= h
    end
    return ghats
end

struct TrajectorylengthLDG{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    omstates::OnlineMoments{T}
    omproposals::OnlineMoments{T}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
end

# TODO enable trajectorylength_adam_schedule, separate from stepsize_adam_schedule
function TrajectorylengthLDG(
    initial_trajectorylength::AbstractVector{T},
    dims,
    warmup;
    maxleapfrogsteps=1000,
    kwargs...,
) where {T}
    adam = Adam(1, warmup, T; kwargs...)
    omstates = OnlineMoments(T, dims, 1)
    omproposals = OnlineMoments(T, dims, 1)
    return TrajectorylengthMALT(
        adam,
        omstates,
        omproposals,
        initial_trajectorylength,
        initial_trajectorylength,
        maxleapfrogsteps,
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthLDG, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg
)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        q = qs[:, chain]
        ldq, gradientq = ldg(q)
        ldp, gradientpos = ldg(positions[:, chain])
        dsq = ldq^2 - ldp^2
        fd = dsq * dot(gradientq, ps[:, chain])
        fd2 = -dsq * dot(gradientpos, previousps[:, chain])
        ghats[chain] = 2 * (fd + fd2) - dsq^2 / t
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
