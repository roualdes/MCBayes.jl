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

function reset!(tla::AbstractTrajectorylengthAdapter, stepsize, decay_steps, args...; kwargs...)
    tla.trajectorylength[1] = stepsize
    tla.trajectorylength_bar[1] = stepsize
    reset!(tla.adam; decay_steps = decay_steps, kwargs...)
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
    ldg!,
    args...;
    kwargs...,
    )


    if any(isnan.(qs))
        println("proposed states $(qs)")
    end

    update!(tla.omstates, positions; kwargs...)
    update!(tla.omproposals, qs; kwargs...)

    ghats = trajectorylength_gradient(
        tla, m, αs, positions, previousps, ps, qs, stepsize, r, ldg!
    )

    for (i, (ai, gi)) in enumerate(zip(αs, ghats)) # [7]#L214
        if isnan(gi)
            ghats[i] = 0
        end
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
    ldg!,
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
        tla, m, positions, previousps, ps, qs, stepsize, meanθ, meanq, r, ldg!
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
    tlc::TrajectorylengthChEES, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg!
)
    t = tlc.trajectorylength[1] + stepsize
    h = halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    @views for chain in 1:chains
        if any(isnan.(ps[:, chain]))
            ghats[chain] = 1e-20
            continue
        end
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
    tlc::TrajectorylengthMALT, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg!
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
    return TrajectorylengthSNAPER(
        adam,
        omstates,
        omproposals,
        copy(initial_trajectorylength),
        copy(initial_trajectorylength),
        maxleapfrogsteps,
    )
end

function sampler_trajectorylength_gradient(
    tlc::TrajectorylengthSNAPER, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg!
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
    tlc::TrajectorylengthLDG, m, positions, previousps, ps, qs, stepsize, mθ, mq, r, ldg!
)
    t = tlc.trajectorylength[1] + stepsize
    h = 1 # halton(m)
    T = eltype(positions)
    dims, chains = size(positions)
    ghats = zeros(chains)
    gradientq = zeros(dims)
    gradientpos = zeros(dims)
    @views for chain in 1:chains
        q = qs[:, chain]
        ldq = ldg!(q, gradientq)
        ldp = ldg!(positions[:, chain], gradientpos)
        dsq = ldq^2 - ldp^2
        fd = dsq * dot(gradientq, ps[:, chain])
        fd2 = -dsq * dot(gradientpos, previousps[:, chain])
        ghats[chain] = 2 * (fd + fd2) - dsq^2 / t
        ghats[chain] *= h
    end
    return ghats
end


struct TrajectorylengthDualAverageLDG{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    da::DualAverage{T}
    trajectorylength::Vector{T}
end


function TrajectorylengthDualAverageLDG(
    stepsize::AbstractVector{T};
    kwargs...,
    ) where {T}
    return TrajectorylengthDualAverageLDG(
        DualAverage(1, T; kwargs...),
        copy(stepsize),
    )
end

function update!(
    tla::TrajectorylengthDualAverageLDG,
    m,
    αs,
    previouspositions,
    proposedpositions,
    proposedmomentum,
    stepsize,
    ldg!,
    args...;
    kwargs...,
)

    ghats = sampler_trajectorylength_gradient(
        tla, m, previouspositions, proposedpositions, proposedmomentum, stepsize, ldg!
    )

    # acceptance probability weighted mean of chains' gradients => ghat
    T = eltype(previouspositions)
    a = zero(T)
    ghat = zero(T)
    for (i, (ai, gi)) in enumerate(zip(αs, ghats)) # [7]#L214
        if !isfinite(gi) || ai < 1e-4
            ai = 1e-20
        end
        a += ai
        ghat += ai * (gi - ghat) / a
    end

    tla.trajectorylength .= update!(tla.da, ghat; kwargs...)
    # TODO necessary? prefer to skip: logupdate = clamp(as, -0.35, 0.35)           # [3]#L759
end

function sampler_trajectorylength_gradient(
    tla::TrajectorylengthDualAverageLDG, m, previouspositions, proposedpositions, proposedmomentum, stepsize, ldg!
    )

    T = eltype(previouspositions)
    dims, chains = size(previouspositions)

    τ = tla.trajectorylength[1] + mean(stepsize)
    h = halton(m)
    ghats = zeros(chains)
    gradient_previous = zeros(dims)
    gradient_proposed = zeros(dims)

    @views for chain in 1:chains
        previous = previouspositions[:, chain]
        ld_previous = ldg!(previous, gradient_previous)

        proposed = proposedpositions[:, chain]
        ld_proposed = ldg!(proposed, gradient_proposed)

        d = (ld_proposed ^ 2 - ld_previous ^ 2)
        dg = d * dot(gradient_proposed, proposedmomentum[:, chain])
        ghats[chain] = 2 * dg - d ^ 2 / τ
        ghats[chain] *= h
    end
    return ghats
end

function reset!(tla::TrajectorylengthDualAverageLDG, args...; kwargs...)
    reset!(tla.da; kwargs...)
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


struct AdamSNAPER{T<:AbstractFloat} <: AbstractTrajectorylengthAdapter{T}
    adam::Adam{T}
    mpositions::Vector{T}
    mproposals::Vector{T}
    N::Vector{Int}
    trajectorylength::Vector{T}
    trajectorylength_bar::Vector{T}
    maxleapfrogsteps::Int
    alpha::T
end

function AdamSNAPER(
    initial_trajectorylength::AbstractVector{T},
    dims,
    warmup;
    maxleapfrogsteps=1000,
    dualaverage_snaper_smooth_factor = 1 - 8/9,
    kwargs...,
) where {T}
    adam = Adam(1, warmup, T; μ = 0, kwargs...)
    return AdamSNAPER(
        adam,
        zeros(T, dims),
        zeros(T, dims),
        ones(Int, 1),
        copy(initial_trajectorylength),
        copy(initial_trajectorylength),
        maxleapfrogsteps,
        convert(T, dualaverage_snaper_smooth_factor)::T
    )
end

function update!(
    das::AdamSNAPER,
    m,
    αs,
    previous_positions,
    previous_momentum,
    proposed_momentum,
    proposed_positions,
    stepsize,
    pca,
    args...;
    kwargs...,
    )

    T = eltype(previous_positions)
    dims, chains = size(previous_positions)

    # v = zero(T)
    # mean_positions = zeros(T, dims)
    # mean_proposals = zeros(T, dims)

    # @views for chain in 1:chains
    #     @. mean_positions += (previous_positions[:, chain] - mean_positions) / chain
    #     a = αs[chain]
    #     v += a
    #     if !all(isnan.(proposed_positions[:, chain]))
    #         @. mean_proposals += a * (proposed_positions[:, chain] - mean_proposals) / v
    #     end
    # end

    # N = das.N[1]
    # mw = N / (N + chains)

    # @. das.mpositions = mw * das.mpositions + (1 - mw) * mean_positions
    # if !all(isnan.(mean_proposals))
    #     @. das.mproposals = mw * das.mproposals + (1 - mw) * mean_proposals
    # end


    ghats = sampler_trajectorylength_gradient(
        das,
        m,
        previous_positions,
        proposed_positions,
        previous_momentum,
        proposed_momentum,
        stepsize,
        pca
    )

    for (i, (ai, gi)) in enumerate(zip(αs, ghats)) # [7]#L214
        if isnan(gi) || !isfinite(gi)
            ghats[i] = 0
            αs[i] = 1e-20
        end
        # if !isfinite(gi) || ai < 1e-4
        #     αs[i] = 1e-20
        # end
    end

    ghat = weighted_mean(ghats, αs)
    as = update!(das.adam, ghat, m; kwargs...)[1]

    #logupdate = clamp(as, -0.35, 0.35)           # [3]#L759
    T = das.trajectorylength[1] * exp(as)             # [3]#L761
    T = min(T, stepsize * das.maxleapfrogsteps) # [3]#L773

    das.trajectorylength_bar[1] = exp(
        das.alpha * log(T) + (1 - das.alpha) * log(1e-10 + das.trajectorylength_bar[1])
    )

    maxtrajectorylength = stepsize * das.maxleapfrogsteps
    das.trajectorylength .= min.(T, maxtrajectorylength)
    das.trajectorylength_bar .= min.(das.trajectorylength_bar, maxtrajectorylength)
end

function sampler_trajectorylength_gradient(
    das::AdamSNAPER, m, previous_positions, proposed_positions, previous_momentum, proposed_momentum, stepsize, pca
    )
    τ = das.trajectorylength[1] + stepsize
    T = eltype(previous_positions)
    _, chains = size(previous_positions)
    # mq = das.mproposals
    # mθ = das.mpositions
    ghats = Vector{T}(undef, chains)
    @views for chain in 1:chains
        # CHEES
        # proposed = centered_sum(abs2, proposed_positions[:, chain], mq)
        # previous = centered_sum(abs2, previous_positions[:, chain], mθ)
        # dsq = proposed - previous
        # rho_proposed = dsq * centered_dot(proposed_positions[:, chain], mq, proposed_momentum[:, chain])
        # rho_previous = -dsq * centered_dot(previous_positions[:, chain], mq, -previous_momentum[:, chain])
        # SNAPER
        # proposed_pca = centered_dot(proposed_positions[:, chain], mq, pca)
        # previous_pca = centered_dot(previous_positions[:, chain], mθ, pca)
        proposed_pca = proposed_positions[:, chain]' * pca
        previous_pca = previous_positions[:, chain]' * pca
        dsq = proposed_pca ^ 2 - previous_pca ^ 2
        rho_proposed = dsq * proposed_pca * dot(proposed_momentum[:, chain], pca)
        rho_previous = -dsq * previous_pca * dot(-previous_momentum[:, chain], pca)
        ghats[chain] = 2 * (rho_proposed + rho_previous) / τ - (dsq / τ) ^ 2
    end
    return ghats
end

function reset!(das::AdamSNAPER, stepsize, args...; kwargs...)
    reset!(das.adam; kwargs...)
    das.trajectorylength[1] = stepsize
    das.trajectorylength_bar[1] = stepsize
    das.mpositions .= 0
    das.mproposals .= 0
    das.N .= 1
end
