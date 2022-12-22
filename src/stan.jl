abstract type AbstractSampler{T<:AbstractFloat} end
abstract type AbstractStan{T} <: AbstractSampler{T} end

struct Stan{T} <: AbstractStan{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    seed::Vector{Int}
    dims::Int
    chains::Int
    maxtreedepth::Int
    maxdeltaH::T
end

function Stan(
    dims,
    chains=1,
    T=Float64;
    metric=ones(T, dims, chains),
    stepsize=ones(T, chains),
    seed=rand(1:typemax(Int), chains),
    maxtreedepth=10,
    maxdeltaH=convert(T, 1000),
)
    return Stan(metric, stepsize, seed, dims, chains, maxtreedepth, maxdeltaH)
end

function sample!(
    sampler::AbstractSampler{T},
    ldg;
    iterations=2000,
    warmup=div(iterations, 2),
    rngs=Random.Xoshiro.(sampler.seed),
    draws_initializer=:stan,
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize),
    trajectorylength_adapter=TrajectorylengthConstant(zeros(sampler.chains)),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    adaptation_schedule=WindowedAdaptationSchedule(warmup),
    kwargs...,
) where {T<:AbstractFloat}
    M = iterations + warmup
    draws = Array{T,3}(undef, M + 1, sampler.dims, sampler.chains)
    momenta = randn(T, sampler.dims, sampler.chains) .* sampler.metric
    acceptance_probabilities = rand(sampler.chains)
    diagnostics = trace(sampler, M)

    initialize_draws!(draws_initializer, draws, rngs, ldg; kwargs...)

    @views initialize_stepsize!(
        stepsize_adapter, sampler.metric, rngs, ldg, draws[1, :, :]; kwargs...
    )
    set_stepsize!(sampler, stepsize_adapter; kwargs...)

    for m in 1:M
        transition!(sampler, m, ldg, draws, rngs, diagnostics; kwargs...)

        adapt!(
            sampler,
            adaptation_schedule,
            diagnostics,
            m,
            ldg,
            draws,
            rngs,
            metric_adapter,
            stepsize_adapter,
            trajectorylength_adapter;
            kwargs...,
        )
    end
    return draws, diagnostics
end

function transition!(sampler::Stan, m, ldg, draws, rngs, trace; kwargs...)
    for chain in axes(draws, 3) # TODO multi-thread-able
        @views metric = sampler.metric[:, chain]
        stepsize = sampler.stepsize[chain]
        @views info = stan_kernel!(
            draws[m, :, chain],
            rngs[chain],
            sampler.dims,
            metric,
            stepsize,
            sampler.maxdeltaH,
            sampler.maxtreedepth,
            ldg;
            kwargs...,
        )
        draws[m + 1, :, chain] = info.position_next
        record!(trace, info, m, chain)
    end
end

function stan_kernel!(
    position, rng, dims, metric, stepsize, maxdeltaH, maxtreedepth, ldg; kwargs...
)
    T = eltype(position)
    z = PSPoint(position, rand_momentum(rng, dims, metric))
    ld, gradient = ldg(z.position; kwargs...)
    H0 = hamiltonian(ld, z.momentum, metric)

    zf = copy(z)
    zb = copy(z)
    zsample = copy(z)
    zpr = copy(z)

    # Momentum and sharp momentum at forward end of forward subtree
    pff = copy(z.momentum)
    psharpff = z.momentum .* metric

    # Momentum and sharp momentum at backward end of forward subtree
    pfb = copy(pff)
    psharpfb = copy(psharpff)

    # Momentum and sharp momentum at forward end of backward subtree
    pbf = copy(pff)
    psharpbf = copy(psharpff)

    # Momentum and sharp momentum at backward end of backward subtree
    pbb = copy(pff)
    psharpbb = copy(psharpff)

    # Integrated momenta along trajectory
    rho = copy(pff)

    α = zero(T)
    lsw = zero(T)
    depth = zero(maxtreedepth)
    nleapfrog = zero(Int)

    divergence = zero(Bool)
    accepted = zero(Bool)

    while depth < maxtreedepth
        rhof = zero(rho)
        rhob = zero(rho)
        lswsubtree = typemin(T)

        if rand(rng, Bool)
            rhob .= rho
            pbf .= pff
            psharpbf .= psharpff
            z .= zf

            validsubtree, nleapfrog, lswsubtree, α = buildtree!(
                depth,
                z,
                zpr,
                metric,
                rng,
                psharpfb,
                psharpff,
                rhof,
                pfb,
                pff,
                H0,
                1,
                stepsize,
                maxdeltaH,
                ldg,
                nleapfrog,
                lswsubtree,
                α;
                kwargs...,
            )
            zf .= z
        else
            rhof .= rho
            pfb .= pbb
            psharpfb .= psharpbb
            z .= zb

            validsubtree, nleapfrog, lswsubtree, α = buildtree!(
                depth,
                z,
                zpr,
                metric,
                rng,
                psharpbf,
                psharpbb,
                rhob,
                pbf,
                pbb,
                H0,
                -1,
                stepsize,
                maxdeltaH,
                ldg,
                nleapfrog,
                lswsubtree,
                α;
                kwargs...,
            )
            zb .= z
        end

        if !validsubtree
            divergence = true
            break
        end
        depth += one(depth)

        if lswsubtree > lsw
            zsample .= zpr
            accepted = true
        else
            if rand(rng, T) < exp(lswsubtree - lsw)
                zsample .= zpr
                accepted = true
            end
        end

        lsw = logsumexp(lsw, lswsubtree)

        # Demand satisfication around merged subtrees
        @. rho = rhob + rhof
        persist = stancriterion(psharpbb, psharpff, rho)

        # Demand satisfaction between subtrees
        rhoextended = rhob + pfb
        persist &= stancriterion(psharpbb, psharpfb, rhoextended)

        @. rhoextended = rhof + pbf
        persist &= stancriterion(psharpbf, psharpff, rhoextended)

        !persist && break
    end # end while

    position_next = zsample.position
    ld, gradient = ldg(position_next; kwargs...)

    return (;
        position_next,
        accepted,
        divergence,
        stepsize,
        energy=hamiltonian(ld, zsample.momentum, metric),
        acceptstat=α / nleapfrog,
        treedepth=depth,
        leapfrog=nleapfrog,
    )
end

function buildtree!(
    depth,
    z,
    zpropose,
    metric,
    rng,
    psharpbeg,
    psharpend,
    rho,
    pbeg,
    pend,
    H0,
    direction,
    stepsize,
    maxdeltaH,
    ldg,
    nleapfrog,
    logsumweight,
    α;
    kwargs...,
)
    T = eltype(z.position)
    if iszero(depth)
        ld, gradient = ldg(z.position; kwargs...)
        ld, gradient = leapfrog!(
            z.position,
            z.momentum,
            ldg,
            gradient,
            direction .* stepsize .* metric,
            1;
            kwargs...,
        )

        nleapfrog += 1
        zpropose .= z

        H = hamiltonian(ld, z.momentum, metric)
        isnan(H) && (H = typemin(T))
        divergent = divergence(H0, H, maxdeltaH)

        Δ = H - H0
        logsumweight = logsumexp(logsumweight, Δ)
        α += Δ > zero(Δ) ? one(Δ) : exp(Δ)

        @. psharpbeg = z.momentum * metric
        psharpend .= psharpbeg

        rho .+= z.momentum
        pbeg .= z.momentum
        pend .= z.momentum

        return !divergent, nleapfrog, logsumweight, α
    end

    lswinit = typemin(T)

    pinitend = similar(z.momentum)
    psharpinitend = similar(z.momentum)
    rhoinit = zero(rho)

    validinit, nleapfrog, lswinit, α = buildtree!(
        depth - one(depth),
        z,
        zpropose,
        metric,
        rng,
        psharpbeg,
        psharpinitend,
        rhoinit,
        pbeg,
        pinitend,
        H0,
        direction,
        stepsize,
        maxdeltaH,
        ldg,
        nleapfrog,
        lswinit,
        α;
        kwargs...,
    )

    if !validinit
        return validinit, nleapfrog, logsumweight, α
    end

    zfinalpr = copy(z)
    lswfinal = typemin(T)

    psharpfinalbeg = similar(z.momentum)
    pfinalbeg = similar(z.momentum)
    rhofinal = zero(rho)

    validfinal, nleapfrog, lswfinal, α = buildtree!(
        depth - one(depth),
        z,
        zfinalpr,
        metric,
        rng,
        psharpfinalbeg,
        psharpend,
        rhofinal,
        pfinalbeg,
        pend,
        H0,
        direction,
        stepsize,
        maxdeltaH,
        ldg,
        nleapfrog,
        lswfinal,
        α;
        kwargs...,
    )

    if !validfinal
        return validfinal, nleapfrog, logsumweight, α
    end

    lswsubtree = logsumexp(lswinit, lswfinal)

    if lswfinal > lswsubtree
        zpropose .= zfinalpr
    else
        if rand(rng, T) < exp(lswfinal - lswsubtree)
            zpropose .= zfinalpr
        end
    end

    logsumweight = logsumexp(logsumweight, lswsubtree)

    rhosubtree = rhoinit + rhofinal
    rho .+= rhosubtree

    # Demand satisfaction around merged subtrees
    persist = stancriterion(psharpbeg, psharpend, rhosubtree)

    # Demand satisfaction between subtrees
    @. rhosubtree = rhoinit + pfinalbeg
    persist &= stancriterion(psharpbeg, psharpfinalbeg, rhosubtree)

    @. rhosubtree = rhofinal + pinitend
    persist &= stancriterion(psharpinitend, psharpend, rhosubtree)

    return persist, nleapfrog, logsumweight, α
end

function adapt!(
    sampler,
    schedule::WindowedAdaptationSchedule,
    trace,
    m,
    ldg,
    draws,
    rngs,
    metric_adapter,
    stepsize_adapter,
    trajectorylength_adapter;
    kwargs...,
)
    warmup = schedule.warmup
    if m <= warmup
        @views accept_stats = trace.acceptstat[m, :]
        update!(stepsize_adapter, accept_stats; warmup, kwargs...)
        set_stepsize!(sampler, stepsize_adapter; kwargs...)

        # TODO(ear) this is attempting to plan ahead;
        # to actually use update!() will require
        # more arguments, for additional information on which
        # the trajectorylength could be learned
        update!(trajectorylength_adapter; kwargs...)
        set_trajectorylength!(sampler, trajectorylength_adapter; kwargs...)

        if schedule.firstwindow <= m <= schedule.lastwindow
            @views update!(metric_adapter, draws[m + 1, :, :]; kwargs...)
        end

        if m == schedule.closewindow
            @views initialize_stepsize!(
                stepsize_adapter,
                optimum(metric_adapter),
                rngs,
                ldg,
                draws[m + 1, :, :];
                kwargs...,
            )
            set_stepsize!(sampler, stepsize_adapter; kwargs...)
            reset!(stepsize_adapter)

            set_metric!(sampler, metric_adapter; kwargs...)
            reset!(metric_adapter)

            calculate_nextwindow!(schedule)
        end
    else
        set_stepsize!(sampler, stepsize_adapter; kwargs...)
    end
end
