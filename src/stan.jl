abstract type AbstractStan{T} <: AbstractSampler{T} end

struct Stan{T} <: AbstractStan{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    dims::Int
    chains::Int
    maxtreedepth::Int
    maxdeltaH::T
end

"""
    Stan(dims, chains, T = Float64; kwargs...)

Initialize Stan sampler object.  The number of dimensions `dims` and number of
chains `chains` are the only required arguments.  The type `T` of the ...

Optionally, via keyword arguments, can set the metric, stepsize, seed, maxtreedepth, and maxdeltaH.
"""
function Stan(
    dims,
    chains=4,
    T=Float64;
    metric=ones(T, dims, chains),
    stepsize=ones(T, chains),
    maxtreedepth=10,
    maxdeltaH=convert(T, 1000),
)
    D = convert(Int, dims)::Int
    return Stan(metric, stepsize, D, chains, maxtreedepth, maxdeltaH)
end

"""
    sample!(sampler::Stan, ldg)

Sample with Stan sampler object.  User must provide a function `ldg(position;
kwargs...)` which accepts `position::Vector` and returns a tuple containing
the evaluation of the joint log density function and a vector of the gradient,
each evaluated at the argument `position`.  The remaining keyword arguments
attempt to replicate [Stan](https://mc-stan.org/) defaults.
"""
function sample!(
    sampler::Stan{T},
    ldg;
    iterations=1000,
    warmup=1000,
    rngs=Random.Xoshiro.(rand(1:typemax(Int), sampler.chains)),
    draws_initializer=:stan,
    stepsize_adapter=StepsizeDualAverage(sampler.stepsize),
    trajectorylength_adapter=TrajectorylengthConstant(zeros(sampler.chains)),
    metric_adapter=MetricOnlineMoments(sampler.metric),
    adaptation_schedule=WindowedAdaptationSchedule(warmup),
    kwargs...,
) where {T<:AbstractFloat}
    M = iterations + warmup
    draws = Array{T,3}(undef, M + 1, sampler.dims, sampler.chains)
    diagnostics = trace(sampler, M + 1)

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
    return draws, diagnostics, rngs
end

function transition!(sampler::Stan, m, ldg, draws, rngs, trace; kwargs...)
    for chain in axes(draws, 3) # TODO multi-thread-able
        @views info = stan_kernel!(
            draws[m, :, chain],
            draws[m + 1, :, chain],
            rngs[chain],
            sampler.dims,
            sampler.metric[:, chain],
            sampler.stepsize[chain],
            sampler.maxdeltaH,
            sampler.maxtreedepth,
            ldg;
            kwargs...,
        )
        record!(trace, info, m + 1, chain)
    end
end

function stancriterion(pbeg, pend, rho)
    return dot(pbeg, rho) > 0 && dot(pend, rho) > 0
end

function stan_kernel!(
    position,
    position_next,
    rng,
    dims,
    metric,
    stepsize,
    maxdeltaH,
    maxtreedepth,
    ldg;
    kwargs...,
)
    T = eltype(position)
    z = PSPoint(position, randn(rng, T, dims))
    ld, gradient = ldg(z.position; kwargs...)
    H0 = hamiltonian(ld, z.momentum)

    zf = copy(z)
    zb = copy(z)
    zsample = copy(z)
    zpr = copy(z)

    # Momentum and sharp momentum at forward end of forward subtree
    pff = copy(z.momentum)
    psharpff = z.momentum

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
        depth += 1

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

    ld, gradient = ldg(zsample.position; kwargs...)
    position_next .= zsample.position

    return (;
        accepted,
        divergence,
        stepsize,
        energy=hamiltonian(ld, zsample.momentum),
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
            direction .* stepsize .* sqrt.(metric),
            1;
            kwargs...,
        )

        nleapfrog += 1
        zpropose .= z

        H = hamiltonian(ld, z.momentum)
        isnan(H) && (H = typemax(T))
        divergent = (H - H0) > maxdeltaH

        Δ = H0 - H
        logsumweight = logsumexp(logsumweight, Δ)
        α += Δ > zero(Δ) ? one(Δ) : exp(Δ)

        psharpbeg .= z.momentum
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
        depth - 1,
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
        depth - 1,
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
        return validfinal, nleapfrog, lswinit, α
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
        accept_stats = trace.acceptstat[m, :]
        update!(stepsize_adapter, accept_stats; warmup, kwargs...)
        set_stepsize!(sampler, stepsize_adapter; kwargs...)

        # TODO(ear) this is attempting to plan ahead;
        # to actually use update!() will require
        # more arguments, for additional information on which
        # the trajectorylength could be learned; re SGA methods
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
            reset!(stepsize_adapter; kwargs...)

            set_metric!(sampler, metric_adapter; kwargs...)
            reset!(metric_adapter)

            calculate_nextwindow!(schedule)
        end
    else
        set_stepsize!(sampler, stepsize_adapter; smoothed=true, kwargs...)
    end
end
