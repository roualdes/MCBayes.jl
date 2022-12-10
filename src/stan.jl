abstract type AbstractSampler{T <: AbstractFloat} end
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

function Stan(dims,
              chains = 1,
              T = Float64;
              metric = ones(T, dims, chains),
              stepsize = ones(T, chains),
              seed = [1:chains;],
              maxtreedepth = 10,
              maxdeltaH = convert(T, 1000))
    return Stan(metric,
                stepsize,
                seed,
                dims,
                chains,
                maxtreedepth,
                maxdeltaH)
end

function sample(sampler::AbstractSampler{T}, ldg;
                iterations = 2000,
                warmup = div(iterations, 2),
                rng = Random.Xoshiro.(sampler.seed),
                draws_initializer = :stan,
                stepsize_adapter = DualAverage(sampler.chains),
                trajectory_lengthadapter = (; initializer = :stan),
                metric_adapter = OnlineMoments(T, sampler.dims, sampler.chains),
                adaptation_schedule = WindowedAdaptationSchedule(warmup),
                integrator = :leapfrog,
                trace = (; acceptstat = zeros(T, iterations + warmup, sampler.chains)), # TODO doc expects sizes as iterations by chains
                kwargs...) where {T <: AbstractFloat}
    M = iterations + warmup
    draws = Array{T, 3}(undef, M, sampler.dims, sampler.chains)
    momenta = randn(T, sampler.dims, sampler.chains) .* metric(metric_adapter)
    acceptance_probabilities = rand(sampler.chains)
    # TODO double check argument order for initialize_* methods
    initialize_draws!(draws_initializer,
                      draws,
                      rng,
                      ldg;
                      kwargs...)

    initialize_stepsize!(stepsize_adapter,
                         metric(metric_adapter),
                         rng,
                         ldg,
                         draws;
                         integrator,
                         kwargs...)
    set_stepsize!(sampler, stepsize_adapter; kwargs...)

    # for m in 1:M
    #     info = transition!(sampler,
    #                        m,
    #                        ldg,
    #                        draws,
    #                        rng,
    #                        momenta,
    #                        metric(metric_adapter),
    #                        stepsize(stepsize_adapter),
    #                        acceptance_probabilities;
    #                        kwargs...)
    #     adapt!(sampler,
    #            adaptation_schedule,
    #            m,
    #            draws,
    #            metric_adapter,
    #            stepsize_adapter,
    #            trajectorylength_adapter;
    #            info...,
    #            kwargs...)
    #     # update_trace(trace, m, info)
    # end
    # return draws, sampler, diagnostics
end

function transition!(sampler::Stan, m, ldg, draws, rng, momenta, acceptance_probabilities; kwargs...)
    for chain in axes(draws, 3)
        metric = sampler.metric[:, chain]
        stepsize = sampler.stepsize[chain]
        # TODO double check arguments and their order
        stan_kernel!(draws[m, :, chain], draws[m+1, :, chain], rng, sampler.dims, metric, stepsize, sampler.maxdeltaH, sampler.maxtreedepth)
        # TODO copy stankernel from previous efforts and double return values
        # and note how update_trace wants things, iterations by chains => collect returned info from stan_kernel! here
    end
end


function _stankernel!(position, position_next, rng, dims, metric, stepsize, maxdeltaH, maxtreedepth, ldg)
    T = eltype(position)
    z = PhasePoint(position, rand_momentum(rng, dims, metric))
    ld, gradient = ldg(z.position)
    H0 = hamiltonian(ld, z.momentum, metric)

    zf = copy(z)
    zb = copy(z)
    zsample = copy(z)
    zpr = copy(z)

    # Momentum and sharp momentum at forward end of forward subtree
    pff = copy(z.momentum)
    psharpff = z.p .* metric

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

            z.q .= zf.q
            z.p .= zf.p

            validsubtree, nleapfrog, lswsubtree, α  =
                buildtree!(depth, z, zpr, M, rng,
                           psharpfb, psharpff, rhof, pfb, pff,
                           H0, ε, maxdeltaH, lp,
                           nleapfrog, lswsubtree, α)

            zf.q .= z.q
            zf.p .= z.p
        else
            rhof .= rho
            pfb .= pbb
            psharpfb .= psharpbb

            z.q .= zb.q
            z.p .= zb.p

            validsubtree, nleapfrog, lswsubtree, α =
                buildtree!(depth, z, zpr, M, rng,
                           psharpbf, psharpbb, rhob, pbf, pbb,
                           H0, -ε, maxdeltaH, lp,
                           nleapfrog, lswsubtree, α)

            zb.q .= z.q
            zb.p .= z.p
        end

        if !validsubtree
            divergence = true
            break
        end
        depth += one(depth)

        if lswsubtree > lsw
            zsample.q .= zpr.q
            zsample.p .= zpr.p
            accepted = true
        else
            if rand(rng, T) < exp(lswsubtree - lsw)
                zsample.q .= zpr.q
                zsample.p .= zpr.p
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


    θ2 .= zsample.q
    val_grad!(lp, θ2)
    return (accepted = accepted,
            divergence = divergence,
            energy = hamiltonian(val(lp), zsample.p, M),
            stepsize = ε,
            acceptstat = α / nleapfrog,
            treedepth = depth,
            leapfrog = nleapfrog)
end


function adapt!(sampler,
                schedule::WindowedAdaptationSchedule,
                i, ldg, draws, gradients, rng,
                metric_adapter, stepsize_adapter, trajectorylength_adapter; kwargs...)
    warmup = schedule.warmup
    if i <= warmup
        update!(stepsize_adapter, kwargs[:acceptstat]; warmup, kwargs...)
        set_stepsize!(sampler, stepsize_adapter; kwargs...)

        update!(trajectorylength_adapter; kwargs...)
        set_trajectorylength!(sampler, trajectorylength_adapter; kwargs...)

        if schedule.firstwindow <= i <= schedule.lastwindow
            @views update!(metric_adapter, draws[i, :, :]; kwargs...)
        end

        if i == schedule.closewindow
            initialize_stepsize!(stepsize_adapter, sampler.metric, rng, ldg, draws, gradients; kwargs...)
            set_stepsize!(sampler, stepsize_adapter; kwargs...)
            reset!(stepsize_adapter)

            set_metric!(sampler, metric_adapter; kwargs...)
            reset!(metric_adapter)
        end
    end
end

function set_stepsize!(sampler, adapter; kwargs...)
    sampler.stepsize .= optimum(adapter)
end

function set_trajectorylength!(sampler::Stan, adapter; kwargs...)
end

function set_metric!(sampler, adapter; kwargs...)
    sampler.metric .= metric(adapter; kwargs...)
end


function update_trace!(sampler::Stan, trace, m, info)
    keys = (
        :accepted,
        :divergence,
        :energy,
        :stepsize,
        :acceptstat,
        :treedepth,
        :leapfrog
    )
    for k in keys
        if haskey(info, k)
            # TODO expects sizes as iteration by chains
            trace[k][m, :] .= info[k]
        end
    end
end
