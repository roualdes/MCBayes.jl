abstract type AbstractStan{T} <: AbstractSampler{T} end

mutable struct Stan{T <: AbstractFloat} <: AbstractStan
    metric::Matrix{T}
    om::OnlineMoments{T}
    dualaverage_ε::DualAverage{T}
    const dims::Int
    const chains::Int
    const maxtreedepth::Int
    const maxdeltaH::T
end

function Stan(dims, chains, T = Float64;
              metric = ones(T, dims, chains),
              ws = OnlineMoments(T, dims, chains),
              daulaverage_ε = DualAverage(),
              maxtreedepth = 10,
              maxdeltaH = 1000)
    return Stan(metric,
                ws,
                dualaverage_ε,
                dims,
                chains,
                maxtreedepth,
                maxdeltaH)
end

function sample(stan::AbstractStan{T}, ldg;
                iterations = 2000,
                warmup = div(iteration, 2),
                rng = Random.Xoshiro(rand(1:typemax(Int))),
                wa = WindowedAdapter(warmup),
                diagnostics = (;); # TODO could probably use somesort of
                # helper function
                kwargs...) where {T <: AbstractFloat}
    M = iterations + warmup
    draws = Array{T, 3}(undef, M, stan.chains, stan.dims)
    initialize_draws!(stan, ldg, draws; kwargs...)
    initialize_stepsize!(stan, ldg, draws; kwargs...)
    for m in 1:M
        info = transition!(stan, m, ldg, draws;
                           warmup, diagnostics, kwargs...)
        # adapt!(...)
        # update(diagnostics, m, info)
    end
    return draws, stan, diagnostics
end

function transition!(stan::AbstractStan, i, ldg, draws; kwargs...)
    info = kernel!(stan, i, draws)
    update_info!(get(kwargs, :diagnostics, (;)), i, info)
    adapt_chains!(stan, i, ldg, draws; acceptstat = info.acceptstat, kwargs...)
end

# TODO function kernel!(...)

function update_info!(dstore, i, info)
    diagnostics = (
        :accepted,
        :divergence,
        :energy,
        :stepsize,
        :acceptstat,
        :treedepth,
        :leapfrog
    )
    for d in diagnostics
        if haskey(info, d)
            dstore.accepted[i] = info[d]
        end
    end
end

# TODO what would it look like for adapt_chains to specialize both
# a sampler and a LearningSchedule(Windowed, Exponential, Linear, Polynomial, ...)
# BUT I really want each adaptation parameter to be able to be adapted under different schedules
# the adaptation schedule should be part of the adapter(dualaverage, Adam, Adamw)



function adapt_chains!(stan::AbstractStan, i, ldg, draws; kwargs...)
    winowed_adapter = get(kwargs, :wa, WindowedAdapter(0))
    warmup = get(kwargs, :warmup, 0)

    if i <= warmup
        adapt_stepsize!(stan, i, draws; kwargs...)
        adapt_metric!(stan, i, draws; kwargs...)

        if i == stan.owa.closewindow
            reset!(stan.dualaverage_ε)
            reset!(stan.om)
            calculate_nextwindow!(stan.owa)
        end
    else
        stan.dualaverage_ε.ε = stan.dualaverage_ε.εbar
    end
end

function adapt_stepsize!(stan::AbstractStan, i, draws; kwargs...)
    update!(stan.dualaverage_ε, i, kwargs[:acceptstat])
end

function adapt_metric!(stan::AbstractStan, i, draws; kwargs...)
    if stan.owa.firstwindow <= i <= stan.owa.lastwindow
        update!(stan.om, draws[i, :, :]; kwargs...)
    end

    if i == stan.owa.closewindow
        w = stan.om.n / (stan.om.n + 5)
        for chain in 1:stan.chains
            stan.metric[:, chain] .= w .* stan.om.v[:, chain] .+ (1 - w) * 1e-3
        end
    end
end
