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

function sample(sampler::Stan{T}, ldg;
                iterations = 2000,
                warmup = div(iteration, 2),
                rng = Random.Xoshiro(sampler.seed),
                draws_initializer = :stan,
                stepsize_adapter = DualAverage(sampler.chains),
                trajectory_lengthadapter = (; initializer = :stan),
                metric_adapter = OnlineMoments(T, sampler.dims, sampler.chains),
                adaptation_schedule = WindowedAdaptationSchedule(warmup),
                integrator = :leapfrog,
                trace = (; acceptstat = zeros(T, iterations + warmup, sampler.chains)); # TODO doc expects sizes as iterations by chains
                kwargs...) where {T <: AbstractFloat}
    M = iterations + warmup
    draws = Array{T, 3}(undef, M, sampler.dims, sampler.chains)
    gradients = Matrix{T}(undef, sampler.dims, sampler.chains)
    momenta = randn!(similar(gradients))
    acceptance_probabilities = rand(sampler.chains)
    # TODO double check argument order for initialize_* methods
    initialize_draws!(draws_initializer,
                      rng,
                      ldg,
                      draws,
                      gradient;
                      kwargs...)

    initialize_stepsize!(stepsize_adapter,
                         metric(metric_adapter),
                         rng,
                         ldg,
                         draws,
                         gradient;
                         integrator,
                         kwargs...)

    for m in 1:M
        info = transition!(sampler,
                           m,
                           ldg,
                           draws,
                           rng,
                           momenta,
                           metric(metric_adapter),
                           stepsize(stepsize_adapter),
                           acceptance_probabilities;
                           kwargs...)
        adapt!(sampler,
               adaptation_schedule,
               metric_adapter,
               stepsize_adapter,
               trajectorylength_adapter;
               info...,
               kwargs...)
        # update_trace(trace, m, info)
    end
    return draws, sampler, diagnostics
end

function transition!(sampler::Stan, m, ldg, draws, rng, momenta, acceptance_probabilities; kwargs...)
    for chain in axes(draws, 3)
        metric = sampler.metric[:, chain]
        stepsize = sampler.stepsize[:, chain]
        # TODO double check arguments and their order
        @views stan_kernel!(draws[m, :, chain], draws[m+1, :, chain], rng, sampler.dims, metric, stepsize, sampler.maxdeltaH, sampler.maxtreedepth)
        # TODO copy stankernel from previous efforts and double return values
        # and note how update_trace wants things, iterations by chains => collect returned info from stan_kernel! here
    end

end

function adapt_stepsize!(sampler::Stan, schedule, stepsize_adapter, i, draws; kwargs...)
    update!(stepsize_adapeter, kwargs[:acceptstat]; kwargs...)
end

function set_stepsize!(sampler::Stan, schedule, stepsize_adapter, i, draws; kwargs...)
    sampler.stepsize .= stepsize(stepsize_adapter; kwargs...)
end

function adapt_metric!(sampler::Stan, schedule::WindowedAdaptationSchedule, metric_adapter::OnlineMoments, i, draws; kwargs...)
    if schedule.firstwindow <= i <= schedule.lastwindow
        update!(metric_adapter, draws[i, :, :]; kwargs...)
    end
end

function set_metric!(sampler::Stan, schedule, metric_adapter, i, draws; kwargs...)
    sampler.metric .= metric(metric_adapt; kwargs...)
end

function adapt_trajectorylength!(sampler::Stan, schedule, trajectorylength_adapter, i, draws; kwargs...)
end

function set_trajectorylength!(sampler::Stan, schedule, trajectorylength_adapter, i, draws; kwargs...)
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
