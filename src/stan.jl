abstract type AbstractSampler{T <: AbstractFloat} end
abstract type AbstractStan{T} <: AbstractSampler{T} end

struct Stan{T} <: AbstractStan{T}
    metric::VecOrMat{T}
    stepsize::Vector{T}
    seed::Vector{Int}
    dims::Int
    chains::Int
    maxtreedepth::Int
    maxdeltaH::T
end

function Stan(dims, chains, T = Float64;
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

# TODO the goal is to generalize sample() such that it doesn't need any method specified
function sample(sampler::AbstractSampler{T}, ldg;
                iterations = 2000,
                warmup = div(iteration, 2),
                rng = Random.Xoshiro(sampler.seed),
                draws_initializer = :stan,
                stepsize_adapter = DualAverage(sampler.chains),
                trajectory_lengthadapter = (sampler.chains; initializer = :stan),
                metric_adapter = OnlineMoments(T, sampler.dims, sampler.chains),
                adaptation_schedule = WindowedAdaptationSchedule(warmup),
                integrator = :leapfrog,
                trace = (; acceptstat = zeros(T)); # TODO could probably use somesort of helper function
                # TODO what of passing in user-defined update_diagnostics()?
                kwargs...) where {T <: AbstractFloat}
    M = iterations + warmup
    draws = Array{T, 3}(undef, M, sampler.dims, sampler.chains)
    # TODO can gradients always be a Matrix, even for Stan methods
    # Stan doesn't need to be parallel, right? rather multithreaded.
    gradients = Matrix{T}(undef, sampler.dims, sampler.chains)
    momenta = randn!(similar(gradients))
    # TODO will need to move this scale and shift to somewhere more specific to MEADS
    acceptance_probabilities = 2 .* rand(sampler.chains) .- 1
    # TODO fix argument order for initialize_* methods
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
    
    initialize_trajectorylength!(trajectorylength_adapter,
                                 stepsize(stepsize_adapter);
                                 kwargs...)
    for m in 1:M
        # TODO need a loop for chains
        # how to generalize for MEADS?
        info = transition!(sampler, m, ldg, momenta, draws; kwargs...)
        adapt!(sampler,
               adaptation_schedule,
               metric_adapter,
               stepsize_adapter,
               trajectorylength_adapter;
               info,
               kwargs...)
        # update_trace(trace, m, info)
    end
    return draws, stepsize_adapter, metric_adapter, diagnostics
end

# TODO function kernel!(...)

function adapt!(sampler, schedule::WindowedAdaptationSchedule,
                metric_adapter, stepsize_adapter, trajectorylength_adapter,
                i, ldg, draws; kwargs...)
    if i <= schedule.warmup
        adapt_stepsize!(sampler, schedule, stepsize_adapter, i, ldg, draws; kwargs...)
        set_stepsize!(sampler, schedule, stepsize_adapter, i, ldg, draws; kwargs...)
        
        adapt_metric!(sampler, schedule, metric_adapter, i, ldg, draws; kwargs...)
        set_metric!(sampler, schedule, metric_adapter, i, ldg, draws; kwargs...)
        
        adapt_trajectorylength!(sampler, schedule, trajectorylength_adapter, i, ldg, draws; kwargs...)
        set_trajectorylength!(sampler, schedule, trajectorylength_adapter, i, ldg, draws; kwargs...)

        if i == schedule.closewindow
            # TODO aren't I missing re-initialize_stepsize?
            reset!(stepsize_adapter)
            reset!(metric_adapter)
            calculate_nextwindow!(schedule)
        end
    else
        set_stepsize!(sampler, schedule, stepsize_adapter, i, ldg, draws;
                      weighted_average = true, kwargs...)
    end
end

function adapt_stepsize!(sampler, schedule, stepsize_adapter, i, draws; kwargs...)
    update!(stepsize_adapeter, i, kwargs[:acceptstat]; kwargs...)
end

function set_stepsize!(sampler, schedule, stepsize_adapter, i, draws; kwargs...)
    for chain in 1:sampler.chains
        sampler.stepsize[chain] = stepsize(stepsize_adapter, chain; kwargs...)
    end
end

function adapt_metric!(sampler, schedule, metric_adapter::OnlineMoments, i, ldg, draws; kwargs...)
    if schedule.firstwindow <= i <= schedule.lastwindow
        update!(metric_adapter, draws[i, :, :]; kwargs...)
    end
end

function set_metric!(sampler, schedule, metric_adapter::OnlineMoments, i, ldg, draws; kwargs...)
    for chain in 1:sampler.chains
        sampler.metric[:, chain] .= metric(metric_adapt, chain; kwargs...)
    end
end

function update_trace!(trace, m, info)
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
            trace.accepted[m] = info[k]
        end
    end
end

