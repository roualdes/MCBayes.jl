abstract type AbstractSampler{T <: AbstractFloat} end


function sample(sampler::AbstractSampler{T}, ldg;
                iterations = 2000,
                warmup = div(iterations, 2),
                rng = Tuple(Random.Xoshiro(rand(UInt32)) for _ in 1:sampler.chains);
                kwargs...) where {T <: AbstractFloat}
    M = iterations + warmup
    draws = Array{T, 3}(undef, iterations, sampler.chains, sampler.dims)
    diagnostics = preallocate_diagnostics(sampler; kwargs...) # TODO(ear) implement
    initialize_draws!(sampler, ldg, draws; kwargs...)
    initialize_stepsize!(sampler, ldg, draws; kwargs...)
    for m in 1:M
        transition!(sampler, m, ldg, draws; diagnostics, kwargs...)
        adapt_chains!(sampler, m, ldg, draws; wa, kwargs...)
    end
    return draws, sampler, diagnostics
end

# HMC(initial_draw;
#     chains = 10,
#     initialize_draws = Uniform(),
#     stepsize = SGAStepSize(),
#     trajectorylength = FixedTrajectoryLength(T = 1),
#     metric = AdaptiveMetric(:median))

# ChEES(initial_draw;
#       chains = 10,
#       initialize_draws = Adam(steps = 100, tuing_parameters...),
#       stepsize = SGAStepSize(tuning_parameters...),
#       trajectorylength = AdaptiveChEES(),
#       metric = AdaptiveMetric(:mean))

# SNAPER(initial_draw;
#        chains = 10,
#        initialize_draws = Adam(steps = 100, tuing_parameters...),
#        stepsize = SGAStepSize(tuning_parameters...),
#        trajectorylength = AdaptiveSNAPER(),
#        metric = AdaptiveMetric(:mean))
