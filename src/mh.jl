abstract type AbstractMH{T} <: AbstractSampler{T} end

struct MH{T} <: AbstractMH{T}
    metric::Matrix{T}
    stepsize::Vector{T}
    dims::Int
    chains::Int
end

function MH(
    dims, chains=4, T=Float64; metric=ones(T, dims, chains), stepsize=ones(T, chains)
)
    D = convert(Int, dims)::Int
    return MH(metric, stepsize, D, chains)
end

function sample!(
    sampler::MH{T},
    ld;
    iterations=10000,
    warmup=10000,
    rngs=Random.Xoshiro.(rand(1:typemax(Int), sampler.chains)),
    draws_initializer=:mh,
    stepsize_initializer=:mh,
    stepsize_adapter=StepsizeDualAverage(
        sampler.stepsize; initializer=stepsize_initializer, δ=0.3
    ),
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
        stepsize_adapter, sampler.metric, rngs, ld, draws[1, :, :]; kwargs...
    )
    set_stepsize!(sampler, stepsize_adapter; kwargs...)

    for m in 1:M
        transition!(sampler, m, ld, draws, rngs, diagnostics; kwargs...)

        adapt!(
            sampler,
            adaptation_schedule,
            diagnostics,
            m,
            ld,
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

function transition!(sampler::MH, m, ld, draws, rngs, trace; kwargs...)
    for chain in axes(draws, 3) # TODO multi-thread-able
        @views info = mh_kernel!(
            draws[m, :, chain],
            draws[m + 1, :, chain],
            rngs[chain],
            sampler.dims,
            sampler.metric[:, chain],
            sampler.stepsize[chain],
            ld;
            kwargs...,
        )
        record!(trace, info, m + 1, chain)
    end
end

function mh_kernel!(position, position_next, rng, dims, metric, stepsize, ld; kwargs...)
    T = eltype(position)
    position_next .= position .+ randn(rng, T, dims) .* stepsize .* sqrt.(metric)
    a = exp(ld(copy(position_next); kwargs...) - ld(copy(position); kwargs...))
    acceptstat = min(1, a)
    accepted = rand(rng, T) < acceptstat
    position_next .= position_next .* accepted .+ position .* (1 - accepted)
    return (; accepted, acceptstat, stepsize)
end
