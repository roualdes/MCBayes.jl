function record!(trace::NamedTuple, info, iteration, chain)
    keys = (
        :accepted,
        :divergence,
        :energy,
        :stepsize,
        :acceptstat,
        :treedepth,
        :leapfrog,
        :damping,
        :noise,
        :drift,
    )
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
end

function trace(sampler::Stan{T}, iterations) where {T}
    chains = sampler.chains
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        divergence=zeros(Bool, iterations, chains),
        energy=zeros(T, iterations, chains),
        stepsize=zeros(T, iterations, chains),
        treedepth=zeros(Int, iterations, chains),
        leapfrog=zeros(Int, iterations, chains),
    )
end

function trace(sampler::RWM{T}, iterations) where {T}
    chains = sampler.chains
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        stepsize=zeros(T, iterations, chains),
    )
end

function trace(sampler::MEADS{T}, iterations) where {T}
    chains = sampler.chains
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        stepsize=zeros(T, iterations, sampler.folds),
        energy=zeros(T, iterations, chains),
        divergence=zeros(Bool, iterations, chains),
        # TODO want damping, noise, drift?
    )
end

function trace(sampler::MALA{T}, iterations) where {T}
    chains = sampler.chains
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        stepsize=zeros(T, iterations, chains),
        divergence=zeros(Bool, iterations, chains),
        energy=zeros(T, iterations, chains),
    )
end

function trace(sampler::AbstractSGA{T}, iterations) where {T}
    chains = sampler.chains
    return (;
            acceptstat=zeros(T, iterations, chains),
            accepted=zeros(Bool, iterations, chains),
            stepsize=zeros(T, iterations, 1),
            trajectorylength=zeros(T, iterations, 1),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            )
end
