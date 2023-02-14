# TODO sampler specific records is the only way forward here; especially for SGA methods
# until we redesign things to have their own sample!() methods
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
    dims = sampler.dims
    return (;
            acceptstat=zeros(T, iterations, chains),
            accepted=zeros(Bool, iterations, chains),
            stepsize=zeros(T, iterations, 1),
            position=zeros(T, dims, chains),
            momentum=zeros(T, dims, chains),
            trajectorylength=zeros(T, iterations, 1),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            )
end
