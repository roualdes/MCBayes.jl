# TODO sampler specific records is the only way forward here; especially for SGA methods
# until we redesign things to have their own sample!() methods

function trace(sampler::Stan{T}, iterations) where {T}
    chains = sampler.chains
    return (;
            accepted=zeros(Bool, iterations, chains),
            acceptstat=zeros(T, iterations, chains),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            leapfrog=zeros(Int, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            treedepth=zeros(Int, iterations, chains),
            ld=zeros(T, iterations, chains),
    )
end

function record!(sampler::Stan{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (:accepted, :acceptstat, :divergence, :energy, :leapfrog, :stepsize, :treedepth, :ld)
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
end

function trace(sampler::RWM{T}, iterations) where {T}
    chains = sampler.chains
    return (;
        accepted=zeros(Bool, iterations, chains),
        acceptstat=zeros(T, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            ld=zeros(T, iterations, chains),
    )
end

function record!(sampler::RWM{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (:accepted, :acceptstat, :stepsize, :ld)
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
end

function trace(sampler::MEADS{T}, iterations) where {T}
    chains = sampler.chains
    folds = sampler.folds
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        damping=zeros(T, iterations, folds),
        divergence=zeros(Bool, iterations, chains),
        drift=zeros(T, iterations, folds),
        energy=zeros(T, iterations, chains),
        noise=zeros(T, iterations, folds),
            stepsize=zeros(T, iterations, folds),
            ld=zeros(T, iterations, folds),
    )
end

function record!(
    sampler::MEADS{T}, trace::NamedTuple, info, iteration, chain, fold
) where {T}
    keys = (
        :accepted, :acceptstat, :damping, :divergence, :drift, :energy, :noise, :stepsize, :ld,
    )
    trace[:accepted][iteration, chain] = info[:accepted]
    trace[:acceptstat][iteration, chain] = info[:acceptstat]
    trace[:damping][iteration, fold] = info[:damping]
    trace[:divergence][iteration, chain] = info[:divergence]
    trace[:drift][iteration, fold] = info[:drift]
    trace[:energy][iteration, chain] = info[:energy]
    trace[:noise][iteration, fold] = info[:noise]
    trace[:stepsize][iteration, fold] = info[:stepsize]
    trace[:ld][iteration, fold] = info[:ld]
end

function trace(sampler::MALA{T}, iterations) where {T}
    chains = sampler.chains
    return (;
            acceptstat=zeros(T, iterations, chains),
            accepted=zeros(Bool, iterations, chains),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            ld=zeros(T, iterations, chains),
    )
end

function record!(sampler::MALA{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (:accepted, :acceptstat, :divergence, :energy, :stepsize, :ld)
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
end

function trace(sampler::MALT{T}, iterations) where {T}
    chains = sampler.chains
    return (;
            acceptstat=zeros(T, iterations, chains),
            accepted=zeros(Bool, iterations, chains),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            damping=zeros(T, iterations, chains),
            noise=zeros(T, iterations, chains),
            trajectorylength=zeros(T, iterations, chains),
            ld=zeros(T, iterations, chains),
    )
end

function record!(sampler::MALT{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (:accepted, :acceptstat, :divergence, :energy, :stepsize, :noise, :damping, :ld, :trajectorylength)
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
end

function trace(sampler::AbstractSGA{T}, iterations) where {T}
    chains = sampler.chains
    dims = sampler.dims
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        divergence=zeros(Bool, iterations, chains),
        energy=zeros(T, iterations, chains),
        momentum=zeros(T, dims, chains),
        position=zeros(T, dims, chains),
        stepsize=zeros(T, iterations, 1),
        steps=zeros(Int, iterations, 1),
            trajectorylength=zeros(T, iterations, 1),
            ld=zeros(T, iterations, chains),
    )
end

function record!(
    sampler::AbstractSGA{T}, trace::NamedTuple, info, iteration, chain
) where {T}
    keys = (:accepted, :acceptstat, :divergence, :energy, :ld)
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
    trace[:steps][iteration] = info[:steps]
    trace[:trajectorylength][iteration] = info[:trajectorylength]
    trace[:stepsize][iteration] = info[:stepsize]
    trace[:momentum][:, chain] .= info[:momentum]
    trace[:position][:, chain] .= info[:position]
end
