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

function record!(trace::NamedTuple, info, iteration, chain)
    keys = (:accepted, :divergence, :energy, :stepsize, :acceptstat, :treedepth, :leapfrog)
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
end
