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
    keys = (
        :accepted, :acceptstat, :divergence, :energy, :leapfrog, :stepsize, :treedepth, :ld
    )
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
        :accepted,
        :acceptstat,
        :damping,
        :divergence,
        :drift,
        :energy,
        :noise,
        :stepsize,
        :ld,
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
    dims = sampler.dims
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        divergence=zeros(Bool, iterations, chains),
        energy=zeros(T, iterations, chains),
        stepsize=zeros(T, iterations, chains),
        steps=zeros(Int, iterations),
        damping=zeros(T, iterations, chains),
        noise=zeros(T, iterations, chains),
        trajectorylength=zeros(T, iterations, chains),
        ld=zeros(T, iterations, chains),
        previousmomentum=zeros(T, dims, chains),
        momentum=zeros(T, dims, chains),
        position=zeros(T, dims, chains),
        pca=zeros(T, iterations, dims),
    )
end

function record!(sampler::MALT{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (
        :accepted,
        :acceptstat,
        :divergence,
        :energy,
        :stepsize,
        :noise,
        :damping,
        :ld,
        :trajectorylength,
    )
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
    trace[:steps][iteration] = info[:steps]
    trace[:previousmomentum][:, chain] .= trace[:momentum][:, chain]
    trace[:momentum][:, chain] .= info[:momentum]
    trace[:position][:, chain] .= info[:position]
    trace[:pca][iteration, :] .= info[:pca]
end

function trace(sampler::AbstractSGA{T}, iterations) where {T}
    chains = sampler.chains
    dims = sampler.dims
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        divergence=zeros(Bool, iterations, chains),
        energy=zeros(T, iterations, chains),
        previousmomentum=zeros(T, dims, chains),
            proposedp=zeros(T, dims, chains),
            proposedq=zeros(T, dims, chains),
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
    trace[:previousmomentum][:, chain] .= info[:previousmomentum]
    trace[:proposedp][:, chain] .= info[:proposedp]
    trace[:proposedq][:, chain] .= info[:proposedq]
end

function trace(sampler::XHMC{T}, iterations) where {T}
    chains = sampler.chains
    dims = sampler.dims
    return (;
        acceptstat=zeros(T, iterations, chains),
        accepted=zeros(Bool, iterations, chains),
        divergence=zeros(Bool, iterations, chains),
        energy=zeros(T, iterations, chains),
        stepsize=zeros(T, iterations, chains),
        steps=zeros(Int, iterations, 1),
        damping=zeros(T, iterations, chains),
        noise=zeros(T, iterations, chains),
        trajectorylength=zeros(T, iterations, chains),
        ld=zeros(T, iterations, chains),
        previousmomentum=zeros(T, dims, chains),
        momentum=zeros(T, dims, chains),
        position=zeros(T, dims, chains),
        retries=zeros(Int, sampler.K, chains),
    )
end

function record!(sampler::XHMC{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (
        :accepted,
        :acceptstat,
        :divergence,
        :energy,
        :stepsize,
        :noise,
        :damping,
        :ld,
        :trajectorylength,
    )
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
    trace[:steps][iteration] = info[:steps]
    trace[:previousmomentum][:, chain] .= trace[:momentum][:, chain]
    trace[:momentum][:, chain] .= info[:momentum]
    trace[:position][:, chain] .= info[:position]
    trace[:retries][info[:retries], chain] += 1
end

function trace(sampler::DrMALA{T}, iterations) where {T}
    chains = sampler.chains
    dims = sampler.dims
    return (;
            acceptstat=zeros(T, iterations, chains),
            finalacceptstat=zeros(T, iterations, chains),
            accepted=zeros(Bool, iterations, chains),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            leapfrog=zeros(Int, iterations, chains),
            damping=zeros(T, iterations, dims, chains),
            noise=zeros(T, iterations, chains),
            ld=zeros(T, iterations, chains),
            retries=zeros(Int, 3, iterations, chains),
            reductionfactor=zeros(T, iterations),
            proposedq=zeros(T, dims, chains),
            proposedp=zeros(T, dims, chains),
            previousmomentum=zeros(T, dims, chains)
    )
end

function record!(sampler::DrMALA{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (
        :accepted,
        :acceptstat,
        :finalacceptstat,
        :divergence,
        :energy,
        :stepsize,
        :ld,
        :leapfrog,
    )
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
    trace[:previousmomentum][:, chain] .= info[:previousmomentum]
    trace[:proposedq][:, chain] .= info[:proposedq]
    trace[:proposedp][:, chain] .= info[:proposedp]
    trace[:reductionfactor][iteration] = info[:reductionfactor]
    trace[:noise][iteration, chain] = info[:noise]
    trace[:damping][iteration, :, chain] .= info[:damping]
    trace[:retries][info[:retries], iteration, chain] += 1
end

function trace(sampler::DRGHMC{T}, iterations) where {T}
    chains = sampler.chains
    dims = sampler.dims
    return (;
            acceptstat=zeros(T, iterations, chains),
            finalacceptstat=zeros(T, iterations, chains),
            accepted=zeros(Bool, iterations, chains),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            leapfrog=zeros(Int, iterations, chains),
            damping=zeros(T, iterations, dims, chains),
            noise=zeros(T, iterations, chains),
            ld=zeros(T, iterations, chains),
            retries=zeros(Int, 3, iterations, chains),
            reductionfactor=zeros(T, iterations),
    )
end

function record!(sampler::DRGHMC{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (
        :accepted,
        :acceptstat,
        :finalacceptstat,
        :divergence,
        :energy,
        :stepsize,
        :ld,
        :leapfrog,
    )
    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end
    trace[:reductionfactor][iteration] = info[:reductionfactor]
    trace[:noise][iteration, chain] = info[:noise]
    trace[:damping][iteration, :, chain] .= info[:damping]
    trace[:retries][info[:retries], iteration, chain] += 1
end

function trace(sampler::DRHMC{T}, iterations) where {T}
    chains = sampler.chains
    dims = sampler.dims
    return (;
            firstacceptstat=zeros(T, iterations, chains),
            finalacceptstat=zeros(T, iterations, chains),
            accepted=zeros(Bool, iterations, chains),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            leapfrog=zeros(Int, iterations, chains),
            steps=zeros(Int, iterations, chains),
            # damping=zeros(T, iterations, dims, chains),
            # noise=zeros(T, iterations, chains),
            ld=zeros(T, iterations, chains),
            retries=zeros(Int, 3, iterations, chains),
            reductionfactor=zeros(T, iterations),
            proposedq=zeros(T, dims, chains),
            proposedp=zeros(T, dims, chains),
            previousmomentum=zeros(T, dims, chains),
            nextmomentum=zeros(T, dims, chains),
            firsttry=zeros(Int, iterations, chains)
    )
end

function record!(sampler::DRHMC{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (
        :stepsize,
        :accepted,
        :firstacceptstat,
        :finalacceptstat,
        :divergence,
        :energy,
        :ld,
        :leapfrog,
        :steps,
    )

    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end

    trace[:previousmomentum][:, chain] .= trace[:nextmomentum][:, chain]
    trace[:nextmomentum][:, chain] .= info[:momentum]
    trace[:proposedq][:, chain] .= info[:proposedq]
    trace[:proposedp][:, chain] .= info[:proposedp]
    trace[:reductionfactor][iteration] = info[:reduction_factor]
    # trace[:noise][iteration, chain] = info[:noise]
    # trace[:damping][iteration, :, chain] .= info[:damping]
    trace[:retries][info[:retries], iteration, chain] += 1
    trace[:firsttry][iteration, chain] = info[:firsttry]
end

function trace(sampler::DHMC{T}, iterations) where {T}
    chains = sampler.chains
    dims = sampler.dims
    return (;
            accepted=zeros(Bool, iterations, chains),
            acceptstat=zeros(T, iterations, chains),
            divergence=zeros(Bool, iterations, chains),
            energy=zeros(T, iterations, chains),
            stepsize=zeros(T, iterations, chains),
            leapfrog=zeros(Int, iterations, chains),
            treedepth=zeros(Int, iterations, chains),
            steps=zeros(Int, iterations, chains),
            ld=zeros(T, iterations, chains),
            retries=zeros(Int, 3, iterations, chains),
            reductionfactor=zeros(T, iterations),
            proposedq=zeros(T, dims, chains),
            proposedp=zeros(T, dims, chains),
            previousmomentum=zeros(T, dims, chains),
            pca = zeros(T, iterations, dims)
    )
end

function record!(sampler::DHMC{T}, trace::NamedTuple, info, iteration, chain) where {T}
    keys = (
        :stepsize,
        :accepted,
        :acceptstat,
        :divergence,
        :energy,
        :ld,
        :leapfrog,
        :treedepth,
    )

    for k in keys
        if haskey(info, k)
            trace[k][iteration, chain] = info[k]
        end
    end

    trace[:pca][iteration, :] .= info.pca
    D = size(trace[:proposedq], 1)
    trace[:proposedq][:, chain] .= get(info, :proposedq, randn(D))
    # trace[:previousmomentum][:, chain] .= trace[:nextmomentum][:, chain]
    trace[:previousmomentum][:, chain] .= get(info, :previousmomentum, randn(D))
    # trace[:proposedq][:, chain] .= info[:proposedq]
    trace[:proposedp][:, chain] .= get(info, :proposedp, randn(D))
    # trace[:reductionfactor][iteration] = info[:reduction_factor]
    # # trace[:noise][iteration, chain] = info[:noise]
    # # trace[:damping][iteration, :, chain] .= info[:damping]
    # trace[:retries][info[:retries], iteration, chain] += 1
    # trace[:firsttry][iteration, chain] = info[:firsttry]
end
