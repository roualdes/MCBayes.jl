function hmc!(integrator, next_position, position, momenta, ldg, gradient, stepsize, steps, maxdeltaH;
              kwargs...)
    T = eltype(position)
    next_position .= position

    lp = ldg(position, gradient; kwargs...)
    H1 = hamiltonian(lp, momenta)
    isnan(H1) && (H1 = typemin(T))

    integrate!(integrator, ldg, position, momenta, gradient, stepsize, steps; kwargs...)

    H2 = hamiltonian(lp, momenta)
    isnan(H2) && (H2 = typemin(T))
    divergent = divergence(H2, H1, maxdeltaH)

    a = min(1, exp(H1 + H2))
    accepted = rand(rng, T) < a
    accepted && (next_position .= position)

    return (;
            accepted,
            divergent,
            acceptstat = a
            )
end

function pghmc!(integrator, next_position, position, momenta, ldg, gradient, stepsize, acceptance_probability, δ, nonreversible_update, maxdeltaH;
                kwargs...)
    T = eltype(position)
    next_position .= position
    next_momenta .= momenta

    lp = ldg(position, gradient; kwargs...)
    H1 = hamiltonian(lp, momenta)
    isnan(H1) && (H1 = typemin(T))

    integrate!(integrator, ldg, position, momenta, gradient, stepsize, 1; kwargs...)

    H2 = hamiltonian(lp, momenta)
    isnan(H2) && (H2 = typemin(T))
    divergent = divergence(H2, H1, maxdeltaH)

    a = H1 + H2
    accepted =  log(abs(acceptance_probability)) < a
    if accepted
        next_position .= position
        next_momenta .= momenta
        acceptance_probability *= exp(-a)
    else
        next_momenta .*= -1
    end

    acceptance_probability = if nonreversible_update
        (acceptance_probability + 1 + δ) % 2 - 1
    else
        rand(rng, T)
    end

    return (;
            accepted,
            divergent,
            acceptstat = abs(acceptance_probability)
            )
end

function rand_momentum(rng, dims, metric)
    return randn(rng, eltype(metric), dims) ./ sqrt.(metric)
end

function hamiltonian(ld, momenta, metric)
    return -ld + dot(momenta, Diagonal(metric), momenta) / 2
end
