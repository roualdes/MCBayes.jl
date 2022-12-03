function initialize_stepsize!(adapter, metric, rng, ldg, draws, gradient; kwargs...)
    init_stepsize!(adapter.initializer, adapter, metric, rng, ldg, draws, gradient;
                         kwargs...)
end

function init_stepsize!(method::Symbol, adapter, metric, rng, ldg, draws, gradient; kwargs...)
    init_stepsize!(Val{method}(), adapter, metric, rng, ldg, draws, gradients; kwargs...)
end

function init_stepsize!(::Val{:stan}, adapter, metric, rng, ldg,
                        draws::Array, gradients::Matrix; kwargs...)
    for chain in axes(draws, 3)
        stan_init_stepsize!(adapter, metric, rng, ldg,
                            draws[1, :, chain], gradients[:, chain]; kwargs...)
    end
end

function stan_init_stepsize!(adapter, metric, rng, ldg, position, gradient; kwargs...)
    T = eltype(draws)
    dims = size(draws, 2)
    q = copy(position)
    momenta = rand_momenta(rng, dims, metric) # TODO need this function;
    # probably located wherever leapfrog should go

    lp = ldg(q, gradient; kwargs...)
    H0 = hamiltonian(lp, momenta, metric) # TODO need this function;
    # probably located wherever leapfrog should go

    # TODO need function integrate which acts on a method::Symbol
    # see if we can skip metric inside the integrator
    integrator = get(kwargs, integrator, :leapfrog)
    ε = metric .* stepsize(adapter)
    lp = integrate!(integrator, q, momenta, lp, gradient)
    H = hamiltonian(lp, momenta, metric) # TODO see next TODO about negatives; re bridgestan
    isnan(H) && (H = typemax(T)) # TODO typemin(T), I think, when using bridgestan

    ΔH = H0 - H
    dh = convert(T, log(0.8))
    direction = ΔH > dh ? 1 : -1

    while true
        q .= position
        momenta = rand_momenta(rng, dims, metric)
        H0 = hamiltonian(lp, momenta, metric)

        lp = integrate!(integrator, q, momenta, lp, gradient)
        H = hamiltonian(lp, momenta, metric) # TODO copied from above
        isnan(H) && (H = typemax(T)) # TODO copied from above

        ΔH = H0 - H
        if direction == 1 && !(ΔH > dh)
            break
        elseif direction == -1 && !(ΔH < dh)
            break
        else
            adapter.ε = direction == 1 ? 2 * adapter.ε : 0.5 * adapter.ε
        end

        @assert adapter.ε <= 1.0e7 "Posterior is impropoer.  Please check your model."
        @assert adapter.ε >= 0.0 "No acceptable small step size could be found.  Perhaps the posterior is not continuous."
    end
end

function initialize_stepsize!(::Val{:adam}, adapter, metric, rng, ldg,
                              draws::Array, gradients::Matrix; kwargs...)
    # TODO need do anything here?
end

function init_stepsize!(::Val{:chees}, adapter, metric, rng, ldg,
                        draws::Array, gradients::Matrix; kwargs...)
    T = eltype(draws)
    chains = size(draws, 3)
    num_metrics = size(metrics, 2)

    αs = zeros(T, chains)
    ε = 2 * one(T)
    harmonic_mean = zero(T)
    tmp = similar(draws[1, :, 1])

    while harmonic_mean < oftype(x, 0.5)
        ε /= 2

        for (metric, chain) in zip(Iterators.cylce(1:metrics), 1:chains)
            # TODO keep, if this is needed. Otherwise, ditch. adapter.ε = ε
            # TODO info = hmc!()
            αs[chain] = info.acceptstat
        end
        harmonic_mean = inv(mean(inv, αs))
    end
    adapter.ε = ε
end
