function initialize_stepsize!(adapter, metric, rng, ldg, draws, gradients; kwargs...)
    init_stepsize!(adapter.initializer, adapter, metric, rng, ldg, draws, gradients; kwargs...)
end

function init_stepsize!(method::Symbol, adapter, metric, rng, ldg, draws, gradients; kwargs...)
    init_stepsize!(Val{method}(), adapter, metric, rng, ldg, draws, gradients; kwargs...)
end

function init_stepsize!(::Val{:stan}, adapter, metric, rng, ldg, draws, gradients; kwargs...)
    stepsize = optimum(adapter)
    for chain in eachindex(draws[1])
        stan_init_stepsize!(stepsize[chain:chain], metric[:, chain], rng[chain], ldg, draws[1][chain], gradients[chain]; kwargs...)
    end
end

function stan_init_stepsize!(stepsize, metric, rng, ldg, position, gradient; kwargs...)
    T = eltype(position)
    dims = length(position)
    q = copy(position)
    momenta = rand_momenta(rng, dims, metric)

    ld = ldg(q, gradient; kwargs...)
    H0 = hamiltonian(ld, momenta, metric)

    integrator = get(kwargs, :integrator, :leapfrog)
    ε = metric .* stepsize[]
    ld = integrate!(integrator, ldg, q, momenta, gradient, ε, 1)
    H = hamiltonian(ld, momenta, metric) # TODO see next TODO about negatives; re bridgestan
    isnan(H) && (H = typemin(T)) # TODO typemin(T), I think, when using bridgestan

    ΔH = H0 - H
    dh = convert(T, log(0.8))
    direction = ΔH > dh ? 1 : -1

    while true
        q .= position
        momenta = rand_momenta(rng, dims, metric)
        H0 = hamiltonian(ld, momenta, metric)

        println(1 ./ (1 .+ exp.(-q)))
        ε = metric .* stepsize[]
        ld = integrate!(integrator, ldg, q, momenta, gradient, ε, 1)
        H = hamiltonian(ld, momenta, metric)
        isnan(H) && (H = typemin(T))

        ΔH = H0 - H
        if direction == 1 && !(ΔH > dh)
            break
        elseif direction == -1 && !(ΔH < dh)
            break
        else
            stepsize[] = direction == 1 ? 2 * stepsize[] : stepsize[] / 2
        end

        @assert stepsize[] <= 1.0e7 "Posterior is impropoer.  Please check your model."
        @assert stepsize[] >= 0.0 "No acceptable small step size could be found.  Perhaps the posterior is not continuous."
    end
end

function initialize_stepsize!(::Val{:adam}, adapter, metric, rng, ldg,
                              draws::Array, gradients::Matrix; kwargs...)
    # TODO need do anything here?
end

function init_stepsize!(::Val{:chees}, adapter, metric, rng, ldg, draws, gradients; kwargs...)
    T = eltype(draws[1][1][1])
    chains = length(draws[1])
    num_metrics = size(metrics, 2)

    αs = zeros(T, chains)
    ε = 2 * one(T)
    harmonic_mean = zero(T)
    tmp = similar(draws[1][1])

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
