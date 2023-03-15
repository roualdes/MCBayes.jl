# TODO needs better name since the intention is to not initialize
# TODO do the other initialized values need such a don't initialize type?

struct StepsizeInitializer end

function initialize_stepsize!(
    initialzer::StepsizeInitializer,
    stepsize_adapter,
    sampler,
    rngs,
    ldg,
    positions;
    kwargs...,
) end

struct StepsizeInitializerStan end

function initialize_stepsize!(
    initialzer::StepsizeInitializerStan,
    stepsize_adapter::StepsizeConstant,
    sampler,
    rngs,
    ldg,
    positions;
    kwargs...,
    )
end

function initialize_stepsize!(
    initialzer::StepsizeInitializerStan,
    stepsize_adapter,
    sampler,
    rngs,
    ldg,
    positions;
    kwargs...,
    )

    for chain in 1:size(positions, 2)
        stepsize_adapter.stepsize[chain] = stan_init_stepsize(
            sampler.stepsize[chain],
            sampler.metric[:, chain],
            rngs[chain],
            ldg,
            positions[:, chain];
            kwargs...,
        )
    end

    set!(sampler, stepsize_adapter; kwargs...)
end

function stan_init_stepsize(stepsize, metric, rng, ldg, position; kwargs...)
    T = eltype(position)
    dims = length(position)
    q = copy(position)
    momentum = randn(rng, T, dims)

    ld, gradient = ldg(q; kwargs...)
    H0 = hamiltonian(ld, momentum)

    ld, gradient = leapfrog!(
        q, momentum, ldg, gradient, stepsize .* sqrt.(metric), 1; kwargs...
    )
    H = hamiltonian(ld, momentum)
    isnan(H) && (H = typemax(T))

    ΔH = H0 - H
    dh = convert(T, log(0.8))::T
    direction = ΔH > dh ? 1 : -1

    while true
        momentum .= randn(rng, T, dims)
        q .= position
        ld, gradient = ldg(q; kwargs...)
        H0 = hamiltonian(ld, momentum)

        ld, gradient = leapfrog!(
            q, momentum, ldg, gradient, stepsize .* sqrt.(metric), 1; kwargs...
        )
        H = hamiltonian(ld, momentum)
        isnan(H) && (H = typemax(T))

        ΔH = H0 - H
        isnan(ΔH) && (ΔH = typemax(T))

        if direction == 1 && !(ΔH > dh)
            break
        elseif direction == -1 && !(ΔH < dh)
            break
        else
            stepsize = direction == 1 ? 2 * stepsize : stepsize / 2
        end

        if stepsize > 1e7
            throw("Posterior is impropoer.  Please check your model.")
        end
        if stepsize <= 0.0
            throw(
                "No acceptable small step size could be found.  Perhaps the posterior is not continuous.",
            )
        end
    end
    return stepsize
end

struct StepsizeInitializerRWM end

function initialize_stepsize!(
    initialzer::StepsizeInitializerRWM,
    stepsize_adapter,
    sampler,
    rngs,
    ldg,
    positions;
    kwargs...,
)
    dims = size(positions, 1)
    stepsize_adapter.stepsize .= 2.38 / sqrt(dims) # TODO double check this number
    set!(sampler, stepsize_adapter; kwargs...)
end

struct StepsizeInitializerMEADS end

function initialize_stepsize!(
    initialzer::StepsizeInitializerMEADS,
    stepsize_adapter,
    sampler,
    rngs,
    ldg,
    positions;
    kwargs...,
)
    for f in 1:(sampler.folds)
        k = (f + 1) % sampler.folds + 1
        kfold = sampler.partition[:, k]

        q = positions[:, kfold]
        sigma = std(q; dims=2)

        update!(stepsize_adapter, ldg, q, sigma, f; kwargs...)
    end
    set!(sampler, stepsize_adapter; kwargs...)
end

struct StepsizeInitializerSGA end

function initialize_stepsize!(
    initialzer::StepsizeInitializerSGA,
    stepsize_adapter::StepsizeConstant,
    sampler,
    rngs,
    ldg,
    positions;
    kwargs...,
    )
end

function initialize_stepsize!(initialzer::StepsizeInitializerSGA, stepsize_adapter, sampler, rngs, ldg, positions; kwargs...)
    T = eltype(stepsize_adapter)
    num_chains = size(positions, 2)

    αs = zeros(T, num_chains)
    stepsize = 2 * sampler.stepsize[1]
    harmonic_mean = zero(T)
    tmp = similar(positions)

    metric = sampler.metric
    num_metrics = size(metric, 2)
    cycle_metrics = Iterators.cycle(1:num_metrics)

    while harmonic_mean < oftype(harmonic_mean, 0.5)
        stepsize /= 2
        for (c, m) in zip(1:num_chains, cycle_metrics)
            info = hmc!(positions[:, c],
                        tmp[:, c],
                        ldg,
                        rngs[c],
                        sampler.dims,
                        metric[:, m],
                        stepsize,
                        1,
                        1000;
                        kwargs...)
            αs[c] = info.acceptstat
        end
        harmonic_mean = inv(mean(inv, αs))
    end

    stepsize_adapter.stepsize .= stepsize
    set!(sampler, stepsize_adapter; kwargs...)
end
