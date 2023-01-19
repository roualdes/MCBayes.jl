# TODO move this into stepsize_adapter
function initialize_stepsize!(stepsize_adapter, sampler, rngs, ldg, positions; kwargs...)
    init_stepsize!(
        stepsize_adapter.initializer,
        stepsize_adapter,
        sampler,
        rngs,
        ldg,
        positions;
        kwargs...,
    )
end

function init_stepsize!(
    method::Symbol, stepsize_adapter, sampler, rngs, ldg, positions; kwargs...
)
    init_stepsize!(Val{method}(), stepsize_adapter, sampler, rngs, ldg, positions; kwargs...)
end

function init_stepsize!(
    ::Val{:mh}, stepsize_adapter, sampler, rngs, ld, positions; kwargs...
)
    dims = size(positions, 1)
    stepsize_adapter.stepsize .= 2.38 / sqrt(dims) # TODO double check this number
end

function init_stepsize!(
    ::Val{:none}, stepsize_adapter, sampler, rngs, ldg, positions; kwargs...
) end

function init_stepsize!(
    ::Val{:stan}, stepsize_adapter, sampler, rngs, ldg, positions; kwargs...
)
    stepsize = stepsize_adapter.stepsize
    metric = sampler.metric

    for chain in axes(positions, 2)
        @views stepsize_adapter.stepsize[chain] = stan_init_stepsize(
            stepsize[chain],
            metric[:, chain],
            rngs[chain],
            ldg,
            positions[:, chain];
            kwargs...,
        )
    end
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

function init_stepsize!(::Val{:meads}, stepsize_adapter, sampler, rngs, ld, positions; kwargs...)
    for f in 1:sampler.folds
        k = (f + 1) % sampler.folds + 1
        kfold = sampler.partition[:, k]

        q = positions[:, kfold]
        sigma = std(q, dims = 2)

        update!(stepsize_adapter, ldg, q, sigma, f; kwargs...)
    end
end

# # TODO needs a second look
# function init_stepsize!(::Val{:chees}, adapter, metric, rng, ldg, draws; kwargs...)
#     T = eltype(draws[1][1][1])
#     chains = length(draws[1])
#     num_metrics = size(metrics, 2)

#     αs = zeros(T, chains)
#     ε = 2 * one(T)
#     harmonic_mean = zero(T)
#     tmp = similar(draws[1][1])

#     while harmonic_mean < oftype(x, 0.5)
#         ε /= 2

#         for (metric, chain) in zip(Iterators.cylce(1:metrics), 1:chains)
#             # TODO keep, if this is needed. Otherwise, ditch. adapter.ε = ε
#             # TODO info = hmc!()
#             αs[chain] = info.acceptstat
#         end
#         harmonic_mean = inv(mean(inv, αs))
#     end
#     adapter.ε = ε
# end
