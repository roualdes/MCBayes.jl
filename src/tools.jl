function hmc!(
    integrator,
    next_position,
    position,
    momenta,
    ldg,
    gradient,
    stepsize,
    steps,
    maxdeltaH;
    kwargs...,
)
    # TODO(ear) needs updating to current interface
    T = eltype(position)
    next_position .= position

    lp, gradient = ldg(position; kwargs...)
    H1 = hamiltonian(lp, momenta)
    isnan(H1) && (H1 = typemin(T))

    integrate!(ldg, position, momenta, gradient, stepsize, steps; kwargs...)

    H2 = hamiltonian(lp, momenta)
    isnan(H2) && (H2 = typemin(T))
    divergent = divergence(H2, H1, maxdeltaH)

    a = min(1, exp(H1 + H2))
    accepted = rand(rng, T) < a
    accepted && (next_position .= position)

    return (; accepted, divergent, acceptstat=a)
end

function pghmc!(
    position,
    position_next,
    momentum,
    ldg,
    rng,
    dims,
    metric,
    stepsize,
    acceptance_probability,
    noise,
    drift,
    nonreversible_update,
    maxdeltaH;
    kwargs...,
)
    T = eltype(position)
    q = copy(position)
    p = copy(momentum)

    ld, gradient = ldg(q; kwargs...)
    H0 = hamiltonian(ld, p)
    isnan(H0) && (H0 = typemax(T))

    ld, gradient = leapfrog!(q, p, ldg, gradient, stepsize .* sqrt.(metric), 1; kwargs...)

    H = hamiltonian(ld, p)
    isnan(H) && (H = typemax(T))
    divergent = (H - H0) > maxdeltaH

    a = H0 - H
    accepted = log(abs(acceptance_probability[])) < a
    if accepted
        position_next .= q
        momentum .= p
        acceptance_probability[] *= exp(-a)
    else
        position_next .= position
        momentum .*= -1
    end
    energy = hamiltonian(ld, momentum)
    momentum .= sqrt.(1 .- noise .^ 2) .* momentum .+ noise .* randn(rng, T, dims)

    acceptance_probability[] = if nonreversible_update
        (acceptance_probability[] + 1 + drift) % 2 - 1
    else
        rand(rng, T)
    end

    return (; accepted, divergent, energy, acceptstat=a > zero(a) ? one(a) : exp(a))
end

function rand_momentum(rng, dims, metric)
    return randn(rng, eltype(metric), dims) ./ sqrt.(metric)
end

function hamiltonian(ld, momenta, metric)
    return -ld + dot(momenta, Diagonal(metric), momenta) / 2
end

function hamiltonian(ld, momenta)
    return -ld + dot(momenta, momenta) / 2
end

function halton(n::Int; base::Int=2)
    x = 0.0
    s = 1.0
    while n > 0
        s /= base
        n, r = divrem(n, base)
        x += s * r
    end
    return x
end

function log1pexp(a)
    a > zero(a) && return a + log1p(exp(-a))
    return log1p(exp(a))
end

function logsumexp(a, b)
    T = typeof(a)
    a == typemin(T) && return b
    isinf(a) && isinf(b) && return typemax(T)
    a > b && return a + log1pexp(b - a)
    return b + log1pexp(a - b)
end

function logsumexp(v::AbstractVector)
    T = eltype(v)
    length(v) == 0 && return typemin(T)
    m = maximum(v)
    isinf(m) && return m
    return m + log(sum(vi -> exp(vi - m), v))
end

function max_eigenvalue(x)
    N = size(x, 2)
    trace_est = sum(z -> z^2, x)
    trace_sq_est = sum(z -> z^2, x' * x) / N
    return trace_sq_est / trace_est
end

function standardize_draws(x)
    scale = std(x; dims=2)
    z = x ./ scale
    location = mean(z; dims=2)
    z .-= location
    return z, scale
end
