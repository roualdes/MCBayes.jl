function drhmc!(
    position,
    position_next,
    momentum,
    ldg!,
    rng,
    dims,
    metric,
    stepsize,
    steps,
    noise,
    J,
    reduction_factor,
    maxdeltaH;
    kwargs...,
)
    T = eltype(position)
    q = copy(position)
    p = noise .* momentum .+ sqrt.(1 .- noise .^ 2) .* randn(rng, T, dims)

    gradient = similar(q)
    ld = ldg!(q, gradient; kwargs...)
    H0 = hamiltonian(ld, p)
    isnan(H0) && (H0 = typemax(T))

    avec = zeros(T, J)
    ptries = ones(T, J)
    ss = stepsize .* sqrt.(metric)

    accepted = false
    divergent = false

    qj = similar(q)
    pj = similar(p)
    jf = 0

    for j in 0:(J - 1)
        qj .= q
        ld = ldg!(qj, gradient; kwargs...)
        pj .= p

        a = reduction_factor^j
        ld = leapfrog!(qj, pj, ldg!, gradient, ss ./ a, round(Int, steps * sqrt(a)); kwargs...)

        Hj = hamiltonian(ld, pj)
        isnan(Hj) && (Hj = typemax(T))
        divergent = Hj - H0 > maxdeltaH

        pfac = exp(H0 - Hj)
        if isapprox(position, qj)
            pfac = zero(T)
        end

        jdx = 1:j
        den = prod(1 .- avec[jdx]) * prod(ptries[jdx])
        (num, divergent) = get_num(
            j, qj, -pj, Hj, gradient, ldg!, steps, ss, reduction_factor, maxdeltaH; kwargs...
        )

        prob = pfac * num / den
        if isnan(prob) || isinf(prob)
            prob = zero(T)
        end

        avec[j + 1] = min(1, prob)

        if isnan(pfac) || isinf(pfac)
            ptries[j + 1] = one(T)
        else
            ptries[j + 1] = 1 - avec[j + 1]
        end

        # TODO non-reversible update
        accepted = rand(rng, T) < avec[j + 1]
        if accepted
            jf = j
            break
        end
    end

    if accepted
        position_next .= qj
        momentum .= pj
    else
        position_next .= position
        momentum .*= -1
    end

    return (;
        accepted,
        divergent,
        stepsize,
        steps,
        noise,
            ld,
            retries = jf + 1,
            acceptstat=avec[1],
            finalacceptstat= jf > 0 ? avec[jf + 1] : -1,
            energy=hamiltonian(ld, position_next),
        # momentum=pj,
        # position=qj,
    )
end

function get_num(
    J,
    position,
    momentum,
    H,
    gradient,
    ldg!,
    steps,
    stepsize,
    reduction_factor,
    maxdeltaH;
    kwargs...,
)
    T = eltype(position)
    avec = zeros(T, J)
    ptries = ones(T, J)
    divergent = false

    qj = similar(position)
    pj = similar(momentum)

    for j in 0:(J - 1)
        qj .= position
        ld = ldg!(qj, gradient; kwargs...)
        pj .= momentum

        a = reduction_factor^j
        ld = leapfrog!(qj, pj, ldg!, gradient, stepsize / a, round(Int, steps * sqrt(a)); kwargs...)

        Hj = hamiltonian(ld, pj)
        divergent = Hj - H > maxdeltaH

        pfac = exp(H - Hj)
        if isapprox(position, qj)
            pfac = zero(T)
        end

        prob = zero(T)
        if j > 0
            jdx = 1:j
            den = prod(1 .- avec[jdx]) * prod(ptries[jdx])
            (num, divergent) = get_num(
                j, qj, -pj, Hj, gradient, ldg!, steps, stepsize, reduction_factor, maxdeltaH
            )
            prob = pfac * num / den
        else
            prob = pfac
        end

        if isinf(prob) || isnan(prob)
            return (; num=zero(T), divergent)
        else
            avec[j + 1] = min(1, prob)
        end

        if isinf(pfac) || isnan(pfac)
            ptries[j + 1] = one(T)
        else
            ptries[j + 1] = 1 - avec[j + 1]
        end

        pa = prod(1 .- avec)
        if iszero(pa)
            return (; num=zero(T), divergent)
        end
    end

    return (; num=prod(1 .- avec) * prod(ptries),
            divergent)
end

function xhmc!(
    position,
    position_next,
    momentum,
    ldg!,
    rng,
    dims,
    metric,
    stepsize,
    steps,
    noise,
    K,
    maxdeltaH;
    kwargs...,
)
    T = eltype(position)
    q = copy(position)
    p = noise .* momentum .+ sqrt.(1 .- noise .^ 2) .* randn(rng, T, dims)
    gradient = similar(q)

    ld = ldg!(q, gradient; kwargs...)
    H0 = hamiltonian(ld, p)
    isnan(H0) && (H0 = typemax(T))

    u = rand(rng, T)
    a = zero(T)
    acceptstat = zero(T)
    k = 0
    divergent = false
    H = zero(T)

    while u > a && k < K
        k += 1
        ld = minimal_norm!(
            q, p, ldg!, gradient, stepsize .* sqrt.(metric), steps; kwargs...
        )

        H = hamiltonian(ld, p)
        isnan(H) && (H = typemax(T))
        divergent = H - H0 > maxdeltaH

        a = min(1, max(a, exp(H0 - H)))
        H0 = H
        if k == 1
            acceptstat = a
        end
    end

    accepted = u <= a
    if accepted
        position_next .= q
        momentum .= p
    else
        position_next .= position
        momentum .*= -1
    end

    return (;
        accepted,
        divergent,
        stepsize,
        steps,
        noise,
        ld,
        acceptstat,
        energy=hamiltonian(ld, p), # TODO this should be hamiltonian(ld, position_next)
        momentum=p,
        position=q,
        retries=k,
    )
end

function malt!(
    position,
    position_next,
    ldg!,
    rng,
    dims,
    metric,
    stepsize,
    steps,
    noise,
    maxdeltaH;
    kwargs...,
)
    T = eltype(position)
    q = copy(position)
    p = randn(rng, T, dims)
    gradient = similar(q)

    ld0 = ldg!(q, gradient; kwargs...)
    isnan(ld0) && (ld0 = typemin(T))

    Δ, ld = langevin_trajectory!(
        q, p, ldg!, gradient, stepsize .* sqrt.(metric), steps, noise; kwargs...
    )

    isnan(ld) && (ld = typemin(T))
    Δ += ld - ld0
    divergent = -Δ > maxdeltaH

    a = min(1, exp(Δ))
    accepted = rand(rng, T) < a
    if accepted
        position_next .= q
    else
        position_next .= position
    end

    return (;
        accepted,
        divergent,
        stepsize,
        steps,
        noise,
        ld,
        acceptstat=a,
        energy=hamiltonian(ld, p), # TODO this should be hamiltonian(ld, position_next)
        momentum=p,
        position=q,
    )
end

function hmc!(
    position, position_next, ldg!, rng, dims, metric, stepsize, steps, maxdeltaH; kwargs...
)
    T = eltype(position)
    q = copy(position)
    p = randn(rng, T, dims)
    gradient = similar(q)

    ld = ldg!(q, gradient; kwargs...)
    H0 = hamiltonian(ld, p)
    isnan(H0) && (H0 = typemax(T))

    ld = leapfrog!(
        q, p, ldg!, gradient, stepsize .* sqrt.(metric), steps; kwargs...
    )

    H = hamiltonian(ld, p)
    isnan(H) && (H = typemax(T))
    divergent = H - H0 > maxdeltaH

    a = min(1, exp(H0 - H))
    accepted = rand(rng, T) < a
    if accepted
        position_next .= q
    else
        position_next .= position
    end

    return (;
        accepted,
        divergent,
        stepsize,
        steps,
        ld,
        acceptstat=a,
        energy=H,
        momentum=p,
        position=q,
    )
end

function pghmc!(
    position,
    position_next,
    momentum,
    ldg!,
    rng,
    dims,
    metric,
    stepsize,
    acceptance_probability,
    noise,
    drift,
    damping,
    nonreversible_update,
    maxdeltaH;
    kwargs...,
)
    T = eltype(position)
    q = copy(position)
    p = copy(momentum)
    gradient = similar(q)

    ld = ldg!(q, gradient; kwargs...)
    H0 = hamiltonian(ld, p)
    isnan(H0) && (H0 = typemax(T))

    ld = leapfrog!(q, p, ldg!, gradient, stepsize .* sqrt.(metric), 1; kwargs...)

    H = hamiltonian(ld, p)
    isnan(H) && (H = typemax(T))
    divergence = (H - H0) > maxdeltaH

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

    return (;
        accepted,
        divergence,
        energy,
        noise,
        drift,
        damping,
        stepsize,
        ld,
        acceptstat=a > zero(a) ? one(a) : exp(a),
    )
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

function weighted_mean(x, w)
    T = eltype(x)
    a = zero(T)
    m = zero(T)
    for i in eachindex(x, w)
        wi = w[i]
        a += wi
        m += wi * (x[i] - m) / a
    end
    return m
end

function centered_sum(f, x, mx=zero(x))
    s = zero(eltype(f(first(x))))
    for n in eachindex(x, mx)
        s += f(x[n] - mx[n])
    end
    return s
end

function centered_dot(x, mx, y, my=zero(y))
    s = zero(eltype(dot(first(x), first(y))))
    for n in eachindex(x, y)
        s += (x[n] - mx[n]) * (y[n] - my[n])
    end
    return s
end


function maybe_mean(x)
    T = eltype(x)
    m = zero(T)
    all_negative_ones = true
    n = 0
    for i in eachindex(x)
        if x[i] != -1
            n += 1
            m += (x[i] - m) / n
            all_negative_ones = false
        end
    end
    all_negative_ones && return -1
    return m
end
