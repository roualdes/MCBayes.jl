struct DrawsInitializerNoop end

function initialize_draws!(initializer::DrawsInitializerNoop, args... ; kwargs...) end

struct DrawsInitializer end

function initialize_draws!(initializer::DrawsInitializer, draws, args...; kwargs...)
    if haskey(kwargs, :initial_draw)
        draws[1, :, :] .= kwargs[:initial_draw]
    else
        _, dims, chains = size(draws)
        error(
            "With DrawsInitializer, " *
            "supply initial_draw as keyword argument " *
            "with dimensions (dims=$dims, chains=$chains); " *
            "initial_draw = randn(dims, chains).",
        )
    end
end

struct DrawsInitializerRWM end

function initialize_draws!(
    initializer::DrawsInitializerRWM, draws, rngs, ld; radius=2, kwargs...
)
    T = eltype(draws)
    _, dims, chains = size(draws)
    for chain in 1:chains
        draws[1, :, chain] = radius .* (2 .* rand(rngs[chain], T, dims) .- 1)
    end
end

struct DrawsInitializerStan end

function initialize_draws!(initializer::DrawsInitializerStan, draws, rngs, ldg!; kwargs...)
    for chain in axes(draws, 3)
        @views draws[1, :, chain] = stan_initialize_draw(
            draws[1, :, chain], ldg!, rngs[chain]; kwargs...
        )
    end
end

function stan_initialize_draw(position, ldg!, rng; radius=2, attempts=100, kwargs...)
    initialized = false
    a = 0
    T = eltype(position)
    dims = length(position)
    q = copy(position)
    gradient = similar(q)

    while a < attempts && !initialized
        q .= radius .* (2 .* rand(rng, T, dims) .- 1)
        ld = ldg!(q, gradient; kwargs...)

        if isfinite(ld) && !isnan(ld)
            initialized = true
        end

        g = sum(gradient)
        if isfinite(g) && !isnan(g)
            initialized &= true
        end

        a += 1
    end

    if a > attempts
        throw("Failed to find inital values in $(attempts) attempts.")
    end
    return q
end

struct DrawsInitializerAdam end

function initialize_draws!(
    initializer::DrawsInitializerAdam,
    draws,
    rngs,
    ldg!;
    radius=2,
    adam_steps=100,
    number_threads=Threads.nthreads(),
    kwargs...,
)
    T = eltype(draws)
    _, dims, chains = size(draws)
    if haskey(kwargs, :initial_draw)
        draws[1, :, :] .= kwargs[:initial_draw]
    else
        draws[1, :, :] = zeros(T, dims, chains)
    end

    @sync for it in 1:number_threads
        Threads.@spawn for chain in it:number_threads:chains
            gradient = Vector{T}(undef, dims)
            adm = Adam(dims, 100, T; kwargs...)
            for s in 1:adam_steps
                q = draws[1, :, chain]
                ld = ldg!(q, gradient; kwargs...)
                draws[1, :, chain] .-= update!(adm, -gradient, s)
            end
            draws[1, :, chain] .+= randn(rngs[chain], T, dims) .* 0.1
        end
    end
end

struct DrawsInitializerUTurn end

function initialize_draws!(
    initializer::DrawsInitializerUTurn,
    draws,
    rngs,
    ldg!,
    stepsize;
    radius=2,
    steps_uturn=100,
    number_threads=Threads.nthreads(),
    kwargs...,
    )

    T = eltype(draws)
    _, dims, chains = size(draws)
    gradient = zeros(T, dims)

    for chain in 1:chains
        q = draws[1, :, chain]
        ldg!(q, gradient; kwargs...)

        p = randn(rngs[chain], T, dims)
        pb = copy(p)
        qb = copy(q)

        for step in 1:steps_uturn
            leapfrog!(q, p, ldg!, gradient, stepsize, 1)
            if dot(p, q .- qb) < 0  && dot(pb, qb .- q) < 0
                break
            end
        end
        draws[1, :, chain] .= q
    end
end
