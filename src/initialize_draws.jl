function initialize_draws!(method::Symbol, draws, rng, ldg; kwargs...)
    return initialize_draws!(Val{method}(), draws, rng, ldg; kwargs...)
end

function initialize_draws!(::Val{:stan}, draws, rng, ldg; kwargs...)
    for chain in axes(draws, 3)
        @views draws[1, :, chain] = stan_initialize_draw(
            draws[1, :, chain], ldg, rng[chain]; kwargs...
        )
    end
end

function stan_initialize_draw(position, ldg, rng; radius=2, attempts=100, kwargs...)
    initialized = false
    a = 0
    T = eltype(position)
    dims = length(position)
    q = copy(position)

    while a < attempts && !initialized
        q .= radius .* (2 .* rand(rng, T, dims) .- 1)
        ld, gradient = ldg(q; kwargs...)

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

# TODO needs a second look
# function initialize_draws!(::Val{:sga}, draws, rng, ldg;
#                            steps = 100,
#                            number_threads = Threads.nthreads(),
#                            kwargs...)
#     chains = size(draws, 3)
#     @sync for it in 1:number_threads
#         Threads.@spawn for chain in it:number_threads:chains
#             for s in 1:steps
#                 _ = ldg(draws[1, :, chain], gradients[:, chain])
#                 draws[1, :, chain] .-= update!(initialize_draws_adam, gradients[:, chain], s)
#             end
#         end
#     end
# end

function initialize_draws!(
    ::Val{:none}, draws::AbstractArray, gradients, rng, ldg; kwargs...
)
    if haskey(kwargs, :initial_draw)
        draws[1, :, :] .= initial_draw
    else
        _, dims, chains = size(draws)
        error(
            "With draws_initializer = :none, " *
            "supply initial_draw with dimensions $dims by $chains",
        )
    end
end
