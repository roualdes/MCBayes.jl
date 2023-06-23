function leapfrog!(position, momentum, ldg, gradient, stepsize, steps; kwargs...)
    ld = zero(eltype(position))
    @. momentum += 0.5 * stepsize * gradient

    for step in 1:steps
        @. position += stepsize * momentum
        ld, gradient = ldg(position; kwargs...)
        if step != steps
            @. momentum += stepsize * gradient
        end
    end

    @. momentum += 0.5 * stepsize * gradient
    return ld, gradient
end

function langevin_trajectory!(
    position, momentum, ldg, gradient, stepsize, steps, noise; kwargs...
)
    T = eltype(position)
    Δ = zero(T)
    ld = zero(T)
    ξ = randn(length(momentum), steps)

    for step in 1:steps
        @. @views momentum = noise * momentum + sqrt(1 - noise^2) * ξ[:, step]
        Δ += 0.5 * (momentum' * momentum)
        ld, gradient = leapfrog!(position, momentum, ldg, gradient, stepsize, 1; kwargs...)
        Δ -= 0.5 * (momentum' * momentum)
    end

    return Δ, ld
end


# adapted from
# https://arxiv.org/pdf/hep-lat/0505020.pdf eq. 20
# https://github.com/JaimeRZP/MicroCanonicalHMC.jl/blob/master/src/integrators.jl
const lambda = 0.1931833275037836

function minimal_norm!(
    position, momentum, ldg, gradient, stepsize, steps; kwargs...
)

    ld = zero(eltype(position))

    for step in 1:steps
        @. momentum += lambda * stepsize * gradient
        @. position += 0.5 * stepsize * momentum
        ld, gradient = ldg(position; kwargs...)
        @. momentum += (1 - 2 * lambda) * stepsize * gradient
        @. position += 0.5 * stepsize * momentum
        ld, gradient = ldg(position; kwargs...)
        @. momentum += lambda * stepsize * gradient
    end
    return ld, gradient
end
