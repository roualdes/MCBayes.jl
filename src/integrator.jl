function leapfrog!(position, momentum, ldg, gradient, stepsize, steps; kwargs...)
    ld = zero(eltype(position))
    @. momentum += stepsize * gradient / 2

    for step in 1:steps
        @. position += stepsize * momentum
        ld, gradient = ldg(position; kwargs...)
        if step != steps
            @. momentum += stepsize * gradient
        end
    end

    @. momentum += stepsize * gradient / 2
    return ld, gradient
end

function langevin_trajectory!(position, momentum, ldg, stepsize, steps, noise; kwargs...)
    T = eltype(position)
    Δ = zero(T)
    ld = zero(T)
    ld_original, gradient = ldg(position; kwargs...)
    momentum_previous = copy(momentum)

    for step in 1:steps
        momentum .= noise .* momentum .+ sqrt.(1 .- noise .^ 2) .* randn(T, size(momentum))
        ld, gradient = leapfrog!(position, momentum, ldg, gradient, stepsize, 1; kwargs...)

        energy_difference = (momentum_previous' * momentum_previous - momentum' * momentum) / 2
        if isnan(energy_difference)
            Δ = typemin(T)
            break
        end
        Δ += energy_difference

        momentum_previous .= momentum
    end

    Δ += ld - ld_original

    return Δ, ld
end
