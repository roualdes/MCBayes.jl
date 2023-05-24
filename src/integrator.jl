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

function langevin_trajectory!(position, momentum, ldg, gradient, stepsize, steps, noise; kwargs...)
    T = eltype(position)
    Δ = zero(T)
    ld = zero(T)
    momentum_previous = copy(momentum)
    ξ = similar(momentum)

    for step in 1:steps
        randn!(ξ)
        momentum .= noise .* momentum .+ sqrt.(1 .- noise .^ 2) .* ξ
        ld, gradient = leapfrog!(position, momentum, ldg, gradient, stepsize, 1; kwargs...)

        Δ += (momentum_previous' * momentum_previous - momentum' * momentum) / 2
        momentum_previous .= momentum
    end

    return Δ, ld
end
