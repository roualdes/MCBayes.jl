# TODO(ear) use z::PSPoint instead of position, momentum?
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
    ld_previous, gradient = ldg(position; kwargs...)
    position_previous = copy(position)
    gradient_previous = copy(gradient)

    for step in 1:steps
        momentum .= noise .* momentum .+ sqrt.(1 .- noise .^ 2) .* randn(T, size(momentum))
        ld, gradient = leapfrog!(position, momentum, ldg, gradient, stepsize, 1; kwargs...)
        momentum .= noise .* momentum .+ sqrt.(1 .- noise .^ 2) .* randn(T, size(momentum))

        ed = langevin_trajectory_energy_difference(position_previous,
                                                   ld_previous,
                                                   gradient_previous,
                                                   position,
                                                   ld,
                                                   gradient,
                                                   stepsize)
        if isnan(ed)
            Δ = typemin(T)
            break
        end
        Δ += ed

        position_previous .= position
        ld_previous = ld
        gradient_previous .= gradient
    end

    return Δ, ld
end
