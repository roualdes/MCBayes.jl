# TODO(ear) use z::PSPoint instead of position, momentum
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
