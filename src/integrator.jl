function integrate!(method::Symbol, ldg, position, momenta, gradient, stepsize, steps; kwargs...)
    integrate!(Val{method}(), ldg, position, momenta, gradient, stepsize, steps; kwargs...)
end

function integrate!(::Val{:leapfrog}, ldg, position, momenta, gradient, stepsize, steps;
                    kwargs...)
    T = eltype(stepsize)
    onehalf = convert(T, 0.5)
    ld = zero(T)
    @. momenta -= onehalf * stepsize * gradient

    for step in 1:steps
        @. position += stepsize * momenta
        ld = ldg(position, gradient; kwargs...)
        if step != steps
            @. momenta -= stepsize * gradient
        end
    end

    @. momenta -= stepsize * gradient
    return ld
end
