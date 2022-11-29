function integrate(method::Symbol, ldg, position, momenta, gradient, stepsize, steps; kwargs...)
    integrate(Val{method}(), ldg, position, momenta, gradient, stepsize, steps; kwargs...)
end


function integrate(::Val{:leapfrog}, ldg, position, momenta, gradient, stepsize, steps;
                   kwargs...)
    onehalf = oftype(stepsize, 0.5)
    @. momenta += onehalf * stepsize * gradient

    for step in 1:steps
        @. position += stepsize * momenta
        lp = ldg(position, gradient; kwargs...)
        if step != steps
            @. momenta += stepsize * gradient
        end
    end

    @. momenta += stepsize * gradient
end



