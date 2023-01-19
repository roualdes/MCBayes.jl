function initialize_sampler!(sampler;
                             stepsize_adapter,
                             trajectorylength_adapter,
                             metric_adapter,
                             damping_adapter,
                             noise_adapter,
                             drift_adapter)

end

    function f(; kwargs...)
    fields = string.(fieldnames(typeof(Stan(10))))
    adapters = string.(keys(kwargs))
    # TODO must this be quadratic?
    for field in fields
        if any(occursin.(field, adapters))
            println(field)
        end
    end
end


f(; noise_adapter = 1, metric_adapter = 2, stepsize_adapter = 3)
    # TODO change all set_(adapter)! methods to set!, so that this works
    # don't worry, the adapter is encoded in the type
    adapters = string.((:stepsize, :trajectorylength, :metric, :damping, :noise, :drift))
    fields = fieldnames(typeof(sampler))

    for adapter in adapters
        if adapter in fields

            set!(sampler, adapter)
        end
    end
end
