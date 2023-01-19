function initialize_sampler!(sampler; kwargs...)
    adapters = string.(keys(kwargs))
    fields = string.(fieldnames(typeof(sampler)))

    # TODO ugth, quadratic.
    for (a, adapter) in pairs(kwargs)
        if any(contains.(adapter, fields))
            set!(sampler, adapter)
        end
    end
end
