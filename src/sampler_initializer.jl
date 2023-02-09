function initialize_sampler!(sampler; kwargs...)
    fields = string.(fieldnames(typeof(sampler)))
    # TODO ugth, quadratic.
    for (k, v) in pairs(kwargs)
        adapter = string(k)
        if any(contains.(adapter, fields))
            set!(sampler, v)
        end
    end
end
