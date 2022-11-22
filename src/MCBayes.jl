module MCBayes

function sample(method::Symbol, initialdraw, lp, data; kwargs...)
    sampler = initialize_sampler(method, initialdraw; kwargs...) # TODO
    draws = zeros(sampler.T, sampler.D, sampler.C)
    if kwargs[:init_draws]
        kwargs[:init_draws](sampler, draws, data) # TODO: need example
    else
        if kwargs[:init_draws]
            initialize_draws!(sampler, draws)
        end
    end
    for t in 1:T
        transition!(sampler, t, draws, data)
    end
    return draws, sampler
end


function initialize_draws!(sampler, draws)
    u = rand(sampler.D, sampler.chains)
    b = kwargs[:init_ub]
    a = kwargs[:init_lb]
    draws[1, :, :] .= (b - a) .* u .- a
end


end
