mutable struct OnlineMoments{T <: AbstractFloat} <: AbstractAdapter
    n::Int
    m::Matrix{T}
    s::Matrix{T}
    const update::Bool
end

function RunningMoments(T, d, c = 1)
    return RunningMoments(zero(Int),
                          zeros(T, d, c),
                          zeros(T, d, c))
end

RunningMoments(d, c = 1) = RunningMoments(Float64, d, c = 1)

function update!(om::OnlineMoments, x; kwargs...)
    if om.update
        om.n += 1
        num_chains = size(x, 1)
        num_metrics = size(om.m, 2)
        for (metric, chain) in zip(Iterators.cylce(1:num_metrics), 1:num_chains)
            xc = x[chain, :]
            d = rm.m .- xc
            w = 1 / om.n
            @. rm.m[:, metric] += d * w
            @. rm.v[:, metric] += -v * w + d ^ 2 * w * (1 - w)
        end
    end
end

function reset!(om::OnlineMoments)
    om.n = zero(om.n)
    om.m = zero(om.m)
    om.s = zero(om.s)
end
