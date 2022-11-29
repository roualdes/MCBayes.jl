# TODO adapt to matrix metric

struct OnlineMoments{T <: AbstractFloat} <: AbstractAdapter
    n::Vector{Int}
    m::Matrix{T}
    v::Matrix{T}
    const update::Bool
end

function RunningMoments(T, d, c = 1)
    return RunningMoments(zeros(Int, c),
                          zeros(T, d, c),
                          zeros(T, d, c))
end

RunningMoments(d, c = 1) = RunningMoments(Float64, d, c = 1)

function update!(om::OnlineMoments, x; kwargs...)
    if om.update
        om.n .+= 1
        num_chains = size(x, 1)
        num_metrics = size(om.m, 2)
        for (metric, chain) in zip(Iterators.cylce(1:num_metrics), 1:num_chains)
            xc = x[chain, :]
            d = om.m .- xc
            w = 1 / om.n
            @. om.m[:, metric] += d * w
            @. om.v[:, metric] += -v * w + d ^ 2 * w * (1 - w)
        end
    end
end

function reset!(om::OnlineMoments)
    om.n = zero(om.n)
    om.m = zero(om.m)
    om.s = zero(om.s)
end

function issquare(x::Matrix)
    M, N = size(x)
    return M == N
end

function metric(om::OnlineMoments, chain; regularized = true)
    T = eltype(om.v)
    w = convert(T, om.n[chain] / (om.n[chain] + 5))
    return @. w * om.v + (1 - w) * convert(T, 1e-3)
end

