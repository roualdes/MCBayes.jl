struct OnlinePCA{T<:AbstractFloat}
    k::Int
    l::T
    n::Vector{Int}
    m::Vector{T}
    ev::Matrix{T}
end

function OnlinePCA(T, d, k, l)
    return OnlinePCA(k, l, zeros(Int, 1), zeros(T, d), zeros(T, d, k))
end

OnlinePCA(d, k = 1, l = 2) = OnlinePCA(Float64, d, k, l)

update!(opca::OnlinePCA, x::Matrix; kwargs...)
# TODO fill in from https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf
end
