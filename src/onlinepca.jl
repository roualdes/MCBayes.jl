# adapted from https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf
struct OnlinePCA{T<:AbstractFloat}
    l::T
    n::Vector{Int}
    pc::Vector{T}
end

function OnlinePCA(T, d, l)
    return OnlinePCA(l, zeros(Int, 1), randn(T, d))
end

OnlinePCA(d, l = 2.0) = OnlinePCA(Float64, d, l)

"""
Assumes x is centered
"""
function update!(opca::OnlinePCA, x::AbstractMatrix; kwargs...)

    dims, chains = size(x)
    d = length(opca.pc)

    if dims != d
        throw(DimensionMismatch("size(x, 1) should equal size(opca.pc, 1) == $d"))
    end

    n = opca.n[1]
    l = opca.l
    @views for chain in 1:chains
        n += 1
        u = x[:, chain]

        f = ((n - 1 - l) / n) .* opca.pc
        s = ((1 + l) / n) .* u .* (u' * opca.pc) ./ norm(opca.pc)
        opca.pc .= f .+ s
    end

    opca.n[1] = n
end

function reset!(opca::OnlinePCA; kwargs...)
    opca.n .= 0
    randn!(opca.pc)
end
