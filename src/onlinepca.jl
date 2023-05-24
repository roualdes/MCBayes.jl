# adapted from https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf
struct OnlinePCA{T<:AbstractFloat}
    l::T
    n::Vector{Int}
    m::Vector{T}
    pc::Vector{T}
end

function OnlinePCA(T, d, l)
    return OnlinePCA(l, zeros(Int, 1), zeros(T, d), zeros(T, d))
end

OnlinePCA(d, l = 2.0) = OnlinePCA(Float64, d, l)

function update!(opca::OnlinePCA, x::AbstractMatrix; kwargs...)

    dims, chains = size(x)
    d = length(opca.m)

    if dims != d
        throw(DimensionMismatch("size(x, 1) should equal size(opca.m, 1) == $d"))
    end

    nm = opca.n[1]
    @views for chain in 1:chains
        nm += 1
        opca.m .+= (x[:, chain] .- opca.m) ./ nm
    end

    nv = opca.n[1]
    u = similar(opca.m)
    @views for chain in 1:chains
        nv += 1
        u .= x[:, chain] .- opca.m
        l = opca.l

        if nv == 1
            opca.pc .= x[:, chain]
        else
            f = ((nv - 1 - l) / nv) .* opca.pc
            s = ((1 + l) / nv) .* u .* (u' * opca.pc) ./ norm(opca.pc)
            opca.pc .= f .+ s
        end
    end

    opca.n[1] += chains
end

function reset!(opca::OnlinePCA; kwargs...)
    opca.n .= 0
    opca.m .= 0
    opca.pc .= 0
end
