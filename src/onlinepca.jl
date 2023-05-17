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
    # TODO fill in from https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf
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
            opca.pc .= u
        else
            f = (nv - 1 - l) .* opca.pc ./ nv
            s = (1 + l) .* u .* (u' * opca.pc) ./ nv
            opca.pc .= f .+ s ./ norm(opca.pc)
        end
    end

    opca.n[1] += chains
end
