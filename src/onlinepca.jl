# adapted from https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf
struct OnlinePCA{T<:AbstractFloat}
    pc::Matrix{T}
    n::Vector{Int}
    l::T
end

function OnlinePCA(T, d, c, l)
    return OnlinePCA(randn(T, d, c), zeros(Int,c), l)
end

OnlinePCA(d, c, l = 2.0) = OnlinePCA(Float64, d, c, l)

"""
Assumes x is centered and scaled
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
        w = 1 / n
        f = (n - 1 - l) .* w .* opca.pc
        s = (1 + l) .* w .* u .* (u' * opca.pc) ./ norm(opca.pc)
        opca.pc .= f .+ s
    end

    opca.n[1] = n
end

"""
Centers x::AbstractVector using location and scale
"""
function update!(pc, x::AbstractVector, location, scale, n, l, u, f; kwargs...)
    u .= (x .- location) ./ scale
    w = 1 / n
    f .= (n - 1 - l) .* w .* pc
    f .+= (1 + l) .* w .* u .* (u' * pc) ./ norm(pc)
    pc .= f
end


"""
Centers x using location and scale
"""
function update!(opca::OnlinePCA, x::AbstractMatrix, location::AbstractMatrix, scale::AbstractMatrix; kwargs...)
    dims, chains = size(x)
    d, pcas = size(opca.pc)
    _, metrics = size(location)

    if dims != d
        throw(DimensionMismatch("size(x, 1) should equal size(opca.pc, 1) == $d"))
    end

    T = eltype(x)
    u = Vector{T}(undef, dims)
    f = Vector{T}(undef, dims)

    n = opca.n[1]
    l = opca.l
    @views for (pca, metric, chain) in zip(Iterators.cycle(1:pcas), Iterators.cycle(1:metrics), 1:chains)
        n += 1
        u .= (x[:, chain] .- location[:, metric]) ./ scale[:, metric]
        w = 1 / n
        f .= (n - 1 - l) .* w .* opca.pc[:, pca]
        f .+= (1 + l) .* w .* u .* (u' * opca.pc[:, pca]) ./ norm(opca.pc[:, pca])
        opca.pc[:, pca] .= f
    end

    opca.n[1] = n
end

function reset!(opca::OnlinePCA; reset_pc = false, kwargs...)
    opca.n .= 0
    if reset_pc
        randn!(opca.pc)
    end
end
