"""
Point in general phase space.
"""
struct PSPoint{T} <: AbstractArray{T, 1}
    position::Vector{T}
    momentum::Vector{T}
end



Base.size(z::PSPoint) = (2 * Base.length(z.position),)

Base.IndexStyle(::Type{<:PSPoint}) = IndexLinear()

function Base.getindex(z::PSPoint, ind::Int)
    l = length(z.position)
    if ind <= length(z.position)
        z.position[ind]
    else
        ind -= l
        z.momentum[ind]
    end
end

function Base.setindex!(z::PSPoint, val, ind::Int)
    l = length(z.position)
    if ind <= l
        z.position[ind] = val
    else
        ind -= l
        z.momentum[ind] = val
    end
end

Base.similar(z::PSPoint{T}) where {T} = PSPoint{T}(similar(z.position), similar(z.momentum))

Base.BroadcastStyle(::Type{<:PSPoint}) = Broadcast.ArrayStyle{PSPoint}()

function Base.similar(bc::Broadcast.Broadcasted{<:PSPoint}, ::Type{T}) where T
    return PSPoint{T}(Base.similar(z.position), Base.similar(z.momentum))
end

Base.similar(z::PSPoint, ::Type{T}, dims::Dims) where {T} = SparseArray(T, dims)



Base.copy(z::PSPoint{T}) where {T} = PSPoint{T}(Base.copy(z.position), Base.copy(z.momentum))

Base.eltype(z::PSPoint{T}) where {T} = T

Base.show(io::IO, z::PSPoint) = print(io, "$([z.position; z.momentum])")

function Base.show(io::IO, ::MIME"text/plain", z::PSPoint)
    T = eltype(z)
    print(io, "$(length(z))-element Phase Space Point{$(T)}\nposition: $(z.position)\nmomentum: $(z.momentum)")
end

position(z::PSPoint) = z.position

mometnum(z::PSPoint) = z.momentum
