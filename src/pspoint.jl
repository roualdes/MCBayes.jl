"""
Point in general phase space.
"""
struct PSPoint{T<:AbstractFloat} <: AbstractArray{T,1}
    position::Vector{T}
    momentum::Vector{T}
    function PSPoint(position, momentum)
        T = eltype(position)
        if length(position) != length(momentum) || T != eltype(momentum)
            error("position and momentum must have same length and type")
        end
        return new{T}(position, momentum)
    end
end

Base.size(z::PSPoint) = (2 * Base.length(z.position),)

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

function Base.similar(z::PSPoint, ::Type{T}, dims::Dims) where {T}
    return PSPoint(similar(z.position), similar(z.momentum))
end

Base.IndexStyle(::Type{<:PSPoint}) = IndexLinear()
