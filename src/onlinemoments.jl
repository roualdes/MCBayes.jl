struct OnlineMoments{T<:AbstractFloat}
    n::Vector{Int}
    m::Matrix{T}
    v::Matrix{T}
end

function OnlineMoments(T, d, c)
    return OnlineMoments(zeros(Int, c), zeros(T, d, c), zeros(T, d, c))
end

"""
    OnlineMoments(d, c, update = true)

Returns an OnlineMoments struct with mean and variance `Matrix`es of
size (d, c). When `update!(om::OnlineMoments, x::Matrix)` is called, update
determines whether or not any updates will actually be applied.

"""
OnlineMoments(d, c=1) = OnlineMoments(Float64, d, c)

"""
    update!(om::OnlineMoments, x::Matrix; kwargs...)

Update om's mean and variance `Matrix`es with the data contained in x.
The rows of x and om.m (and thus om.v) must match.  The columns of x
and om.m must either match or om.m must have only 1 column.  In the
latter case, all columns of x will be used to update the same moments
om.m and om.v.

"""
function update!(om::OnlineMoments, x::AbstractMatrix; kwargs...)
    dims, chains = size(x)
    d, metrics = size(om.m)

    if dims != d
        throw(DimensionMismatch("size(x, 1) should equal size(om.m, 1) == $d"))
    end

    if chains != metrics && metrics != 1
        throw(
            DimensionMismatch(
                "size(x, 2) should equal size(om.m, 2) == $metrics or equal 1"
            ),
        )
    end

    @views for (metric, chain) in zip(Iterators.cycle(1:metrics), 1:chains)
        om.n[metric] += 1
        m = x[:, chain] .- om.m[:, metric]
        w = 1 / om.n[metric]
        @. om.m[:, metric] += m * w
        @. om.v[:, metric] += -om.v[:, metric] * w + m^2 * w * (1 - w)
    end
end

function reset!(om::OnlineMoments; kwargs...)
    om.n .= 0
    om.m .= 0
    om.v .= 0
end

function optimum(om::OnlineMoments; regularized=true, kwargs...)
    T = eltype(om.v)
    if om.n[1] > 1
        v = if regularized
            w = reshape(convert.(T, om.n ./ (om.n .+ 5)), 1, :)
            @. w * om.v + (1 - w) * convert(T, 1e-3)
        else
            om.v
        end
        return v
    else
        return ones(T, size(om.v))
    end
end
