# Adapted from
# https://github.com/stan-dev/rstan
# Copyright (C) 2012, 2013, 2014, 2015, 2016, 2017, 2018 Trustees of Columbia University
# Copyright (C) 2018, 2019 Aki Vehtari, Paul Bürkner

# References

# Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki
# Vehtari and Donald B. Rubin (2013). Bayesian Data Analysis, Third
# Edition. Chapman and Hall/CRC.

# Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, and
# Paul-Christian Bürkner (2019). Rank-normalization, folding, and
# localization: An improved R-hat for assessing convergence of
# MCMC. arXiv preprint arXiv:1903.08008
#
# https://github.com/stan-dev/posterior/blob/master/R/convergence.R

# TODO(ear) all of this can/should? be adapted to follow the type of the input
# or does leaving it all untyped just work?
# TODO(ear) organize this file

# TODO(ear) add docs to all that expect Array{3, T} to have iterations x dimensions x chains

# TODO(ear) uses FFTW; wait on
# https://github.com/JuliaLang/julia/pull/47040
# and the related PR
# https://github.com/JuliaLang/Pkg.jl/pull/3216
# function autocovariance(x)
#     N = length(x)
#     Mt2 = 2 * fft_nextgoodsize(N)
#     yc = x .- mean(x)
#     append!(yc, repeat([0.0], Mt2 - N))
#     t = bfft(yc)
#     ac = bfft(conj(t) .* t)
#     return real(ac)[1:N] ./ (N * N * 2)
# end
# until then here's a O(N^2) implementation
function autocovariance(x)
    N = length(x)
    xc = x .- (sum(x) / N)
    ac = zero(xc)
    for n in 1:N
        for i in 1:(N - n + 1)
            ac[n] += xc[i] * xc[i + n - 1]
        end
    end
    return ac ./ N
end

function autocorrelation(x)
    ac = autocovariance(x)
    return ac / ac[1]
end

function fft_nextgoodsize(N)
    N <= 2 && return 2

    while true
        m = convert(Float64, N)
        while mod(m, 2) == 0
            m /= 2
        end
        while mod(m, 3) == 0
            m /= 3
        end
        while mod(m, 5) == 0
            m /= 5
        end
        if m <= 1
            return N
        end
        N += 1
    end
end

function _ess(x)
    niterations, nchains = size(x)

    (niterations < 3 || any(isnan.(x))) && return NaN
    any(isinf.(x)) && return NaN
    isconstant(x) && return NaN

    acov = mapslices(autocovariance, x; dims=1) # m:iterations x chains
    chain_mean = mean(x; dims=1)                 # v:chains
    mean_var = mean(acov[1, :]) * niterations / (niterations - 1) # scalar
    var_plus = mean_var * (niterations - 1) / niterations         # scalar

    nchains > 1 && (var_plus += var(chain_mean))

    rhohat = zeros(niterations)
    t = 0
    rhohat_even = 1.0
    rhohat[t + 1] = rhohat_even
    rhohat_odd = 1 - (mean_var - mean(acov[t + 2, :])) / var_plus
    rhohat[t + 2] = rhohat_odd

    while t < niterations - 5 &&
              !isnan(rhohat_even + rhohat_odd) &&
              rhohat_even + rhohat_odd > 0
        t += 2
        rhohat_even = 1 - (mean_var - mean(acov[t + 1, :])) / var_plus
        rhohat_odd = 1 - (mean_var - mean(acov[t + 2, :])) / var_plus

        if rhohat_even + rhohat_odd >= 0
            rhohat[t + 1] = rhohat_even
            rhohat[t + 2] = rhohat_odd
        end
    end

    max_t = t
    # this is used in the improved estimate
    rhohat_even > 0 && (rhohat[max_t + 1] = rhohat_even)

    # Geyer's initial monotone sequence
    t = 0
    while t <= max_t - 4
        t += 2
        if rhohat[t + 1] + rhohat[t + 2] > rhohat[t - 1] + rhohat[t]
            rhohat[t + 1] = (rhohat[t - 1] + rhohat[t]) / 2
            rhohat[t + 2] = rhohat[t + 1]
        end
    end

    ess = nchains * niterations
    # Geyer's truncated estimate
    # it's possible max_t == 0; 1:0 does not behave like in R
    τ = -1 + 2.0 * sum(rhohat[1:max(1, max_t)]) + rhohat[max_t + 1]
    # Improved estimate reduces variance in antithetic case
    τ = max(τ, 1.0 / log10(ess))
    return ess / τ
end

function _rhat(x)
    any(isnan.(x)) && return NaN
    any(isinf.(x)) && return NaN
    isconstant(x) && return NaN

    niterations, nchains = size(x)

    chain_mean = mean(x; dims=1)
    chain_var = var(x; dims=1)

    var_between = niterations * var(chain_mean)
    var_within = mean(chain_var)

    return sqrt((var_between / var_within + niterations - 1) / niterations)
end

function _rhat_basic(x)
    return _rhat(splitchains(x))
end

function rhat_basic(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    return mapslices(_rhat_basic, x; dims=(1, 3))
end

function rhat_basic(x::AbstractVecOrMat)
    return mapslices(_rhat_basic, x; dims = 1)
end

function isconstant(x, tol=sqrt(eps(Float64)))
    mn, mx = extrema(@views x[:])
    return isapprox(mn, mx; rtol=tol)
end

function splitchains(x)
    niterations = size(x, 1)
    niterations < 2 && return x

    if isodd(niterations)
        niterations -= 1
    end

    ub_lowerhalf = div(niterations, 2)
    lb_secondhalf = ub_lowerhalf + 1
    return hcat(x[1:ub_lowerhalf, :], x[lb_secondhalf:niterations, :])
end

function fold(x)
    return abs.(x .- median(x))
end

function tiedrank(x)
    # Adapted from StatsBase
    # https://github.com/JuliaStats/StatsBase.jl/blob/master/src/ranking.jl

    # Copyright (c) 2012-2016: Dahua Lin, Simon Byrne, Andreas Noack,
    # Douglas Bates, John Myles White, Simon Kornblith, and other contributors.

    # Permission is hereby granted, free of charge, to any person obtaining
    # a copy of this software and associated documentation files (the
    # "Software"), to deal in the Software without restriction, including
    # without limitation the rights to use, copy, modify, merge, publish,
    # distribute, sublicense, and/or sell copies of the Software, and to
    # permit persons to whom the Software is furnished to do so, subject to
    # the following conditions:
    #
    # The above copyright notice and this permission notice shall be
    # included in all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    # NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
    # LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
    # OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    # WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    if length(size(x)) > 1
        x = x[:]
    end

    n = length(x)
    p = sortperm(x)
    rks = zeros(n)

    if n > 0
        v = x[p[1]]

        s = 1  # starting index of current range
        e = 2  # pass-by-end index of current range
        while e <= n
            cx = x[p[e]]
            if cx != v
                # fill average rank to s : e-1
                ar = (s + e - 1) / 2
                for i in s:(e - 1)
                    rks[p[i]] = ar
                end
                # switch to next range
                s = e
                v = cx
            end
            e += 1
        end

        # the last range (e == n+1)
        ar = (s + n) / 2
        for i in s:n
            rks[p[i]] = ar
        end
    end

    return rks
end

# function quantile_normal(p, l, s)
#     q = SpecialFunctions.erfinv(2 * p - 1)
#     return l + s * sqrt(2) * q
# end

# function zscale(x)
#     r = tiedrank(vec(x))
#     z = quantile_normal.((r .- 0.375) ./ (length(x) + 0.25), 0, 1) # Blom (1958) (6.10.3)
#     if length(size(x)) > 1
#         z = reshape(z, size(x))
#     end
#     return z
# end

# function rhat(x::AbstractArray{T,3}) where {T<:AbstractFloat}
#     return mapslices(max_rhat, x; dims=(1, 3))
# end

# function rhat(x::AbstractVecOrMat)
#     return mapslices(max_rhat, x; dims=1)
# end

# function max_rhat(x)
#     rhat_bulk = rhat_basic(zscale(splitchains(x)))
#     rhat_tail = rhat_basic(zscale(splitchains(fold(x))))
#     return max(rhat_bulk, rhat_tail)
# end

# function _ess_bulk(x)
#     return _ess(zscale(splitchains(x)))
# end

# function ess_bulk(x::AbstractArray{T,3}) where {T<:AbstractFloat}
#     return mapslices(_ess_bulk, x; dims=(1, 3))
# end

# function ess_bulk(x::AbstractVecOrMat)
#     return mapslices(_ess_bulk, x; dims=1)
# end

function _ess_tail(x)
    I05 = x .<= quantile(x[:], 0.05)
    q05_ess = _ess(splitchains(I05))
    I95 = x .<= quantile(x[:], 0.95)
    q95_ess = _ess(splitchains(I95))
    return min(q05_ess, q95_ess)
end

function ess_tail(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    return mapslices(_ess_tail, x; dims=(1, 3))
end

function ess_tail(x::AbstractVecOrMat)
    return mapslices(_ess_tail, x; dims=1)
end

function _ess_quantile(x, prob=0.5)
    @assert prob >= 0 && prob <= 1
    I = x .<= quantile(x[:], prob)
    return _ess(splitchains(I))
end

function ess_quantile(x::AbstractArray{T,3}, prob=0.5) where {T<:AbstractFloat}
    return mapslices(x -> _ess_quantile(x, prob), x; dims=(1, 3))
end

function ess_quantile(x::AbstractVecOrMat, prob=0.5)
    return mapslices(x -> _ess_quantile(x, prob), x; dims=1)
end

function _ess_mean(x)
    return _ess(splitchains(x))
end

function ess_mean(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    return mapslices(_ess_mean, x; dims=(1, 3))
end

function ess_mean(x::AbstractVecOrMat)
    return mapslices(_ess_mean, x; dims=1)
end

function _ess_sq(x)
    return _ess(splitchains(x .^ 2))
end

function ess_sq(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    return mapslices(_ess_sq, x; dims=(1, 3))
end

function ess_sq(x::AbstractVecOrMat)
    return mapslices(_ess_sq, x; dims=1)
end

function _ess_f(f, x)
    return _ess(splitchains(f.(x)))
end

function ess_f(f, x::AbstractArray{T,3}) where {T<:AbstractFloat}
    return mapslices(x -> _ess_f(f, x), x; dims=(1, 3))
end

function ess_f(x::AbstractVecOrMat)
    return mapslices(x -> _ess_f(f, x), x; dims=1)
end

function _ess_std(x)
    return _ess(splitchains(abs.(x .- mean(x))))
end

function ess_std(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    return mapslices(_ess_std, x; dims=(1, 3))
end

function ess_std(x::AbstractVecOrMat)
    return mapslices(_ess_std, x; dims=1)
end

# TODO(ear) uses Distributions; wait on
# https://github.com/JuliaLang/julia/pull/47040
# and the related PR
# https://github.com/JuliaLang/Pkg.jl/pull/3216
# function mcse_quantile(x, prob::Real)
#     ess = ess_quantile(x, prob)
#     p = [0.1586553; 0.8413447]
#     B = Beta(ess * prob + 1, ess * (1 - prob) + 1)
#     a = quantile.(B, p)
#     ssims = sort(x[:])
#     S = length(ssims)
#     th1 = ssims[convert(Int64, max(floor(a[1] * S), 1))]
#     th2 = ssims[convert(Int64, min(ceil(a[2] * S), S))]
#     return (th2 - th1) / 2
# end

function mcse_mean(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    return std(x; dims=(1, 3)) ./ sqrt.(ess_mean(x))
end

function mcse_mean(x::AbstractVecOrMat)
    return std(x; dims=1) ./ sqrt.(ess_mean(x))
end

function mcse_std(x)
    sims_c = x .- mean(x)
    ess = ess_mean(abs.(sims_c))
    Evar = mean(sims_c .^ 2)
    varvar = (mean(sims_c .^ 4) .- Evar .^ 2) ./ ess
    varsd = varvar ./ Evar ./ 4
    return sqrt.(varsd)
end

function mad(x)
    return 1.4826 * median(abs.(x .- median(x)))
end
