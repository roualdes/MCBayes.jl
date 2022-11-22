abstract type AbstractSGA{T} <: AbstractSamlper{T} end

mutable struct ChEES{T <: AbstractFloat} <: AbstractSGA
    γ::T
    adam_ε::Adam{T}
    adam_τ::Adam{T}
    metric::Vector{T}
    om::OnlineMoments{T}
    const dims::Int
    const chains::Int
    const maxdeltaH::T
    const maxtreedepth::Int
end

mutable struct SNAPER{T <: AbstractFloat} <: AbstractSGA
    γ::T
    adam_ε::Adam{T}
    adam_τ::Adam{T}
    metric::Vector{T}
    om::OnlineMoments{T}
    const dims::Int
    const chains::Int
    const maxdeltaH::T
    const maxtreedepth::Int
end

function transition!(sga::AbstractSGA, i, draws; kwargs...)
    kernel!(sga, i, draws)
    nt = kwargs[:nt]
    @sync for it in 1:nt
        Threads.@spawn for c in it:nt:kwargs[:chains]
            info = kernel!(sga, i, draws; kwargs...)
            update_info!(sga, i, info)
        end
    end
    adapt_chains!(sga, i, draws)
end

function adapt_chains!(sga::SGA, i, draws; kwargs...)
    # copy from MCBayes/src/sga.jl
    # ...
    adapt_metric!(sga, i, draws; kwargs...)
end

function adapt_metric!(sga::SGA, i, draws; kwargs...)
    update!(sga.om, draws[i, :, :]; kwargs...)
    η = i ^ sga.γ               # ημ = 1 / (ceil(numchains * i / sga.κ) + 1)
    sga.metric .= η .* sga.om.v .+ (1 - η) .* sga.metric
end
