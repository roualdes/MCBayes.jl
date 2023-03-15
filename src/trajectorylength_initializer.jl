# TODO make structs to dispatch on; don't use symbols
function initialize_trajectorylength!(method::Symbol, adapter, stepsize; kwargs...)
    return initialize_trajectorylength!(Val{method}(), adapter, stepsize; kwargs...)
end

function initialize_trajectorylength!(::Val{:stan}, adapter, stepsize; kwargs...) end

function initialize_trajectorylength!(::Val{:sga}, adapter, stepsize; kwargs...)
    return adapter.tracjectorylength = stepsize
end
