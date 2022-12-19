function initialize_trajectorylength!(method::Symbol, adapter, stepsize; kwargs...)
    initialize_trajectorylength!(Val{method}(), adapter, stepsize; kwargs...)
end

function initialize_trajectorylength!(::Val{:stan}, adapter, stepsize; kwargs...)
end


function initialize_trajectorylength!(::Val{:sga}, adapter, stepsize; kwargs...)
    adapter.tracjectorylength = stepsize
end
    
