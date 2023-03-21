using MCBayes
using Test
using Artifacts
using Statistics
using Serialization
using LinearAlgebra
BLAS.set_num_threads(1)

include("test_onlinemoments.jl")
include("test_pspoint.jl")
include("test_typestability.jl")
include("test_convergence.jl")
include("test_samplers.jl")
