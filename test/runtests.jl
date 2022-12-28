using MCBayes
using Test
using Artifacts
using Statistics
using BridgeStan

const BS = BridgeStan
bsdir = joinpath(homedir(), "./julia/dev/bridgestan")
BS.set_bridgestan_path!(bsdir)

include("test_onlinemoments.jl")
include("test_pspoint.jl")
include("test_typestability.jl")
include("test_convergence.jl")
include("test_stan.jl")
