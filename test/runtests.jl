using MCBayes
using Test
using Artifacts
using Statistics
using BridgeStan

const BS = BridgeStan
bsdir = if get(ENV, "CI", "false") == "true"
    "bridgestan"
else
    joinpath(homedir(), "bridgestan")
end
BS.set_bridgestan_path!(bsdir)

include("test_onlinemoments.jl")
include("test_pspoint.jl")
include("test_typestability.jl")
include("test_convergence.jl")
include("test_stan.jl")
