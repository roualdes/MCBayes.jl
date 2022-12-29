using MCBayes
using Test
using Artifacts
using Statistics
using BridgeStan

const BS = BridgeStan
cwd = if get(ENV, "CI", "false") == "true"
    pwd()
else
    homedir()
end
bsdir = joinpath(cwd, "bridgestan")
BS.set_bridgestan_path!(bsdir)

include("test_onlinemoments.jl")
include("test_pspoint.jl")
include("test_typestability.jl")
include("test_convergence.jl")
include("test_stan.jl")
