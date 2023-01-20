.PHONY: test pretty doc

test:
	julia --threads=2 -e 'using Pkg; Pkg.activate("."); Pkg.test();'

pretty:
	julia -e 'using JuliaFormatter; format(".", verbose = true);'

doc:
	julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("docs/make.jl");'
