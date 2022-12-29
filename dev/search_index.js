var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MCBayes","category":"page"},{"location":"#MCBayes","page":"Home","title":"MCBayes","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MCBayes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Stan\nsample!\nStepsizeDualAverage\nStepsizeConstant\nOnlineMoments\nupdate!\nPSPoint","category":"page"},{"location":"#MCBayes.Stan","page":"Home","title":"MCBayes.Stan","text":"Stan(dims, chains, T = Float64; kwargs...)\n\nInitialize Stan sampler object.  The number of dimensions dims and number of chains chains are the only required arguments.  The type T of the ...\n\nOptionally, via keyword arguments, can set the metric, stepsize, seed, maxtreedepth, and maxdeltaH.\n\n\n\n\n\n","category":"type"},{"location":"#MCBayes.sample!","page":"Home","title":"MCBayes.sample!","text":"sample!(sampler::Stan, ldg)\n\nSample with Stan sampler object.  User must provide a function ldg(position; kwargs...) which accepts position::Vector and returns a tuple containing the evaluation of the joint log density function and a vector of the gradient, each evaluated at the argument position.  The remaining keyword arguments attempt to replicate Stan defaults.\n\n\n\n\n\n","category":"function"},{"location":"#MCBayes.StepsizeDualAverage","page":"Home","title":"MCBayes.StepsizeDualAverage","text":"StepsizeDualAverage(initial_stepsize::Vector)\n\nConstruct a stepsize adapter using the dual averaging method by Nesterov 2009, as used in Stan.  The length of initial_stepsize::Vector must be appropriate for the sampling algorithm for which this stepsize adapter will be used.\n\n\n\n\n\n","category":"type"},{"location":"#MCBayes.StepsizeConstant","page":"Home","title":"MCBayes.StepsizeConstant","text":"StepsizeConstant(initial_stepsize::Vector)\n\nConstruct a stepsize adapter for which the stepsize is fixed at it's initial value.\n\n\n\n\n\n","category":"type"},{"location":"#MCBayes.OnlineMoments","page":"Home","title":"MCBayes.OnlineMoments","text":"OnlineMoments(d, c, update = true)\n\nReturns an OnlineMoments struct with mean and variance Matrixes of size (d, c). When update!(om::OnlineMoments, x::Matrix) is called, update determines whether or not any updates will actually be applied.\n\n\n\n\n\n","category":"type"},{"location":"#MCBayes.update!","page":"Home","title":"MCBayes.update!","text":"Adam update.\n\n\n\n\n\nupdate!(om::OnlineMoments, x::Matrix; kwargs...)\n\nUpdate om's mean and variance Matrixes with the data contained in x. The rows of x and om.m (and thus om.v) must match.  The columns of x and om.m must either match or om.m must have only 1 column.  In the latter case, all columns of x will be used to update the same moments om.m and om.v.\n\n\n\n\n\n","category":"function"},{"location":"#MCBayes.PSPoint","page":"Home","title":"MCBayes.PSPoint","text":"Point in general phase space.\n\n\n\n\n\n","category":"type"}]
}