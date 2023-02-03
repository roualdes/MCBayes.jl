# Design

## User Interface

From a user's perspective, Bayesian inference requires a log joint
density function `ldg`, which may or may not depend on some data.
`MCBayes` further requires the dimension of the input to the log
density gradient.  Consider the log joint density function of the
Gaussian distribution's density function.

```julia
function ldg(q; kwargs...)
    return -q' * q / 2, -q
end
dims = 10
q = randn(dims)
ldg(q)
```

The function `ldg` takes in a vector of length `dims`, and any number
of keyword arguments, and returns a 2-tuple consisting of a scalar and
a vector of the same length as the input.  The returned scalar is the
evaluation of `ldg` at the input `q`.  The returned vector is the
gradient of `ldg` evaluated at `q`.

To obtain samples from the distribution corresponding to the log joint
density function above, using the `MCBayes` implementation of
[Stan](https://mc-stan.org)'s dynamic Hamiltonian Monte Carlo
algorithm, instantiate a Stan sampler object and pass it to the
`MCBayes` function `sample!(...)`.

```julia
stan = Stan(dims)
draws, diagnostics, rngs = sample!(stan, ldg)
```

The output of `sample!(...)` is a 3-tuple.  The first element is a
3-dimensional array of draws (iterations x dimension x chains) from
the distribution corresponding to the log joint density function
`ldg`.  The second element is a tuple of diagnostics, which are meant
to help the user evaluate the success (or lack thereof) of the
sampling algorithm.  The third element is the random number generator
seeds at the point that the sampling algorithm terminated.

The sampler object could be `MEADS(dims)`, for an implentation of
[Tuning-Free Generalized Hamiltonian Monte
Carlo](https://proceedings.mlr.press/v151/hoffman22a.html), or
`MH(dims)` for an implementation of the Metropolis algorithm which
adaptatively tunes the metric for the proposal distribution and the
stepsize.

## Internal API

Almost the entire structure of all sampling algorithms of `MCBayes`
fit into this one function.  For now, focus on only the arguments
`sampler` and `ldg`, and the verbs/functions.  Try to ignore the
adapters for just a moment.

```julia
function run_sampler!(sampler::AbstractSampler, ldg;
                      iterations = 1000,
                      warmup = iterations,
                      stepsize_adapter = StepsizeConstant(...),
                      metric_adapter = MetricConstant(...),
                      trajectorylength_adapter = TrajectorlengthConstant(...),
                      damping_adapter = DampingConstant(...),
                      noise_adapter = NoiseConstant(...),
                      drift_adapter = DriftConstant(...),
                      adaptation_schedule = WindowedAdaptationSchedule(warmup),
                      ...
                      )
    M = iterations + warmup
    draws = preallocate_draws(M, ...)
    diagnostics = preallocate_diagnostics(...)

    initialize_sampler!(sampler, ...)
    initialize_draws!(draws, ldg, ...)
    initialize_stepsize!(sampler, ldg, draws, ...)

    for m in 1:M
        transition!(sampler, m, ldg, draws, ...)
        adapt!(sampler, adaptation_schedule, m, ldg, draws,
               stepsize_adapter,
               metric_adapter,
               trajectorylength_adapter,
               damping_adapter,
               noise_adapter,
               drift_adapter,
               ...
               )
    end

    return draws, diagnostics, rngs
end
```

`MCBayes` makes heavy use of the multiple dispatch capabilities of
Julia.  So each sampler gets its own `transition!(...)` function.  All
samplers, so far, share the one `adapt!(...)` method, since the
internals of `adapt!(...)` dispatch on appropriate adatpers.

You may have noticed that the [User Interface](@ref) example calls
`sample!(...)`, not `run_sampler!(...)`.  In fact, each sampler
object, e.g. Stan, MEADS, MH, gets its own `sample!(...)` function
which specifies the adaptation components appropriate to that sampler
and then immediately calls `run_sampler!(...)`.

For instance, a default run of Stan uses Dual Averaging to
adapt the stepsize during the warmup iterations.  So the function
`sample!(...)` appropriate to Stan has signature something like

```julia
function sample!(sampler::Stan, ldg;
                 iterations = 1000,
                 warmup = iterations,
                 stepsize_adapter = StepsizeDualAverage(sampler.stepsize),
                 metric_adapter = MetricOnlineMoments(sampler.metric),
                 adaptation_schedule = WindowedAdaptationSchedule(warmup),
                 ...)
    return run_sampler!(sampler,
                        ldg;
                        iterations,
                        warmup,
                        stepsize_adapter,
                        metric_adapter,
                        adaptation_schedule,
                        kwargs...)
end
```

The function `sample!(sampler::Stan, ...)`  passes the adaptation
components specific to Stan to `run_sampler!(...)`, thus over-riding
the adapters `run_sampler!(...)` otherwise defaults to.  Now is a
good time to go look back at the default adapters of
`run_sampler!(...)`.

All `*Constant(...)` adapters are effectively a no-op within
`adapt!(...)`.  The constant adapters are thus ignored when not
appropriate for a particular sampler.

Consider the adapter `TrajectorylengthConstant(...)`.  This has zero
effect on the Stan sampler, since Stan's trajectory length is
dynamically determined from a modern implementation of
[NUTS](https://arxiv.org/abs/1111.4246) (see [Michael
Betancourt](https://betanalpha.github.io/)'s [A Conceptual
Introduction to Hamiltonian Monte
Carlo](https://arxiv.org/abs/1701.02434)).  The dynamic trajectory
length is computed within `transition!(sampler::Stan, ...)`, namely
the `transition!(...)` function appropriate to the Stan sampler.

The method `adapt!(...)` itself dispatches to particular `adapt!(...)`
functions, dependent on the `adaptation_schedule`.  Right now, only
Stan's [windowed adaptation
schedule](https://mc-stan.org/docs/reference-manual/hmc-algorithm-parameters.html#automatic-parameter-tuning)
is implemented.  The structure of `adapt!(sampler,
schedule::WindowedAdaptationSchedule, ...)` looks something like

```julia
function adapt!(sampler, schedule::WindowedAdaptationSchedule,
                m, ldg, draws,
                stepsize_adapter,
                metric_adapter,
                trajectorylength_adapter,
                damping_adapter,
                noise_adapter,
                drift_adapter,
                ...)
    # within warmup
    if m <= sampler.warmup
        # update and set stepsize
        update!(stepsize_adapter, ...)
        set!(sampler, stepsize_adapter, ...)

        # update and set trajectory length
        update!(trajectorylength_adapter; kwargs...)
        set!(sampler, trajectorylength_adapter, ...)

        # in appropriate windows, update (without setting) metric
        if schedule.firstwindow <= m <= schedule.lastwindow
            update!(metric_adapter, ...)
        end

        # at end of window
        if m == schedule.closewindow
            # re-initialize, set, and then reset stepsize
            initialize_stepsize!(stepsize_adapter, ...)
            set!(sampler, stepsize_adapter, ...)
            reset!(stepsize_adapter, ...)

            # set and then reset metric
            set!(sampler, metric_adapter, ...)
            reset!(metric_adapter)

            calculate_nextwindow!(schedule)
        end
    else
        set!(sampler, stepsize_adapter, ...)
    end
end
```

When a Stan sampler is passed into the `adapt!(...)` function above,
the adapter within the variable `trajectorylength_adapter` has type
`TrajectorylengthConstant`.  The method `update!(...)` dispatches to
the function appropriate to this type and results in a no-op.
