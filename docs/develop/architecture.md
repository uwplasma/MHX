# Architecture

MHX has one beginner path:

```text
Simulation -> reduced-MHD equations -> JAX compilation -> SimulationResult
```

`mhx.Simulation` accepts the grid, equilibrium, diffusivities, time settings,
integrator, and device count. `mhx.SimulationResult` prints, plots, and saves
the output.

## Source layers

The package separates model code from general numerical algebra:

| Layer | Responsibility |
| --- | --- |
| `simulation` | Public run and result API |
| `ensemble` | Independent case batches and multi-process output |
| `physics` | Equilibria and extra physical terms |
| `equations` | Reduced-MHD right-hand side and residuals |
| `numerics.spectral` | Fourier derivatives and dealiasing |
| `time_integrators` | RK4 and backward-Euler time formulas |
| `diagnostics` | Energy, modes, divergence, and reconnection quantities |
| `parallel` | JAX device mesh and field placement |
| `io` and `plotting` | Files and figures |
| `benchmarks` and `campaigns` | Validation and long-run controls |

The config and command-line modules support recorded benchmark campaigns. New
users can start with `mhx.Simulation` and ignore those modules.

## Where the numerics are documented

The physics section documents each numerical layer beside its equations:

- [The reduced-MHD model](../physics/reduced_mhd.md) for `equations` and
  `physics`.
- [Spectral method](../physics/spectral_method.md) for `numerics.spectral`.
- [Time integration](../physics/time_integration.md) for `time_integrators`.
- [MHX and SOLVAX](../physics/solvax_boundary.md) for the solver contract.
- [Differentiability](../physics/differentiability.md) for the pure
  functional layer that JAX transformations require.

## Device path

`make_spatial_sharding` builds a one-dimensional JAX mesh. `Simulation.run`
can split a field for one large trajectory. `Simulation.run_ensemble` splits
the case axis and gives each device complete local trajectories.

Distributed Fourier transforms require device communication. Independent
cases do not communicate inside the time loop, so case parallelism is the
first choice for scans and seed ensembles. The checked CPU and GPU
measurements are in {doc}`../reference/performance`.

## Design rules

1. Solver state never mutates inside JIT. Steps are pure functions.
2. Time loops run under `jax.lax.scan`, never Python loops.
3. Differentiable values live in parameter PyTrees. Static choices live in
   config objects.
4. Every run artifact records schema and API versions through
   `mhx.versioning`.

## Current limit

The active state contains two two-dimensional fields, magnetic flux and
vorticity. Full three-dimensional MHD, compressibility, and constrained
transport need separate equation and state modules.
