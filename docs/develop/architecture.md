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

## MHX and SOLVAX

MHX owns calculations that carry physical meaning:

- state variables and signs
- boundary and gauge rules
- spatial discretization
- time-discrete physical residuals
- physics-based preconditioners
- diagnostics and validation gates

SOLVAX owns calculations that apply to many models:

- matrix-free operator containers
- GMRES and recycled Krylov methods
- Newton--Krylov solves
- general preconditioner composition
- implicit differentiation

Backward Euler forms its reduced-MHD residual in MHX. It passes that residual
and the spectral diffusion preconditioner to `solvax.newton_krylov`.

## Spectral path

The current solver uses periodic Fourier derivatives:

$$
\partial_x f = \mathcal{F}^{-1}\left[i k_x \mathcal{F}(f)\right].
$$

Configured runs use the two-thirds filter for nonlinear products. The
inverse-Laplacian sets the mean Fourier mode to zero. RK4 keeps both evolving
fields in Fourier space. One batched inverse transform produces the eight
derivatives used by all three Poisson brackets. One batched forward transform
returns those brackets to Fourier space.

## Time path

RK4 uses `jax.lax.scan`. It stores only the requested states. The last state is
always present, even when `save_every` does not divide the step count.

Backward Euler uses a matrix-free Jacobian. SOLVAX computes Jacobian-vector
products with JAX and solves each linear update with GMRES.

## Device path

`make_spatial_sharding` builds a one-dimensional JAX mesh. `Simulation.run`
can split a field for one large trajectory. `Simulation.run_ensemble` splits
the case axis and gives each device complete local trajectories.

Distributed Fourier transforms require device communication. Independent cases
do not communicate inside the time loop, so case parallelism is the first
choice for scans and seed ensembles. The checked CPU and GPU measurements are
in {doc}`../reference/performance`.

## Current limit

The active state contains two two-dimensional fields, magnetic flux and
vorticity. Full three-dimensional MHD, compressibility, and constrained
transport need separate equation and state modules.
