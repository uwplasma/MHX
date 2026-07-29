# Architecture

The rebuild separates the codebase into narrow, testable layers:

- `config`: typed TOML schemas and reproducible run settings.
- `grids`: domain geometry and coordinate generation.
- `numerics`: differentiable discretization operators.
- `equations`: equation-set definitions; currently the active solver path is
  reduced MHD.
- `physics`: optional source terms and closures.
- `diagnostics`: standardized metrics shared by simulations, scans, optimization,
  and plotting.
- `io`: manifests, schemas, and checkpoint/output helpers.
- `cli`: command-line workflows.

The initial numerical path is periodic pseudo-spectral differentiation:

$$
\partial_x f(x) = \mathcal{F}^{-1}\left[i k_x \mathcal{F}(f)\right].
$$

Configured runs use a tensor-product 2/3-rule filter by default. The low-level
RHS retains `dealiasing="none"` for compatibility with existing validation
operators; callers that bypass `RunConfig` must select the filter explicitly.
MHX owns that discretization, together with equation residuals, gauge and
boundary semantics, and physics-derived preconditioners.

This is the smallest robust foundation for the first tearing-mode benchmark and
gradient checks. Finite-volume compressible MHD, constrained transport, and
extended state equations remain future work. The current public API exposes
additive reduced-MHD physics terms and plugin diagnostics.

The first time-dependent workflow is a reduced-MHD flux/vorticity model using a
fixed-step RK4 integrator built on `jax.lax.scan`. This keeps trajectories
differentiable and makes short gradient checks cheap enough for CI. An opt-in
backward-Euler path formulates the physical residual in MHX and delegates its
Jacobian-free Newton--Krylov and FGMRES algebra to SOLVAX. The default
preconditioner is the exact Fourier inverse of the resistive/viscous principal
part. SOLVAX remains optional; the existing direct periodic FFT Poisson solve
stays in MHX because it is already diagonal and non-iterative.

This boundary is intentional:

- MHX owns physical state, equations, discretization, time formulation,
  boundary/gauge meaning, diagnostics, and physics approximations.
- SOLVAX owns reusable algebraic solvers, Krylov iteration, nonlinear stage
  solves, preconditioner composition, and implicit differentiation.
