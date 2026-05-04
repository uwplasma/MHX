# Architecture

The rebuild separates the codebase into narrow, testable layers:

- `config`: typed TOML schemas and reproducible run settings.
- `grids`: domain geometry and coordinate generation.
- `numerics`: differentiable discretization operators.
- `equations`: equation-set definitions such as reduced MHD and compressible MHD.
- `physics`: optional source terms and closures.
- `diagnostics`: standardized metrics shared by simulations, scans, optimization,
  and plotting.
- `io`: manifests, schemas, and checkpoint/output helpers.
- `cli`: command-line workflows.

The initial numerical path is periodic pseudo-spectral differentiation:

$$
\partial_x f(x) = \mathcal{F}^{-1}\left[i k_x \mathcal{F}(f)\right].
$$

This is the smallest robust foundation for the first tearing-mode benchmark and
gradient checks. Finite-volume MHD, constrained transport, and extended physics
terms will be added after this layer is stable.

The first time-dependent workflow is a reduced-MHD flux/vorticity model using a
fixed-step RK4 integrator built on `jax.lax.scan`. This keeps trajectories
differentiable and makes short gradient checks cheap enough for CI.
