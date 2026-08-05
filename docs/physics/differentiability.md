# Differentiability

MHX is differentiable by construction {cite}`jax2018`. The solver core is a
pure function from initial state and parameters to trajectory, so JAX
transformations such as `jax.grad`, `jax.jacfwd`, and `jax.vmap` apply
directly. This page states what differentiates, what does not, and how to
validate a gradient before you trust it.

## What differentiates

The functional core differentiates end to end: spectral operators, Poisson
brackets, physics terms, and RK4 steps under `lax.scan`. Smooth diagnostics
of the fields differentiate with it. A minimal example computes the
sensitivity of the final magnetic energy to the resistivity:

```python
import jax
import jax.numpy as jnp
import math

from mhx.equations.reduced_mhd import reduced_mhd_rhs_spectral, to_spectral_state
from mhx.grids import CartesianGrid
from mhx.physics import CosineTearingEquilibrium
from mhx.state import ReducedMHDParams
from mhx.time_integrators import evolve_rk4

grid = CartesianGrid(
    shape=(32, 32),
    lower=(0.0, 0.0),
    upper=(2.0 * math.pi, 2.0 * math.pi),
)
state0 = to_spectral_state(CosineTearingEquilibrium().initial_state(grid))


def loss(resistivity):
    params = ReducedMHDParams(resistivity=resistivity, viscosity=5e-3)

    def rhs(state):
        return reduced_mhd_rhs_spectral(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(state0, rhs, dt=2e-2, steps=20, save_every=20)
    return jnp.sum(jnp.abs(trajectory.states.psi[-1]) ** 2)


value, gradient = jax.value_and_grad(loss)(jnp.asarray(5e-3))
```

The backward-Euler path also differentiates. SOLVAX implements implicit
differentiation of the Newton--Krylov solve, so gradients do not
differentiate through solver iterations.

`mhx.Simulation.run` itself is not differentiable. It wraps the core with
timing, printing, and file output for convenience. Differentiable work calls
the functional layer directly, as above.

## Validate every gradient

Follow this checklist before using a gradient in a result:

1. **Enable float64.** Set `JAX_ENABLE_X64=1`. Under float32, finite
   differences carry percent-level noise and gradient checks are
   inconclusive.
2. **Check against finite differences** at small resolution:

   $$
   \frac{\partial L}{\partial \eta} \approx
   \frac{L(\eta+\epsilon) - L(\eta-\epsilon)}{2\epsilon},
   $$

   with $\epsilon$ scanned over at least two decades.
3. **Check step-size stability.** The comparison must hold at two time steps.
4. **Prefer smooth objectives.** Integrated energies, smoothed flux
   differences, and spectral amplitudes differentiate cleanly.

The test suite pins this contract with a finite-difference check through an
RK4 solve. Future objectives must stay differentiable or document the
exception.

## What does not differentiate

Some quantities are discrete and carry no useful gradient:

- X-point and O-point **counts** from critical-point detection.
- Plasmoid counts and any thresholded event count.
- Argmax-based island-width proxies at the switch between two maxima.

These diagnostics jump when a critical point appears or disappears, so their
derivative is zero almost everywhere and undefined at the jump. Optimize a
smooth surrogate instead, for example integrated residual flux, and report
the discrete diagnostic beside it.

## Memory for long trajectories

Reverse-mode differentiation stores the forward trajectory. For long runs,
wrap step blocks with `jax.checkpoint` to trade recomputation for memory.
The saved-state cadence (`save_every`) does not affect gradient memory, only
the stored output.
