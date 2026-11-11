# Numerics and differentiability

## Spectral method

MHX discretizes space with a Fourier pseudo-spectral method on a uniform,
doubly periodic grid. Derivatives are exact in the resolved band, dissipation
is purely physical, and the divergence constraint holds to machine precision.
This page defines the discrete operators and the dealiasing rule.

### Fourier representation

A field $f(x,y)$ on an $N_x \times N_y$ grid with lengths $(L_x, L_y)$ has the
discrete transform

$$
f(x,y) = \sum_{\mathbf{k}} \hat f_{\mathbf{k}}\,
  e^{i(k_x x + k_y y)}, \qquad
k_x = \frac{2\pi m}{L_x},\quad
k_y = \frac{2\pi n}{L_y},
$$

with integer mode numbers $m$ and $n$. MHX computes transforms with
`jax.numpy.fft` and keeps both evolving fields in Fourier space between
steps. Derivatives are diagonal there:

$$
\widehat{\partial_x f}_{\mathbf{k}} = i k_x \hat f_{\mathbf{k}}, \qquad
\widehat{\nabla^2 f}_{\mathbf{k}} = -|\mathbf{k}|^2 \hat f_{\mathbf{k}} .
$$

For smooth periodic fields the truncation error decays faster than any power
of the grid spacing {cite}`canuto1988`. In practice the resolution question
is physical: the current-layer width must fit in the resolved band. The
[convergence evidence](../validation/nonlinear.md) measures this
for the reconnection benchmarks.

### Inverse Laplacian and gauge

The stream function solves $\nabla^2\phi = \omega$:

$$
\hat\phi_{\mathbf{k}} = -\frac{\hat\omega_{\mathbf{k}}}{|\mathbf{k}|^2},
\qquad \hat\phi_{\mathbf{0}} = 0 .
$$

Fixing the zero mode removes the free constant in $\phi$ and $\psi$. Only
gradients of the potentials are physical, so this gauge choice does not
affect any observable.

### Dealiasing

The Poisson brackets multiply fields in real space. On a discrete grid the
product of two resolved modes can fold back onto a wrong resolved mode. MHX
removes this aliasing with the two-thirds rule of {cite}`orszag1971`: before
each product it zeros every mode with $|m| > N_x/3$ or $|n| > N_y/3$.

Runs should set `dealiasing="two_thirds"`, and `mhx.Simulation` does so by
default. The `"none"` option exists for operator unit tests, where exactness
matters and no products occur.

### Batched transforms

One right-hand-side evaluation needs eight real-space derivative fields for
the three brackets. MHX computes them with one batched inverse transform and
returns the bracket results with one batched forward transform. On a device
mesh this issues one large collective instead of many small ones, which is
the main cost lever for [sharded runs](../how_to/run_on_gpus.md).

The operators live in
[`numerics/spectral/operators.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/numerics/spectral/operators.py).
The equation assembly lives in
[`equations/reduced_mhd.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/equations/reduced_mhd.py).

## Time integration

MHX advances the reduced-MHD state with one of two fixed-step integrators:
explicit RK4 or backward Euler. Both run inside `jax.lax.scan`, so JAX
compiles the whole time loop into one program. Fixed steps keep runs
reproducible, easy to compare across codes, and differentiable.

### RK4

The default integrator is the classical fourth-order Runge--Kutta scheme. For
$\dot u = F(u)$ and step $\Delta t$:

$$
k_1 = F(u_n), \quad
k_2 = F\!\left(u_n + \tfrac{\Delta t}{2} k_1\right), \quad
k_3 = F\!\left(u_n + \tfrac{\Delta t}{2} k_2\right), \quad
k_4 = F(u_n + \Delta t\, k_3),
$$

$$
u_{n+1} = u_n + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right).
$$

The scan stores one state every `save_every` steps, so memory scales with the
saved trajectory rather than the step count. The final state is always
retained.

#### Choosing the step

Two limits constrain $\Delta t$:

1. **Advection.** The fastest wave is the Alfvén wave on the sharpest
   resolved scale. Keep $\Delta t \lesssim \Delta x / \max(|v|, |B_\perp|)$
   with $\Delta x = L_x/N_x$.
2. **Diffusion.** Explicit stability requires
   $\Delta t\,\max(\eta,\nu)\,k_{\max}^2 \lesssim 2.8$ for RK4, where
   $k_{\max}$ is the largest dealiased wavenumber. The limit scales as
   $\Delta x^2/\max(\eta,\nu)$. The `hyper_resistivity` term scales as
   $\Delta x^4/\eta_4$ and limits the step first when active.

The default `dt=2.0e-2` at $64^2$ resolution satisfies both limits with
margin for the default equilibrium. Halve the step when you double the
resolution, and rerun with a smaller step when energy behaves suspiciously.
The [nonlinear gates](../validation/nonlinear.md) document the checked
step and resolution combinations.

### Backward Euler

The implicit option solves the nonlinear residual

$$
R(u_{n+1}) = u_{n+1} - u_n - \Delta t\, F(u_{n+1}) = 0
$$

at every step with a Jacobian-free Newton--Krylov method
{cite}`knoll2004`. SOLVAX runs the Newton iteration and solves each linear
update with GMRES {cite}`saad1986`. JAX supplies exact Jacobian-vector
products by forward-mode differentiation of $F$, so MHX never forms a matrix.
A spectral diffusion preconditioner inverts the stiff $\eta\nabla^2$ and
$\nu\nabla^2$ parts exactly in Fourier space.

The Newton solve targets tolerances near $10^{-9}$, which float32 cannot
reach. Enable float64 with `JAX_ENABLE_X64=1` for every implicit run, or the
convergence flags report failure on each step.

Backward Euler is first order in time and damps unresolved transients. Use it
when the diffusive step limit, not accuracy, binds the explicit choice:

```python
simulation = mhx.Simulation(
    integrator="backward_euler",
    dt=1.0e-2,
    t_end=1.0e-1,
)
```

After a run, confirm that every step converged:

```python
assert result.diagnostics["implicit_converged"]
assert result.diagnostics["implicit_linear_converged"]
```

A run with a failed Newton or GMRES tolerance is not evidence. The
[SOLVAX boundary](numerics.md) page documents the solver contract, and
[choose settings](../how_to/run_from_toml.md) covers the config fields.

## MHX and SOLVAX

MHX and [SOLVAX](https://github.com/uwplasma/SOLVAX) split the work along one
rule. MHX owns every calculation that carries physical meaning. SOLVAX owns
numerical algebra that applies to any model.

| MHX owns | SOLVAX owns |
| --- | --- |
| state variables and their signs | matrix-free operator containers |
| boundary and gauge rules | GMRES and recycled Krylov methods |
| spatial discretization | Newton--Krylov solves |
| time-discrete physical residuals | preconditioner composition |
| physics-based preconditioners | implicit differentiation |
| diagnostics and validation gates | |

The seam is the backward-Euler step. MHX forms the reduced-MHD residual

$$
R(u_{n+1}) = u_{n+1} - u_n - \Delta t\, F(u_{n+1})
$$

and builds the spectral diffusion preconditioner from its own operators. It
passes both to `solvax.newton_krylov`, which iterates Newton updates and
solves each linear system with GMRES. SOLVAX returns the new state plus
convergence diagnostics that MHX stores in `result.diagnostics`.

This split keeps both codes honest. MHX never reimplements Krylov algebra,
and SOLVAX never hardcodes plasma physics. Solver improvements arrive through
the dependency. The integration contract is pinned by
[`tests/test_solvax_integration.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_solvax_integration.py).

The base MHX install always includes SOLVAX.

## Differentiability

MHX is differentiable by construction {cite}`jax2018`. The solver core is a
pure function from initial state and parameters to trajectory, so JAX
transformations such as `jax.grad`, `jax.jacfwd`, and `jax.vmap` apply
directly. This page states what differentiates, what does not, and how to
validate a gradient before you trust it.

### What differentiates

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

### Validate every gradient

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

### What does not differentiate

Some quantities are discrete and carry no useful gradient:

- X-point and O-point **counts** from critical-point detection.
- Plasmoid counts and any thresholded event count.
- Argmax-based island-width proxies at the switch between two maxima.

These diagnostics jump when a critical point appears or disappears, so their
derivative is zero almost everywhere and undefined at the jump. Optimize a
smooth surrogate instead, for example integrated residual flux, and report
the discrete diagnostic beside it.

### Memory for long trajectories

Reverse-mode differentiation stores the forward trajectory. For long runs,
wrap step blocks with `jax.checkpoint` to trade recomputation for memory.
The saved-state cadence (`save_every`) does not affect gradient memory, only
the stored output.
