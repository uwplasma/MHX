# Time integration

MHX advances the reduced-MHD state with one of two fixed-step integrators:
explicit RK4 or backward Euler. Both run inside `jax.lax.scan`, so JAX
compiles the whole time loop into one program. Fixed steps keep runs
reproducible, easy to compare across codes, and differentiable.

## RK4

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

### Choosing the step

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
The [duration audit](../validation/nonlinear.md) documents the checked
step and resolution combinations.

## Backward Euler

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
[SOLVAX boundary](solvax_boundary.md) page documents the solver contract, and
[choose settings](../how_to/run_from_toml.md) covers the config fields.
