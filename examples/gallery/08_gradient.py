"""Differentiate a reduced-MHD solve and check the gradient numerically.

Computes d(loss)/d(resistivity) of a short RK4 reduced-MHD solve with jax.grad and
compares it with central finite differences in float64. Prints the loss, the
gradient and the relative errors; writes no files. Takes a few seconds on a CPU;
raise N, STEPS or DT for a harder test. Run from the repository root:

    python examples/gallery/08_gradient.py
"""

import math

import jax
import jax.numpy as jnp

from mhx.equations.reduced_mhd import reduced_mhd_rhs_spectral, to_spectral_state
from mhx.grids import CartesianGrid
from mhx.physics import CosineTearingEquilibrium
from mhx.state import ReducedMHDParams
from mhx.time_integrators import evolve_rk4

# --- Input parameters ------------------------------------------------------
N = 32  # grid points per side
RESISTIVITY = 5.0e-3  # the parameter we differentiate with respect to
VISCOSITY = 5.0e-3
DT = 2.0e-2  # time step
STEPS = 20  # number of RK4 steps
EPSILONS = (1.0e-5, 1.0e-6)  # finite-difference step sizes

# Gradient checks need float64. Float32 finite differences carry
# percent-level noise and cannot confirm a derivative.
jax.config.update("jax_enable_x64", True)

# --- Run -------------------------------------------------------------------
print(f"Setting up a {N} x {N} reduced-MHD solve of {STEPS} RK4 steps...", flush=True)
grid = CartesianGrid(
    shape=(N, N),
    lower=(0.0, 0.0),
    upper=(2.0 * math.pi, 2.0 * math.pi),
)
state0 = to_spectral_state(CosineTearingEquilibrium().initial_state(grid))


def loss(resistivity):
    """Final spectral magnetic energy proxy after a short RK4 solve."""
    params = ReducedMHDParams(resistivity=resistivity, viscosity=VISCOSITY)

    def rhs(state):
        return reduced_mhd_rhs_spectral(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(state0, rhs, dt=DT, steps=STEPS, save_every=STEPS)
    return jnp.sum(jnp.abs(trajectory.states.psi[-1]) ** 2)


eta = jnp.asarray(RESISTIVITY)
print("Computing the gradient with jax.value_and_grad...", flush=True)
value, gradient = jax.value_and_grad(loss)(eta)

# --- Results ---------------------------------------------------------------
print(f"loss(eta={RESISTIVITY:g})      = {value:.10e}")
print(f"d(loss)/d(eta)      = {gradient:.10e}")

for epsilon in EPSILONS:
    finite_difference = (loss(eta + epsilon) - loss(eta - epsilon)) / (2.0 * epsilon)
    relative_error = abs(finite_difference - gradient) / abs(gradient)
    print(
        f"central difference  = {finite_difference:.10e} "
        f"(eps={epsilon:.0e}, relative error {relative_error:.2e})",
        flush=True,
    )
