"""Differentiate a reduced-MHD solve and check the gradient numerically."""

import math

import jax
import jax.numpy as jnp

from mhx.equations.reduced_mhd import reduced_mhd_rhs_spectral, to_spectral_state
from mhx.grids import CartesianGrid
from mhx.physics import CosineTearingEquilibrium
from mhx.state import ReducedMHDParams
from mhx.time_integrators import evolve_rk4

# Gradient checks need float64. Float32 finite differences carry
# percent-level noise and cannot confirm a derivative.
jax.config.update("jax_enable_x64", True)

grid = CartesianGrid(
    shape=(32, 32),
    lower=(0.0, 0.0),
    upper=(2.0 * math.pi, 2.0 * math.pi),
)
state0 = to_spectral_state(CosineTearingEquilibrium().initial_state(grid))


def loss(resistivity):
    """Final spectral magnetic energy proxy after a short RK4 solve."""
    params = ReducedMHDParams(resistivity=resistivity, viscosity=5.0e-3)

    def rhs(state):
        return reduced_mhd_rhs_spectral(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(state0, rhs, dt=2.0e-2, steps=20, save_every=20)
    return jnp.sum(jnp.abs(trajectory.states.psi[-1]) ** 2)


eta = jnp.asarray(5.0e-3)
value, gradient = jax.value_and_grad(loss)(eta)
print(f"loss(eta=5e-3)      = {value:.10e}")
print(f"d(loss)/d(eta)      = {gradient:.10e}")

for epsilon in (1.0e-5, 1.0e-6):
    finite_difference = (loss(eta + epsilon) - loss(eta - epsilon)) / (2.0 * epsilon)
    relative_error = abs(finite_difference - gradient) / abs(gradient)
    print(
        f"central difference  = {finite_difference:.10e} "
        f"(eps={epsilon:.0e}, relative error {relative_error:.2e})"
    )
