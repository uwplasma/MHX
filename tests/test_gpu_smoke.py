"""Short device checks for the self-hosted GPU lane (`pytest -m gpu`).

Each test also passes on CPU, so the same checks run on any machine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mhx
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDParams

pytestmark = pytest.mark.gpu

SHAPE = (32, 32)


def _run(**kwargs):
    return mhx.Simulation(
        shape=SHAPE, dt=0.01, t_end=0.1, save_every=5, verbose=False, **kwargs
    ).run()


def test_rk4_steps_are_finite_and_dissipate_energy() -> None:
    result = _run()
    state = result.final_state
    assert np.all(np.isfinite(np.asarray(state.psi)))
    assert np.all(np.isfinite(np.asarray(state.omega)))
    assert result.final_time == pytest.approx(0.1)
    assert result.diagnostics["final_total_energy"] <= result.diagnostics["initial_total_energy"]


def test_zero_equilibrium_stays_zero() -> None:
    result = _run(equilibrium=mhx.ZeroEquilibrium())
    assert float(jnp.max(jnp.abs(result.final_state.psi))) == 0.0
    assert float(jnp.max(jnp.abs(result.final_state.omega))) == 0.0


def test_device_matches_cpu() -> None:
    device = _run().final_state
    with jax.default_device(jax.devices("cpu")[0]):
        host = _run().final_state
    np.testing.assert_allclose(np.asarray(device.psi), np.asarray(host.psi), atol=1e-10)
    np.testing.assert_allclose(np.asarray(device.omega), np.asarray(host.omega), atol=1e-10)


def test_rhs_gradient_is_finite() -> None:
    sim = mhx.Simulation(shape=SHAPE, verbose=False)
    grid = CartesianGrid(shape=SHAPE, lower=sim.lower, upper=sim.upper)
    state = sim.equilibrium.initial_state(grid)

    def energy_rate(eta):
        rhs = reduced_mhd_rhs(
            state,
            ReducedMHDParams(resistivity=eta, viscosity=5.0e-3),
            lengths=grid.lengths,
            dealiasing="two_thirds",
        )
        return jnp.sum(rhs.psi * state.psi) + jnp.sum(rhs.omega * state.omega)

    grad = jax.jit(jax.grad(energy_rate))(5.0e-3)
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0
