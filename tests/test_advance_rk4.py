"""Tests for the scan-based ``advance_rk4`` integrator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import advance_rk4, evolve_rk4


def _decay(rate):
    return lambda y: -rate * y


def test_matches_last_state_of_evolve_rk4() -> None:
    y0 = jnp.array([1.0, -2.0, 0.5])
    rhs = _decay(1.3)
    final = advance_rk4(y0, rhs, dt=0.05, steps=12)
    trajectory = evolve_rk4(y0, rhs, dt=0.05, steps=12, save_every=3)
    np.testing.assert_allclose(final, trajectory.states[-1], rtol=1e-13)


def test_fourth_order_convergence_on_linear_ode() -> None:
    rate, t_end = 1.7, 1.0
    y0 = jnp.array(1.0)
    exact = np.exp(-rate * t_end)

    errors = []
    for steps in (8, 16, 32):
        final = advance_rk4(y0, _decay(rate), dt=t_end / steps, steps=steps)
        errors.append(abs(float(final) - exact))

    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(3.8 < order < 4.3 for order in orders)


def test_checkpoint_gives_identical_values_and_gradients() -> None:
    y0 = jnp.array([1.0, 0.3])

    def loss(rate: jax.Array, checkpoint: bool) -> jax.Array:
        final = advance_rk4(y0, _decay(rate), dt=0.1, steps=10, checkpoint=checkpoint)
        return jnp.sum(final**2)

    rate = jnp.asarray(0.8)
    plain_value, plain_grad = jax.value_and_grad(loss)(rate, False)
    ckpt_value, ckpt_grad = jax.value_and_grad(loss)(rate, True)

    np.testing.assert_allclose(plain_value, ckpt_value, rtol=1e-13)
    np.testing.assert_allclose(plain_grad, ckpt_grad, rtol=1e-12)
    # Analytic derivative of sum(y0^2) exp(-2 rate t) at t = 1.
    exact = -2.0 * float(jnp.sum(y0**2)) * np.exp(-2.0 * 0.8)
    assert float(plain_grad) == pytest.approx(exact, rel=1e-4)


def test_zero_steps_returns_input() -> None:
    y0 = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
    out = advance_rk4(y0, lambda s: jax.tree_util.tree_map(lambda v: -v, s), dt=0.1, steps=0)
    jax.tree_util.tree_map(np.testing.assert_array_equal, out, y0)


def test_pytree_state_reduced_mhd() -> None:
    n = 16
    length = 2.0 * np.pi
    x = jnp.arange(n) * length / n
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    state = ReducedMHDState(psi=0.1 * jnp.cos(xx) * jnp.cos(yy), omega=0.1 * jnp.sin(yy))
    params = ReducedMHDParams(resistivity=0.05, viscosity=0.05)

    def rhs(s: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(s, params, lengths=(length, length))

    final = advance_rk4(state, rhs, dt=0.01, steps=5)
    ckpt = advance_rk4(state, rhs, dt=0.01, steps=5, checkpoint=True)
    trajectory = evolve_rk4(state, rhs, dt=0.01, steps=5, save_every=5)

    assert isinstance(final, ReducedMHDState)
    assert final.psi.shape == (n, n)
    assert bool(jnp.all(jnp.isfinite(final.psi))) and bool(jnp.all(jnp.isfinite(final.omega)))
    assert not np.allclose(final.psi, state.psi)
    np.testing.assert_allclose(final.psi, trajectory.states.psi[-1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(final.omega, trajectory.states.omega[-1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ckpt.psi, final.psi, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_rejects_non_positive_dt(dt) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        advance_rk4(jnp.array(1.0), _decay(1.0), dt=dt, steps=3)


def test_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps must be non-negative"):
        advance_rk4(jnp.array(1.0), _decay(1.0), dt=0.1, steps=-1)


def test_advance_rk4_accepts_a_traced_step_size() -> None:
    """A jitted caller may pass ``dt`` as an argument; only concrete values are validated."""

    def rhs(state):
        return -state

    @jax.jit
    def run(initial, dt):
        return advance_rk4(initial, rhs, dt=dt, steps=10)

    final = run(jnp.asarray(1.0), jnp.asarray(0.01))
    assert float(final) == pytest.approx(float(np.exp(-0.1)), rel=1e-8)
