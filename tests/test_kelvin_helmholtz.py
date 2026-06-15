from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.benchmarks.kelvin_helmholtz import (
    KelvinHelmholtzConfig,
    dye_entropy,
    kelvin_helmholtz_entropy_jvp,
    kelvin_helmholtz_entropy_objective,
    kelvin_helmholtz_entropy_value_and_grad,
    kelvin_helmholtz_grid,
    kelvin_helmholtz_initial_state_from_config,
    run_kelvin_helmholtz_dye,
)


def test_kelvin_helmholtz_fast_run_is_finite() -> None:
    config = KelvinHelmholtzConfig(
        shape=(16, 32),
        dt=1.0e-3,
        t_end=1.0e-2,
        save_every=10,
        viscosity=1.0e-3,
    )

    result = run_kelvin_helmholtz_dye(config)

    assert result.trajectory.times.shape == (1,)
    assert result.trajectory.states.dye.shape == (1, 16, 32)
    assert result.trajectory.states.mhd.omega.shape == (1, 16, 32)
    assert bool(jnp.all(jnp.isfinite(result.trajectory.states.dye)))
    assert bool(jnp.all(jnp.isfinite(result.trajectory.states.mhd.omega)))
    assert np.asarray(result.entropy).shape == (1,)
    assert float(result.entropy[-1]) > 0.0


def test_kelvin_helmholtz_entropy_is_scalar_and_finite() -> None:
    config = KelvinHelmholtzConfig(shape=(16, 32))
    grid = kelvin_helmholtz_grid(config)
    state = kelvin_helmholtz_initial_state_from_config(grid, config)

    entropy = dye_entropy(state.dye, grid)

    assert entropy.shape == ()
    assert math.isfinite(float(entropy))
    assert 0.0 < float(entropy) < 1.0


def test_kelvin_helmholtz_gradient_and_jvp_agree() -> None:
    config = KelvinHelmholtzConfig(
        shape=(16, 32),
        dt=1.0e-3,
        t_end=8.0e-3,
        save_every=8,
        viscosity=1.0e-3,
    )

    value, gradient = kelvin_helmholtz_entropy_value_and_grad(0.01, config)
    jvp_value, tangent = kelvin_helmholtz_entropy_jvp(0.01, 1.0e-3, config)

    assert math.isfinite(float(value))
    assert math.isfinite(float(gradient))
    assert math.isfinite(float(jvp_value))
    assert math.isfinite(float(tangent))
    np.testing.assert_allclose(float(jvp_value), float(value), rtol=1.0e-12)
    np.testing.assert_allclose(float(tangent), float(gradient) * 1.0e-3, rtol=1.0e-6)


def test_kelvin_helmholtz_gradient_matches_finite_difference() -> None:
    config = KelvinHelmholtzConfig(
        shape=(12, 24),
        dt=1.0e-3,
        t_end=4.0e-3,
        save_every=4,
        viscosity=1.0e-3,
    )

    def objective(amplitude: float) -> float:
        value, _ = kelvin_helmholtz_entropy_jvp(amplitude, 0.0, config)
        return float(value)

    _, gradient = kelvin_helmholtz_entropy_value_and_grad(0.01, config)
    epsilon = 1.0e-3
    finite_difference = (objective(0.01 + epsilon) - objective(0.01 - epsilon)) / (
        2.0 * epsilon
    )

    np.testing.assert_allclose(float(gradient), finite_difference, rtol=1.0e-3, atol=1.0e-8)


def test_kelvin_helmholtz_one_gradient_step_changes_parameter() -> None:
    config = KelvinHelmholtzConfig(
        shape=(12, 24),
        dt=1.0e-3,
        t_end=4.0e-3,
        save_every=4,
        viscosity=1.0e-3,
    )

    target_entropy = jnp.asarray(0.082)

    def loss(amplitude: jax.Array) -> jax.Array:
        value = kelvin_helmholtz_entropy_objective(amplitude, config)
        return (value - target_entropy) ** 2

    initial_amplitude = jnp.asarray(0.01)
    loss_value, loss_gradient = jax.value_and_grad(loss)(initial_amplitude)
    updated_amplitude = initial_amplitude - 10.0 * loss_gradient
    updated_loss = loss(updated_amplitude)

    assert math.isfinite(float(loss_value))
    assert math.isfinite(float(loss_gradient))
    assert float(updated_amplitude) != pytest.approx(float(initial_amplitude))
    assert math.isfinite(float(updated_loss))
