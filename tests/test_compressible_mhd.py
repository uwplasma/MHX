from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from mhx.benchmarks.kelvin_helmholtz import (
    CompressibleKelvinHelmholtzConfig,
    compressible_kelvin_helmholtz_grid,
    compressible_kelvin_helmholtz_initial_state,
    run_compressible_kelvin_helmholtz,
)
from mhx.equations.compressible_mhd import compressible_mhd_rhs, uniform_compressible_mhd_state
from mhx.state import (
    CompressibleMHDParams,
    CompressibleMHDPrimitive,
    conservative_from_primitive,
    primitive_from_conservative,
)


def test_compressible_primitive_conservative_roundtrip() -> None:
    params = CompressibleMHDParams(gamma=5.0 / 3.0)
    primitive = CompressibleMHDPrimitive(
        density=jnp.full((4, 6), 1.2),
        velocity_x=jnp.full((4, 6), 0.1),
        velocity_y=jnp.full((4, 6), -0.03),
        pressure=jnp.full((4, 6), 2.5),
        magnetic_x=jnp.full((4, 6), 0.2),
        magnetic_y=jnp.full((4, 6), 0.05),
        dye=jnp.full((4, 6), 0.7),
    )

    restored = primitive_from_conservative(conservative_from_primitive(primitive, params), params)

    np.testing.assert_allclose(restored.density, primitive.density)
    np.testing.assert_allclose(restored.velocity_x, primitive.velocity_x)
    np.testing.assert_allclose(restored.velocity_y, primitive.velocity_y)
    np.testing.assert_allclose(restored.pressure, primitive.pressure)
    np.testing.assert_allclose(restored.magnetic_x, primitive.magnetic_x)
    np.testing.assert_allclose(restored.magnetic_y, primitive.magnetic_y)
    np.testing.assert_allclose(restored.dye, primitive.dye)


def test_uniform_compressible_mhd_rhs_is_zero() -> None:
    params = CompressibleMHDParams(gamma=5.0 / 3.0)
    state = uniform_compressible_mhd_state(
        shape=(8, 10),
        density=1.0,
        velocity=(0.1, -0.02),
        pressure=3.0,
        magnetic=(0.2, 0.05),
        dye=0.4,
        params=params,
    )

    rhs = compressible_mhd_rhs(state, params, lengths=(1.0, 2.0))

    for leaf in rhs:
        assert float(jnp.max(jnp.abs(leaf))) < 1.0e-12


def test_compressible_kelvin_helmholtz_initial_state_has_positive_pressure() -> None:
    config = CompressibleKelvinHelmholtzConfig(shape=(12, 24), pressure=10.0)
    grid = compressible_kelvin_helmholtz_grid(config)
    state = compressible_kelvin_helmholtz_initial_state(grid, config)
    primitive = primitive_from_conservative(state, CompressibleMHDParams(gamma=config.gamma))

    assert float(jnp.min(primitive.density)) > 0.0
    assert float(jnp.min(primitive.pressure)) > 0.0
    assert 0.0 <= float(jnp.min(primitive.dye)) <= 1.0
    assert 0.0 <= float(jnp.max(primitive.dye)) <= 1.0


def test_compressible_kelvin_helmholtz_fast_run_is_finite() -> None:
    result = run_compressible_kelvin_helmholtz(
        CompressibleKelvinHelmholtzConfig(
            shape=(12, 24),
            dt=5.0e-4,
            t_end=5.0e-3,
            save_every=5,
        )
    )

    assert result.trajectory.times.shape == (2,)
    assert result.trajectory.states.density.shape == (2, 12, 24)
    assert bool(jnp.all(jnp.isfinite(result.trajectory.states.density)))
    assert bool(jnp.all(jnp.isfinite(result.trajectory.states.total_energy)))
    assert float(jnp.min(result.density_min)) > 0.0
    assert float(jnp.min(result.pressure_min)) > 0.0
    assert all(math.isfinite(float(value)) for value in result.dye_entropy)
