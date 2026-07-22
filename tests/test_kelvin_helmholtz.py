from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks.kelvin_helmholtz import (
    KELVIN_HELMHOLTZ_VALIDATION_SCHEMA,
    CompressibleKelvinHelmholtzConfig,
    KelvinHelmholtzConfig,
    dye_entropy,
    kelvin_helmholtz_entropy_jvp,
    kelvin_helmholtz_entropy_objective,
    kelvin_helmholtz_entropy_value_and_grad,
    kelvin_helmholtz_grid,
    kelvin_helmholtz_initial_state_from_config,
    run_kelvin_helmholtz_dye,
    run_kelvin_helmholtz_validation,
    write_kelvin_helmholtz_validation,
)
from mhx.cli.main import app


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


def test_kelvin_helmholtz_validation_api_has_gated_schema() -> None:
    result = run_kelvin_helmholtz_validation(
        primary_config=KelvinHelmholtzConfig(
            shape=(16, 32),
            dt=2.0e-3,
            t_end=0.12,
            save_every=10,
        ),
        comparison_config=KelvinHelmholtzConfig(
            shape=(12, 24),
            dt=2.0e-3,
            t_end=0.12,
            save_every=10,
        ),
        compressible_config=CompressibleKelvinHelmholtzConfig(
            shape=(12, 24),
            t_end=0.01,
        ),
        min_saved_samples=3,
        min_entropy_gain=1.0e-4,
        max_resolution_entropy_rdiff=5.0e-2,
    )

    assert result.diagnostics["schema"] == KELVIN_HELMHOLTZ_VALIDATION_SCHEMA
    assert result.validation["passed"] is True
    assert result.validation["checks"]["entropy_gain_observed"] is True
    assert result.validation["checks"]["resolution_entropy_consistent"] is True
    assert result.diagnostics["primary_samples"] >= 3
    assert result.diagnostics["primary_entropy_gain"] > 0.0
    assert result.diagnostics["resolution_entropy_relative_difference"] < 5.0e-2


def test_kelvin_helmholtz_validation_writes_manifest_and_npz(tmp_path: Path) -> None:
    manifest_path, validation = write_kelvin_helmholtz_validation(
        tmp_path,
        movies=True,
        primary_config=KelvinHelmholtzConfig(
            shape=(16, 32),
            dt=2.0e-3,
            t_end=0.12,
            save_every=10,
        ),
        comparison_config=KelvinHelmholtzConfig(
            shape=(12, 24),
            dt=2.0e-3,
            t_end=0.12,
            save_every=10,
        ),
        compressible_config=CompressibleKelvinHelmholtzConfig(
            shape=(12, 24),
            t_end=0.01,
        ),
        min_saved_samples=3,
        min_entropy_gain=1.0e-4,
        max_resolution_entropy_rdiff=5.0e-2,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert validation["passed"] is True
    assert manifest["claim_level"] == "validation"
    assert manifest["config"]["schema"] == KELVIN_HELMHOLTZ_VALIDATION_SCHEMA
    for relative_path in manifest["outputs"].values():
        assert (tmp_path / relative_path).is_file(), relative_path

    with np.load(tmp_path / "kelvin_helmholtz_incompressible.npz", allow_pickle=False) as data:
        assert str(data["schema"]) == KELVIN_HELMHOLTZ_VALIDATION_SCHEMA
        assert data["time"].shape == data["entropy"].shape
        assert data["dye"].shape[0] == data["time"].shape[0]
        assert data["omega"].shape == data["dye"].shape


def test_kelvin_helmholtz_cli_writes_validation_bundle(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "kelvin-helmholtz",
            "--outdir",
            str(tmp_path),
            "--nx",
            "16",
            "--ny",
            "32",
            "--comparison-nx",
            "12",
            "--comparison-ny",
            "24",
            "--compressible-nx",
            "12",
            "--compressible-ny",
            "24",
            "--t-end",
            "0.12",
            "--min-entropy-gain",
            "1e-4",
            "--min-saved-samples",
            "3",
            "--max-resolution-entropy-rdiff",
            "5e-2",
            "--movies",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "figures" / "kelvin_helmholtz_dye.gif").is_file()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"viscosity": -1.0}, "viscosity"),
        ({"resistivity": -1.0}, "resistivity"),
        ({"dt": 0.0}, "dt"),
        ({"t_end": 0.0}, "t_end"),
        ({"save_every": 0}, "save_every"),
        ({"dt": 1.0, "t_end": 0.1}, "advance at least one step"),
        ({"shear_width": 0.0}, "shear_width"),
        ({"perturbation_width": 0.0}, "perturbation_width"),
        ({"flow_speed": 0.0}, "flow_speed"),
    ],
)
def test_kelvin_helmholtz_config_rejects_invalid_controls(
    overrides: dict[str, float | int],
    message: str,
) -> None:
    config = KelvinHelmholtzConfig(**overrides)

    with pytest.raises(ValueError, match=message):
        config.validated()
