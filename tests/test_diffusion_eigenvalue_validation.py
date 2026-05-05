from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    DIFFUSION_EIGENVALUE_SCHEMA,
    POWER_ITERATION_SCHEMA,
    run_diffusion_eigenvalue_validation,
    run_power_iteration_validation,
    write_diffusion_eigenvalue_validation,
    write_power_iteration_validation,
)
from mhx.cli.main import app
from mhx.config import MeshConfig
from mhx.grids import CartesianGrid
from mhx.numerics import MatrixFreeOperator, eigen_residual_norm, power_iteration, rayleigh_quotient


def test_matrix_free_operator_shape_checks_and_rayleigh_quotient() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(8, 8)))
    vector = grid.sinusoid(mode=(1, 0))
    operator = MatrixFreeOperator(shape=grid.shape, matvec=lambda values: -2.0 * values)
    assert float(rayleigh_quotient(operator, vector)) == pytest.approx(-2.0)
    assert float(eigen_residual_norm(operator, vector, -2.0)) < 1.0e-12
    with pytest.raises(ValueError, match="expected shape"):
        operator(jnp.ones((4, 4)))
    bad_operator = MatrixFreeOperator(shape=grid.shape, matvec=lambda values: values[0])
    with pytest.raises(ValueError, match="returned shape"):
        bad_operator(vector)
    with pytest.raises(ValueError, match="iterations"):
        power_iteration(operator, vector, iterations=0)
    with pytest.raises(ValueError, match="zero vector"):
        power_iteration(operator, jnp.zeros_like(vector), iterations=1)


def test_diffusion_eigenvalue_validation_matches_spectral_theory() -> None:
    result = run_diffusion_eigenvalue_validation(shape=(16, 16), mode=(2, 1))
    assert result.diagnostics["schema"] == DIFFUSION_EIGENVALUE_SCHEMA
    assert result.validation["passed"] is True
    assert result.eigenvalue_abs_error < 1.0e-6
    assert result.residual_norm < 5.0e-6


def test_write_diffusion_eigenvalue_validation_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_diffusion_eigenvalue_validation(tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["eigen_scaffold"].startswith("Matrix-free")
    history = np.load(tmp_path / "diffusion_eigenvalue.npz")
    assert history["schema"] == DIFFUSION_EIGENVALUE_SCHEMA
    assert history["eigenfunction"].shape == (32, 32)
    assert (tmp_path / "figures" / "diffusion_eigenvalue_errors.png").stat().st_size > 0

    outdir = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        ["benchmark", "diffusion-eigenvalue", "--outdir", str(outdir)],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()


def test_power_iteration_validation_and_cli(tmp_path) -> None:
    result = run_power_iteration_validation(iterations=30)
    assert result.diagnostics["schema"] == POWER_ITERATION_SCHEMA
    assert result.validation["passed"] is True
    assert result.eigenvalue_abs_error < 1.0e-6
    assert result.residual_norm < 1.0e-6

    manifest_path, validation = write_power_iteration_validation(tmp_path, iterations=30)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    history = np.load(tmp_path / "power_iteration_history.npz")
    assert history["schema"] == POWER_ITERATION_SCHEMA
    assert history["rayleigh_history"].shape == (30,)
    assert (tmp_path / "figures" / "power_iteration_history.png").stat().st_size > 0

    outdir = tmp_path / "power-cli"
    cli_result = CliRunner().invoke(
        app,
        ["benchmark", "power-iteration", "--outdir", str(outdir), "--iterations", "30"],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation.json").exists()
