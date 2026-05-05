from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    LINEARIZED_RHS_SCHEMA,
    run_linearized_rhs_validation,
    write_linearized_rhs_validation,
)
from mhx.cli.main import app
from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import (
    finite_difference_linearized_reduced_mhd_rhs,
    linearized_reduced_mhd_rhs,
)
from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDParams, ReducedMHDState


def test_linearized_rhs_api_matches_diffusion_operator() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    psi = grid.sinusoid(mode=(1, 0))
    state = ReducedMHDState(psi=jnp.zeros_like(psi), omega=jnp.zeros_like(psi))
    perturbation = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))
    params = ReducedMHDParams(resistivity=0.1, viscosity=0.0)
    tangent = linearized_reduced_mhd_rhs(
        state,
        perturbation,
        params,
        lengths=grid.lengths,
    )
    assert float(jnp.max(jnp.abs(tangent.psi + 0.1 * psi))) < 1.0e-10
    assert float(jnp.max(jnp.abs(tangent.omega))) < 1.0e-10
    with pytest.raises(ValueError, match="epsilon"):
        finite_difference_linearized_reduced_mhd_rhs(
            state,
            perturbation,
            params,
            lengths=grid.lengths,
            epsilon=0.0,
        )


def test_linearized_rhs_validation_matches_finite_difference() -> None:
    result = run_linearized_rhs_validation(shape=(12, 12))
    assert result.diagnostics["schema"] == LINEARIZED_RHS_SCHEMA
    assert result.validation["passed"] is True
    assert result.relative_errors["psi"] < 1.0e-3
    assert result.relative_errors["omega"] < 1.0e-3


def test_write_linearized_rhs_validation_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_linearized_rhs_validation(tmp_path, shape=(12, 12))
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["tearing_context"].startswith("Linearized")
    history = np.load(tmp_path / "linearized_rhs.npz")
    assert history["schema"] == LINEARIZED_RHS_SCHEMA
    assert history["jvp_psi"].shape == (12, 12)
    assert (tmp_path / "figures" / "linearized_rhs_errors.png").stat().st_size > 0

    outdir = tmp_path / "cli"
    result = CliRunner().invoke(app, ["benchmark", "linearized-rhs", "--outdir", str(outdir)])
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()
