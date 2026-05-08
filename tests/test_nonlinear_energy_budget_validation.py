from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    NONLINEAR_ENERGY_BUDGET_SCHEMA,
    run_nonlinear_energy_budget_validation,
    write_nonlinear_energy_budget_validation,
)
from mhx.cli.main import app


def test_nonlinear_energy_budget_gate() -> None:
    result = run_nonlinear_energy_budget_validation(shape=(12, 12), steps=40)
    assert result.diagnostics["schema"] == NONLINEAR_ENERGY_BUDGET_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.time.shape == result.total_energy.shape
    assert result.current_dissipation.shape == result.total_energy.shape
    assert result.viscous_dissipation.shape == result.total_energy.shape
    assert np.all(np.diff(result.total_energy) <= 1.0e-9 * result.total_energy[0])
    assert result.max_relative_budget_residual < 2.0e-5
    assert result.nonlinear_rhs_ratio > 5.0e-2
    assert result.relative_energy_drop > 1.0e-2


def test_nonlinear_energy_budget_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        run_nonlinear_energy_budget_validation(shape=(7, 8))
    with pytest.raises(ValueError, match="resistivity"):
        run_nonlinear_energy_budget_validation(resistivity=0.0)
    with pytest.raises(ValueError, match="viscosity"):
        run_nonlinear_energy_budget_validation(viscosity=0.0)
    with pytest.raises(ValueError, match="dt"):
        run_nonlinear_energy_budget_validation(dt=0.0)
    with pytest.raises(ValueError, match="steps"):
        run_nonlinear_energy_budget_validation(steps=3)
    with pytest.raises(ValueError, match="save_every"):
        run_nonlinear_energy_budget_validation(save_every=0)
    with pytest.raises(ValueError, match="max_budget_residual"):
        run_nonlinear_energy_budget_validation(max_budget_residual=0.0)
    with pytest.raises(ValueError, match="max_relative_energy_growth"):
        run_nonlinear_energy_budget_validation(max_relative_energy_growth=-1.0)
    with pytest.raises(ValueError, match="min_nonlinear_rhs_ratio"):
        run_nonlinear_energy_budget_validation(min_nonlinear_rhs_ratio=0.0)
    with pytest.raises(ValueError, match="min_relative_energy_drop"):
        run_nonlinear_energy_budget_validation(min_relative_energy_drop=-1.0)


def test_write_nonlinear_energy_budget_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_nonlinear_energy_budget_validation(
        tmp_path,
        shape=(12, 12),
        steps=40,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("Periodic nonlinear")
    history = np.load(tmp_path / "nonlinear_energy_budget.npz")
    assert history["schema"] == NONLINEAR_ENERGY_BUDGET_SCHEMA
    assert history["time"].shape == history["total_energy"].shape
    assert history["initial_psi"].shape == (12, 12)
    assert history["final_omega"].shape == (12, 12)
    assert (
        tmp_path / "figures" / "nonlinear_energy_budget.png"
    ).stat().st_size > 0

    outdir = tmp_path / "cli-nonlinear-energy-budget"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "nonlinear-energy-budget",
            "--outdir",
            str(outdir),
            "--nx",
            "12",
            "--ny",
            "12",
            "--steps",
            "40",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation.json").exists()
