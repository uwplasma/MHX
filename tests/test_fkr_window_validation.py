from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    FKR_WINDOW_SCHEMA,
    run_fkr_window_validation,
    write_fkr_window_validation,
)
from mhx.cli.main import app


def test_fkr_window_validation_gates_constant_psi_regime() -> None:
    result = run_fkr_window_validation()
    assert result.diagnostics["schema"] == FKR_WINDOW_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert np.all(result.delta_prime_a > 0.0)
    assert np.all(result.constant_psi_product < 0.5)


def test_fkr_window_validation_rejects_non_fkr_samples() -> None:
    with pytest.raises(ValueError, match="at least three"):
        run_fkr_window_validation(ka=(0.2, 0.3))
    with pytest.raises(ValueError, match="positive"):
        run_fkr_window_validation(ka=(0.0, 0.2, 0.3))
    with pytest.raises(ValueError, match="ka < 1"):
        run_fkr_window_validation(ka=(0.2, 0.8, 1.1))
    result = run_fkr_window_validation(max_constant_psi_product=1.0e-5)
    assert result.validation["passed"] is False


def test_write_fkr_window_validation_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_fkr_window_validation(tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["fkr"].startswith("Furth")
    history = np.load(tmp_path / "fkr_window.npz")
    assert history["schema"] == FKR_WINDOW_SCHEMA
    assert history["ka"].shape[0] == 5
    assert (tmp_path / "figures" / "fkr_constant_psi_window.png").stat().st_size > 0

    outdir = tmp_path / "cli"
    result = CliRunner().invoke(app, ["benchmark", "fkr-window", "--outdir", str(outdir)])
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()
