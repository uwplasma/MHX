from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    NONLINEAR_DURATION_AUDIT_SCHEMA,
    run_nonlinear_duration_audit,
    write_nonlinear_duration_audit,
)
from mhx.cli.main import app


def test_nonlinear_duration_audit_flags_short_ci_runs() -> None:
    result = run_nonlinear_duration_audit()
    diagnostics = result.diagnostics
    validation = result.validation

    assert diagnostics["schema"] == NONLINEAR_DURATION_AUDIT_SCHEMA
    assert validation["passed"] is True
    assert validation["checks"]["current_nonlinear_runs_flagged_short"] is True
    assert validation["checks"]["linear_replay_flagged_partial"] is True
    assert diagnostics["required_linear_window"] == pytest.approx(10.0 / 1.31e-2)
    assert diagnostics["fractions"]["current_nonlinear_fraction_of_required_linear_window"] < 0.01
    assert "too short" in diagnostics["interpretation"]
    assert "duration_policy_assessments" in diagnostics
    assert np.all(np.diff(result.plasmoid_efold_times) < 0.0)


def test_write_nonlinear_duration_audit_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_nonlinear_duration_audit(tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "validation.json").exists()
    assert (tmp_path / "nonlinear_duration_audit.npz").exists()
    assert (tmp_path / "figures" / "nonlinear_duration_audit.png").stat().st_size > 0
    persisted = json.loads((tmp_path / "diagnostics.json").read_text())
    assert persisted["schema"] == NONLINEAR_DURATION_AUDIT_SCHEMA
    history = np.load(tmp_path / "nonlinear_duration_audit.npz")
    assert history["schema"] == NONLINEAR_DURATION_AUDIT_SCHEMA
    assert history["target_end_times"].shape == (3,)

    outdir = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        ["benchmark", "nonlinear-duration-audit", "--outdir", str(outdir)],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "figures" / "nonlinear_duration_audit.png").exists()


def test_nonlinear_duration_audit_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="harris_growth_rate"):
        run_nonlinear_duration_audit(harris_growth_rate=0.0)
    with pytest.raises(ValueError, match="requested_linear_efolds"):
        run_nonlinear_duration_audit(requested_linear_efolds=0.0)
    with pytest.raises(ValueError, match="plasmoid_lundquist"):
        run_nonlinear_duration_audit(plasmoid_lundquist=(1.0e4, -1.0))
