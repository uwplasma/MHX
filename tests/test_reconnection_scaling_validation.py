from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    RECONNECTION_SCALING_SCHEMA,
    loglog_slope,
    run_reconnection_scaling_validation,
    write_reconnection_scaling_validation,
)
from mhx.cli.main import app


def test_loglog_slope_and_validation_errors() -> None:
    x_values = np.asarray([1.0, 10.0, 100.0])
    assert loglog_slope(x_values, x_values**0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="positive"):
        loglog_slope(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]))


def test_reconnection_scaling_validation_gates_match_literature_exponents() -> None:
    result = run_reconnection_scaling_validation()
    diagnostics = result.diagnostics
    validation = result.validation
    assert diagnostics["schema"] == RECONNECTION_SCALING_SCHEMA
    assert diagnostics["slopes"]["fkr_gamma"] == pytest.approx(-3.0 / 5.0)
    assert diagnostics["slopes"]["fkr_inner_width"] == pytest.approx(-2.0 / 5.0)
    assert diagnostics["slopes"]["plasmoid_gamma"] == pytest.approx(1.0 / 4.0)
    assert diagnostics["slopes"]["plasmoid_mode"] == pytest.approx(3.0 / 8.0)
    assert diagnostics["slopes"]["ideal_aspect_ratio"] == pytest.approx(-1.0 / 3.0)
    assert validation["passed"] is True
    assert all(validation["checks"].values())


def test_reconnection_scaling_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least three"):
        run_reconnection_scaling_validation(lundquist=(1.0, 2.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        run_reconnection_scaling_validation(lundquist=(1.0e4, 1.0e4, 1.0e5))
    result = run_reconnection_scaling_validation(max_slope_error=1.0e-16)
    assert result.validation["passed"] is False


def test_write_reconnection_scaling_validation_artifacts(tmp_path) -> None:
    manifest_path, validation = write_reconnection_scaling_validation(tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["fkr"].startswith("Furth")
    history = np.load(tmp_path / "scaling_history.npz")
    assert history["schema"] == RECONNECTION_SCALING_SCHEMA
    assert history["lundquist"].shape[0] == 4
    assert (tmp_path / "figures" / "fkr_scaling.png").stat().st_size > 0
    assert (tmp_path / "figures" / "plasmoid_scaling.png").stat().st_size > 0
    assert (tmp_path / "figures" / "ideal_tearing_scaling.png").stat().st_size > 0


def test_reconnection_scaling_cli(tmp_path) -> None:
    outdir = tmp_path / "scaling"
    result = CliRunner().invoke(app, ["benchmark", "scaling", "--outdir", str(outdir)])
    assert result.exit_code == 0, result.stdout
    validation = json.loads((outdir / "validation.json").read_text())
    assert validation["passed"] is True
