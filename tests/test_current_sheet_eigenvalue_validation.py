from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from mhx.benchmarks import (
    PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA,
    run_periodic_current_sheet_eigenvalue_validation,
    write_periodic_current_sheet_eigenvalue_validation,
)
from mhx.cli.main import app


def test_periodic_current_sheet_eigenvalue_gate() -> None:
    result = run_periodic_current_sheet_eigenvalue_validation(shape=(6, 6))
    assert result.diagnostics["schema"] == PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA
    assert result.validation["passed"] is True
    assert result.matrix.shape == (72, 72)
    assert result.gauge_mode_count >= 2
    assert result.max_real_part < 1.0e-9
    assert (
        result.max_non_gauge_real_part
        <= result.validation["thresholds"]["max_allowed_non_gauge_real_part"]
    )
    assert result.selected_residual_norm < 1.0e-9
    assert np.isfinite(result.eigenvalues.real).all()
    assert np.isfinite(result.eigenvalues.imag).all()


def test_write_periodic_current_sheet_eigenvalue_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_periodic_current_sheet_eigenvalue_validation(
        tmp_path,
        shape=(6, 6),
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("Dense tiny-grid")
    history = np.load(tmp_path / "periodic_current_sheet_eigenvalue.npz")
    assert history["schema"] == PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA
    assert history["matrix"].shape == (72, 72)
    assert history["eigenvalues_real"].shape == (72,)
    assert (
        tmp_path / "figures" / "periodic_current_sheet_spectrum.png"
    ).stat().st_size > 0

    outdir = tmp_path / "cli-current-sheet"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "current-sheet-eigenvalue",
            "--outdir",
            str(outdir),
            "--nx",
            "6",
            "--ny",
            "6",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation.json").exists()
