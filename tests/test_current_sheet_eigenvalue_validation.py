from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA,
    PERIODIC_CURRENT_SHEET_NONLINEAR_BRIDGE_SCHEMA,
    PERIODIC_CURRENT_SHEET_TIMEDOMAIN_SCHEMA,
    run_periodic_current_sheet_eigenvalue_validation,
    run_periodic_current_sheet_nonlinear_bridge_validation,
    run_periodic_current_sheet_timedomain_validation,
    write_periodic_current_sheet_eigenvalue_validation,
    write_periodic_current_sheet_nonlinear_bridge_validation,
    write_periodic_current_sheet_timedomain_validation,
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


def test_periodic_current_sheet_timedomain_replays_eigenmode() -> None:
    result = run_periodic_current_sheet_timedomain_validation(shape=(6, 6), t_end=2.5)
    assert result.diagnostics["schema"] == PERIODIC_CURRENT_SHEET_TIMEDOMAIN_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.times.shape == result.amplitudes.shape
    assert result.exact_amplitudes.shape == result.amplitudes.shape
    assert result.relative_state_error.shape == result.amplitudes.shape
    assert result.selected_eigenvalue < 0.0
    np.testing.assert_allclose(
        result.fitted_decay_rate,
        result.selected_eigenvalue,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert float(np.max(result.relative_state_error)) < 1.0e-8
    assert result.amplitudes[-1] < result.amplitudes[0]


def test_periodic_current_sheet_timedomain_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        run_periodic_current_sheet_timedomain_validation(shape=(3, 6))
    with pytest.raises(ValueError, match="dt"):
        run_periodic_current_sheet_timedomain_validation(dt=0.0)
    with pytest.raises(ValueError, match="t_end"):
        run_periodic_current_sheet_timedomain_validation(t_end=0.01)
    with pytest.raises(ValueError, match="save_every"):
        run_periodic_current_sheet_timedomain_validation(save_every=0)
    with pytest.raises(ValueError, match="real_eigenvalue_imag_tolerance"):
        run_periodic_current_sheet_timedomain_validation(real_eigenvalue_imag_tolerance=0.0)
    with pytest.raises(RuntimeError, match="no real decaying eigenpair"):
        run_periodic_current_sheet_timedomain_validation(shape=(6, 6), min_decay_rate=10.0)


def test_periodic_current_sheet_nonlinear_bridge_is_second_order() -> None:
    result = run_periodic_current_sheet_nonlinear_bridge_validation(shape=(6, 6))
    assert result.diagnostics["schema"] == PERIODIC_CURRENT_SHEET_NONLINEAR_BRIDGE_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.epsilons.shape == result.relative_errors.shape
    assert np.all(np.diff(result.relative_errors) < 0.0)
    np.testing.assert_allclose(
        result.convergence_order,
        2.0,
        rtol=0.02,
        atol=0.02,
    )
    assert result.finest_relative_error < 1.0e-7
    assert result.tangent_norm > 0.0


def test_periodic_current_sheet_nonlinear_bridge_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        run_periodic_current_sheet_nonlinear_bridge_validation(shape=(3, 6))
    with pytest.raises(ValueError, match="dt"):
        run_periodic_current_sheet_nonlinear_bridge_validation(dt=0.0)
    with pytest.raises(ValueError, match="steps"):
        run_periodic_current_sheet_nonlinear_bridge_validation(steps=1)
    with pytest.raises(ValueError, match="save_every"):
        run_periodic_current_sheet_nonlinear_bridge_validation(save_every=0)
    with pytest.raises(ValueError, match="at least three"):
        run_periodic_current_sheet_nonlinear_bridge_validation(epsilons=(1.0e-2, 5.0e-3))
    with pytest.raises(ValueError, match="positive"):
        run_periodic_current_sheet_nonlinear_bridge_validation(
            epsilons=(1.0e-2, 0.0, 1.0e-3)
        )
    with pytest.raises(ValueError, match="strictly decreasing"):
        run_periodic_current_sheet_nonlinear_bridge_validation(
            epsilons=(1.0e-2, 2.0e-2, 1.0e-3)
        )
    with pytest.raises(ValueError, match="min_convergence_order"):
        run_periodic_current_sheet_nonlinear_bridge_validation(min_convergence_order=0.0)
    with pytest.raises(ValueError, match="max_finest_relative_error"):
        run_periodic_current_sheet_nonlinear_bridge_validation(max_finest_relative_error=0.0)


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


def test_write_periodic_current_sheet_timedomain_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_periodic_current_sheet_timedomain_validation(
        tmp_path,
        shape=(6, 6),
        t_end=2.5,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("Linear time-domain")
    history = np.load(tmp_path / "periodic_current_sheet_timedomain.npz")
    assert history["schema"] == PERIODIC_CURRENT_SHEET_TIMEDOMAIN_SCHEMA
    assert history["time"].shape == history["amplitude"].shape
    assert history["relative_decay_rate_error"] < 1.0e-8
    assert history["initial_psi"].shape == (6, 6)
    assert (
        tmp_path / "figures" / "periodic_current_sheet_timedomain.png"
    ).stat().st_size > 0

    outdir = tmp_path / "cli-current-sheet-timedomain"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "current-sheet-timedomain",
            "--outdir",
            str(outdir),
            "--nx",
            "6",
            "--ny",
            "6",
            "--t-end",
            "2.5",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation.json").exists()


def test_write_periodic_current_sheet_nonlinear_bridge_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_periodic_current_sheet_nonlinear_bridge_validation(
        tmp_path,
        shape=(6, 6),
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("Nonlinear reduced-MHD")
    history = np.load(tmp_path / "periodic_current_sheet_nonlinear_bridge.npz")
    assert history["schema"] == PERIODIC_CURRENT_SHEET_NONLINEAR_BRIDGE_SCHEMA
    assert history["epsilon"].shape == history["relative_error"].shape
    assert history["convergence_order"] > 1.8
    assert history["tangent_final_psi"].shape == (6, 6)
    assert (
        tmp_path / "figures" / "periodic_current_sheet_nonlinear_bridge.png"
    ).stat().st_size > 0

    outdir = tmp_path / "cli-current-sheet-nonlinear-bridge"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "current-sheet-nonlinear-bridge",
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
