from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    RESISTIVE_DECAY_SCHEMA,
    resistive_decay_rate,
    run_resistive_decay_validation,
    write_resistive_decay_validation,
)
from mhx.cli.main import app


def test_resistive_decay_rate_matches_eta_k_squared() -> None:
    rate = resistive_decay_rate(resistivity=0.05, mode=(2, 1), lengths=(2.0 * np.pi, np.pi))
    assert rate == pytest.approx(0.05 * (2.0**2 + 2.0**2))


def test_resistive_decay_validation_has_physics_gates() -> None:
    result = run_resistive_decay_validation(shape=(32, 32), resistivity=0.05, t1=0.5, dt=0.01)
    diagnostics = result.diagnostics
    validation = result.validation
    assert diagnostics["schema"] == RESISTIVE_DECAY_SCHEMA
    assert diagnostics["decay_rate_eta_k_squared"] == pytest.approx(0.05)
    assert diagnostics["fitted_decay_rate"] == pytest.approx(0.05, rel=1.0e-7)
    assert diagnostics["max_relative_amplitude_error"] < 1.0e-8
    assert diagnostics["max_relative_energy_error"] < 1.0e-8
    assert validation["passed"] is True
    assert validation["checks"]["amplitude_matches_exp_minus_eta_k2_t"] is True
    assert validation["checks"]["energy_matches_exp_minus_2_eta_k2_t"] is True
    assert validation["checks"]["magnetic_energy_monotone_nonincreasing"] is True


def test_resistive_decay_validation_can_fail_strict_gate() -> None:
    result = run_resistive_decay_validation(
        shape=(16, 16),
        resistivity=0.05,
        t1=0.5,
        dt=0.05,
        max_relative_amplitude_error=1.0e-16,
    )
    assert result.validation["passed"] is False


def test_write_resistive_decay_validation_artifacts(tmp_path) -> None:
    manifest_path, validation = write_resistive_decay_validation(
        tmp_path,
        shape=(16, 16),
        t1=0.2,
        dt=0.01,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["reference"] == (
        "single Fourier mode of resistive reduced-MHD induction equation"
    )
    history = np.load(tmp_path / "decay_history.npz")
    assert history["schema"] == RESISTIVE_DECAY_SCHEMA
    assert history["time"].shape == history["numerical_amplitude"].shape
    assert (tmp_path / "figures" / "decay_amplitude.png").stat().st_size > 0
    assert (tmp_path / "figures" / "decay_relative_error.png").stat().st_size > 0
    assert (tmp_path / "figures" / "decay_energy.png").stat().st_size > 0


def test_resistive_decay_cli(tmp_path) -> None:
    outdir = tmp_path / "decay"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "decay",
            "--outdir",
            str(outdir),
            "--nx",
            "16",
            "--ny",
            "16",
            "--t1",
            "0.2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    validation = json.loads((outdir / "validation.json").read_text())
    assert validation["passed"] is True
