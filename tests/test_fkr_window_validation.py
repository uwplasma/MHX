from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    FKR_GROWTH_RATE_SCHEMA,
    FKR_WINDOW_SCHEMA,
    HARRIS_DELTA_PRIME_SCHEMA,
    run_fkr_growth_rate_validation,
    run_fkr_window_validation,
    run_harris_delta_prime_validation,
    write_fkr_growth_rate_validation,
    write_fkr_window_validation,
    write_harris_delta_prime_validation,
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


def test_harris_delta_prime_validation_matches_outer_formula() -> None:
    result = run_harris_delta_prime_validation(steps=2500)
    assert result.diagnostics["schema"] == HARRIS_DELTA_PRIME_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert np.all(result.numerical_delta_prime_a > 0.0)
    assert np.all(np.diff(result.numerical_delta_prime_a) < 0.0)
    assert np.max(result.relative_error) < 1.0e-8


def test_fkr_growth_rate_validation_gates_asymptotic_exponents() -> None:
    result = run_fkr_growth_rate_validation(steps=2500)
    assert result.diagnostics["schema"] == FKR_GROWTH_RATE_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert np.all(result.gamma_vs_lundquist > 0.0)
    assert np.all(result.gamma_vs_delta_prime > 0.0)
    assert result.diagnostics["lundquist_slope"] == pytest.approx(-3.0 / 5.0)
    assert result.diagnostics["delta_prime_slope"] == pytest.approx(4.0 / 5.0)
    assert np.max(result.gamma_relative_error) < 1.0e-7


def test_fkr_growth_rate_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least three lundquist"):
        run_fkr_growth_rate_validation(lundquist=(1.0e4, 1.0e5))
    with pytest.raises(ValueError, match="lundquist samples must be positive"):
        run_fkr_growth_rate_validation(lundquist=(0.0, 1.0e4, 1.0e5))
    with pytest.raises(ValueError, match="strictly increasing"):
        run_fkr_growth_rate_validation(lundquist=(1.0e4, 1.0e5, 3.0e4))
    with pytest.raises(ValueError, match="fixed_ka"):
        run_fkr_growth_rate_validation(fixed_ka=1.0)
    with pytest.raises(ValueError, match="fixed_lundquist"):
        run_fkr_growth_rate_validation(fixed_lundquist=0.0)
    result = run_fkr_growth_rate_validation(max_constant_psi_product=1.0e-9)
    assert result.validation["passed"] is False
    assert result.validation["checks"]["constant_psi_window"] is False


def test_harris_delta_prime_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least three"):
        run_harris_delta_prime_validation(ka=(0.2, 0.3))
    with pytest.raises(ValueError, match="positive"):
        run_harris_delta_prime_validation(ka=(0.0, 0.2, 0.3))
    with pytest.raises(ValueError, match="ka < 1"):
        run_harris_delta_prime_validation(ka=(0.2, 0.8, 1.1))
    with pytest.raises(ValueError, match="strictly increasing"):
        run_harris_delta_prime_validation(ka=(0.2, 0.4, 0.3))
    with pytest.raises(ValueError, match="xmax_over_a"):
        run_harris_delta_prime_validation(xmax_over_a=1.0)
    with pytest.raises(ValueError, match="steps"):
        run_harris_delta_prime_validation(steps=99)


def test_write_harris_delta_prime_validation_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_harris_delta_prime_validation(tmp_path, steps=2500)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["harris_outer"].startswith("Harris-sheet")
    history = np.load(tmp_path / "harris_delta_prime.npz")
    assert history["schema"] == HARRIS_DELTA_PRIME_SCHEMA
    assert history["ka"].shape[0] == 5
    assert (tmp_path / "figures" / "harris_delta_prime.png").stat().st_size > 0

    outdir = tmp_path / "cli-delta"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "harris-delta-prime",
            "--outdir",
            str(outdir),
            "--steps",
            "2500",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()


def test_write_fkr_growth_rate_validation_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_fkr_growth_rate_validation(tmp_path, steps=2500)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["fkr_growth"].startswith("Furth")
    history = np.load(tmp_path / "fkr_growth_rate.npz")
    assert history["schema"] == FKR_GROWTH_RATE_SCHEMA
    assert history["lundquist"].shape[0] == 5
    assert history["ka"].shape[0] == 5
    assert (tmp_path / "figures" / "fkr_growth_rate.png").stat().st_size > 0

    outdir = tmp_path / "cli-growth"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "fkr-growth",
            "--outdir",
            str(outdir),
            "--fixed-ka",
            "0.35",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()


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
