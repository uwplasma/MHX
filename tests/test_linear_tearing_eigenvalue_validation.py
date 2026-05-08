from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    LINEAR_TEARING_DISPERSION_SCHEMA,
    LINEAR_TEARING_EIGENVALUE_SCHEMA,
    LINEAR_TEARING_TIMEDOMAIN_SCHEMA,
    run_linear_tearing_dispersion_validation,
    run_linear_tearing_eigenvalue_validation,
    run_linear_tearing_timedomain_validation,
    write_linear_tearing_dispersion_validation,
    write_linear_tearing_eigenvalue_validation,
    write_linear_tearing_timedomain_validation,
)
from mhx.cli.main import app


def test_linear_tearing_eigenvalue_matches_literature_gate() -> None:
    result = run_linear_tearing_eigenvalue_validation()
    assert result.diagnostics["schema"] == LINEAR_TEARING_EIGENVALUE_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.selected_eigenvalue.real == pytest.approx(0.0131, rel=0.03)
    assert result.extrapolated_growth_rate == pytest.approx(0.0131, rel=0.01)
    assert abs(result.selected_eigenvalue.imag) < 1.0e-10
    assert result.selected_residual_norm < 1.0e-10
    assert result.stable_control_max_real_part < 0.0
    assert result.stable_control_residual_norm < 1.0e-10
    assert np.all(np.diff(result.growth_rates) < 0.0)
    assert result.flux_even_correlation > 0.995
    assert result.stream_odd_correlation > 0.995


def test_linear_tearing_eigenvalue_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least three"):
        run_linear_tearing_eigenvalue_validation(grid_points=(128, 160))
    with pytest.raises(ValueError, match="at least 64"):
        run_linear_tearing_eigenvalue_validation(grid_points=(32, 96, 128))
    with pytest.raises(ValueError, match="strictly increasing"):
        run_linear_tearing_eigenvalue_validation(grid_points=(128, 192, 160))
    with pytest.raises(ValueError, match="half_width"):
        run_linear_tearing_eigenvalue_validation(half_width=1.0)
    with pytest.raises(ValueError, match="lundquist"):
        run_linear_tearing_eigenvalue_validation(lundquist=0.0)
    with pytest.raises(ValueError, match="0 < k < 1"):
        run_linear_tearing_eigenvalue_validation(wavenumber=1.2)
    with pytest.raises(ValueError, match="reference_growth_rate"):
        run_linear_tearing_eigenvalue_validation(reference_growth_rate=0.0)
    with pytest.raises(ValueError, match="stable_control_wavenumber"):
        run_linear_tearing_eigenvalue_validation(stable_control_wavenumber=0.8)
    with pytest.raises(ValueError, match="stable_control_grid_points"):
        run_linear_tearing_eigenvalue_validation(stable_control_grid_points=32)


def test_linear_tearing_dispersion_gates_unstable_band_and_controls() -> None:
    result = run_linear_tearing_dispersion_validation()
    assert result.diagnostics["schema"] == LINEAR_TEARING_DISPERSION_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert np.all(result.growth_rate[result.unstable_band_mask] > 0.0)
    assert np.all(np.diff(result.growth_rate[result.unstable_band_mask]) < 0.0)
    assert np.all(result.growth_rate[result.stable_control_mask] < 0.0)
    assert result.measured_reference_growth_rate == pytest.approx(0.0131, rel=0.06)
    assert np.max(result.residual_norm) < 1.0e-10


def test_linear_tearing_dispersion_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="grid_points"):
        run_linear_tearing_dispersion_validation(grid_points=32)
    with pytest.raises(ValueError, match="half_width"):
        run_linear_tearing_dispersion_validation(half_width=1.0)
    with pytest.raises(ValueError, match="lundquist"):
        run_linear_tearing_dispersion_validation(lundquist=0.0)
    with pytest.raises(ValueError, match="at least four"):
        run_linear_tearing_dispersion_validation(wavenumber=(0.3, 0.5, 1.1))
    with pytest.raises(ValueError, match="positive"):
        run_linear_tearing_dispersion_validation(wavenumber=(0.0, 0.5, 0.7, 1.1))
    with pytest.raises(ValueError, match="strictly increasing"):
        run_linear_tearing_dispersion_validation(wavenumber=(0.3, 0.7, 0.5, 1.1))
    with pytest.raises(ValueError, match="below and above"):
        run_linear_tearing_dispersion_validation(wavenumber=(0.3, 0.5, 0.7, 0.9))
    with pytest.raises(ValueError, match="reference_growth_rate"):
        run_linear_tearing_dispersion_validation(reference_growth_rate=0.0)
    with pytest.raises(ValueError, match="reference_wavenumber"):
        run_linear_tearing_dispersion_validation(reference_wavenumber=0.4)


def test_linear_tearing_timedomain_recovers_eigenmode_growth() -> None:
    result = run_linear_tearing_timedomain_validation(grid_points=96, t_end=40.0)
    assert result.diagnostics["schema"] == LINEAR_TEARING_TIMEDOMAIN_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.times.shape == result.amplitude.shape
    assert result.exact_amplitude.shape == result.amplitude.shape
    assert result.fitted_growth_rate == pytest.approx(result.expected_growth_rate, rel=1.0e-5)
    assert result.max_relative_amplitude_error < 1.0e-4
    assert result.final_mode_alignment > 0.999999
    assert result.selected_residual_norm < 1.0e-10


def test_linear_tearing_timedomain_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="grid_points"):
        run_linear_tearing_timedomain_validation(grid_points=32)
    with pytest.raises(ValueError, match="half_width"):
        run_linear_tearing_timedomain_validation(half_width=1.0)
    with pytest.raises(ValueError, match="lundquist"):
        run_linear_tearing_timedomain_validation(lundquist=0.0)
    with pytest.raises(ValueError, match="0 < k < 1"):
        run_linear_tearing_timedomain_validation(wavenumber=1.2)
    with pytest.raises(ValueError, match="dt"):
        run_linear_tearing_timedomain_validation(dt=0.0)
    with pytest.raises(ValueError, match="t_end"):
        run_linear_tearing_timedomain_validation(t_end=0.5, dt=0.25)
    with pytest.raises(ValueError, match="fit_start_fraction"):
        run_linear_tearing_timedomain_validation(fit_start_fraction=1.0)


def test_write_linear_tearing_eigenvalue_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_linear_tearing_eigenvalue_validation(tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["mactaggart_2019"].startswith("MacTaggart")
    history = np.load(tmp_path / "linear_tearing_eigenvalue.npz")
    assert history["schema"] == LINEAR_TEARING_EIGENVALUE_SCHEMA
    assert history["grid_points"].shape[0] == 3
    assert history["stable_control_max_real_part"] < 0.0
    assert history["spectrum_real"].shape == history["spectrum_imag"].shape
    assert (tmp_path / "figures" / "linear_tearing_eigenvalue.png").stat().st_size > 0

    outdir = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "linear-tearing-eigenvalue",
            "--outdir",
            str(outdir),
            "--grid-points",
            "192,256,320",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()


def test_write_linear_tearing_timedomain_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_linear_tearing_timedomain_validation(
        tmp_path,
        grid_points=96,
        t_end=40.0,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("This is a linear")
    history = np.load(tmp_path / "linear_tearing_timedomain.npz")
    assert history["schema"] == LINEAR_TEARING_TIMEDOMAIN_SCHEMA
    assert history["time"].shape == history["amplitude"].shape
    assert history["relative_growth_error"] < 1.0e-5
    assert (tmp_path / "figures" / "linear_tearing_timedomain.png").stat().st_size > 0

    outdir = tmp_path / "cli-timedomain"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "linear-tearing-timedomain",
            "--outdir",
            str(outdir),
            "--grid-points",
            "96",
            "--t-end",
            "40.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()


def test_write_linear_tearing_dispersion_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_linear_tearing_dispersion_validation(tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["fkr_coppi_scope"].startswith("The scan checks")
    history = np.load(tmp_path / "linear_tearing_dispersion.npz")
    assert history["schema"] == LINEAR_TEARING_DISPERSION_SCHEMA
    assert history["wavenumber"].shape[0] == 6
    assert np.all(history["residual_norm"] < 1.0e-10)
    assert (tmp_path / "figures" / "linear_tearing_dispersion.png").stat().st_size > 0

    outdir = tmp_path / "cli-dispersion"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "linear-tearing-dispersion",
            "--outdir",
            str(outdir),
            "--wavenumber",
            "0.3,0.5,0.7,0.9,1.1,1.2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "validation.json").exists()
