from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA,
    PERIODIC_CURRENT_SHEET_NONLINEAR_BRIDGE_SCHEMA,
    PERIODIC_CURRENT_SHEET_TIMEDOMAIN_SCHEMA,
    PERIODIC_DOUBLE_HARRIS_CONVERGENCE_SCHEMA,
    PERIODIC_DOUBLE_HARRIS_NONLINEAR_GROWTH_SCHEMA,
    PERIODIC_DOUBLE_HARRIS_SEEDED_LONG_RUN_SCHEMA,
    run_periodic_current_sheet_eigenvalue_validation,
    run_periodic_current_sheet_nonlinear_bridge_validation,
    run_periodic_current_sheet_timedomain_validation,
    run_periodic_double_harris_convergence_validation,
    run_periodic_double_harris_nonlinear_growth_validation,
    run_periodic_double_harris_seeded_long_run_validation,
    write_periodic_current_sheet_eigenvalue_validation,
    write_periodic_current_sheet_nonlinear_bridge_validation,
    write_periodic_current_sheet_timedomain_validation,
    write_periodic_double_harris_convergence_validation,
    write_periodic_double_harris_nonlinear_growth_validation,
    write_periodic_double_harris_seeded_long_run_validation,
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


def test_periodic_double_harris_nonlinear_growth_gate() -> None:
    result = run_periodic_double_harris_nonlinear_growth_validation(shape=(8, 8))
    assert result.diagnostics["schema"] == PERIODIC_DOUBLE_HARRIS_NONLINEAR_GROWTH_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.selected_eigenvalue.real > 0.0
    assert result.fitted_growth_rate > 0.0
    assert result.growth_factor > 2.0
    assert result.relative_growth_error < 0.15
    assert result.time.shape == result.perturbation_norm.shape
    assert result.perturbation_norm[-1] > result.perturbation_norm[0]


def test_periodic_double_harris_nonlinear_growth_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        run_periodic_double_harris_nonlinear_growth_validation(shape=(5, 8))
    with pytest.raises(ValueError, match="width"):
        run_periodic_double_harris_nonlinear_growth_validation(width=0.0)
    with pytest.raises(ValueError, match="amplitude"):
        run_periodic_double_harris_nonlinear_growth_validation(amplitude=0.0)
    with pytest.raises(ValueError, match="resistivity"):
        run_periodic_double_harris_nonlinear_growth_validation(resistivity=0.0)
    with pytest.raises(ValueError, match="viscosity"):
        run_periodic_double_harris_nonlinear_growth_validation(viscosity=0.0)
    with pytest.raises(ValueError, match="perturbation_amplitude"):
        run_periodic_double_harris_nonlinear_growth_validation(perturbation_amplitude=0.0)
    with pytest.raises(ValueError, match="dt"):
        run_periodic_double_harris_nonlinear_growth_validation(dt=0.0)
    with pytest.raises(ValueError, match="t_end"):
        run_periodic_double_harris_nonlinear_growth_validation(t_end=0.001)
    with pytest.raises(ValueError, match="save_every"):
        run_periodic_double_harris_nonlinear_growth_validation(save_every=0)
    with pytest.raises(ValueError, match="at least four"):
        run_periodic_double_harris_nonlinear_growth_validation(t_end=0.1, save_every=100)
    with pytest.raises(ValueError, match="min_linear_growth_rate"):
        run_periodic_double_harris_nonlinear_growth_validation(min_linear_growth_rate=0.0)
    with pytest.raises(ValueError, match="min_nonlinear_growth_factor"):
        run_periodic_double_harris_nonlinear_growth_validation(
            min_nonlinear_growth_factor=1.0
        )
    with pytest.raises(ValueError, match="max_relative_growth_error"):
        run_periodic_double_harris_nonlinear_growth_validation(max_relative_growth_error=0.0)
    with pytest.raises(ValueError, match="max_selected_residual_norm"):
        run_periodic_double_harris_nonlinear_growth_validation(
            max_selected_residual_norm=0.0
        )
    with pytest.raises(ValueError, match="max_selected_eigenvalue_imag"):
        run_periodic_double_harris_nonlinear_growth_validation(
            max_selected_eigenvalue_imag=0.0
        )
    with pytest.raises(RuntimeError, match="no unstable"):
        run_periodic_double_harris_nonlinear_growth_validation(
            shape=(8, 8),
            min_linear_growth_rate=10.0,
        )


def test_periodic_double_harris_seeded_long_run_gate() -> None:
    result = run_periodic_double_harris_seeded_long_run_validation(
        shape=(16, 16),
        t_end=10.0,
        save_every=100,
        fit_window=(0.0, 6.0),
        min_max_growth_factor=1.2,
    )
    assert result.diagnostics["schema"] == PERIODIC_DOUBLE_HARRIS_SEEDED_LONG_RUN_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.time.shape == result.perturbation_norm.shape
    assert result.magnetic_energy.shape == result.time.shape
    assert result.kinetic_energy.shape == result.time.shape
    assert result.total_energy.shape == result.time.shape
    assert result.current_density_linf.shape == result.time.shape
    assert result.fitted_early_growth_rate > 0.0
    assert result.early_growth_factor > 1.0
    assert np.max(result.total_energy) <= result.total_energy[0] * (1.0 + 1.0e-8)


def test_periodic_double_harris_seeded_long_run_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        run_periodic_double_harris_seeded_long_run_validation(shape=(6, 8))
    with pytest.raises(ValueError, match="width"):
        run_periodic_double_harris_seeded_long_run_validation(width=0.0)
    with pytest.raises(ValueError, match="amplitude"):
        run_periodic_double_harris_seeded_long_run_validation(amplitude=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        run_periodic_double_harris_seeded_long_run_validation(resistivity=-1.0)
    with pytest.raises(ValueError, match="perturbation_amplitude"):
        run_periodic_double_harris_seeded_long_run_validation(perturbation_amplitude=0.0)
    with pytest.raises(ValueError, match="perturbation_mode"):
        run_periodic_double_harris_seeded_long_run_validation(perturbation_mode=(0, 0))
    with pytest.raises(ValueError, match="dt"):
        run_periodic_double_harris_seeded_long_run_validation(dt=0.0)
    with pytest.raises(ValueError, match="t_end"):
        run_periodic_double_harris_seeded_long_run_validation(t_end=0.0)
    with pytest.raises(ValueError, match="at least two"):
        run_periodic_double_harris_seeded_long_run_validation(t_end=0.001, dt=0.01)
    with pytest.raises(ValueError, match="save_every"):
        run_periodic_double_harris_seeded_long_run_validation(save_every=0)
    with pytest.raises(ValueError, match="ordered"):
        run_periodic_double_harris_seeded_long_run_validation(fit_window=(2.0, 1.0))
    with pytest.raises(ValueError, match="must not exceed"):
        run_periodic_double_harris_seeded_long_run_validation(fit_window=(0.0, 31.0))
    with pytest.raises(ValueError, match="at least three"):
        run_periodic_double_harris_seeded_long_run_validation(
            t_end=2.0,
            save_every=200,
            fit_window=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="min_saved_samples"):
        run_periodic_double_harris_seeded_long_run_validation(min_saved_samples=2)
    with pytest.raises(ValueError, match="min_early_growth_rate"):
        run_periodic_double_harris_seeded_long_run_validation(min_early_growth_rate=0.0)
    with pytest.raises(ValueError, match="min_early_growth_factor"):
        run_periodic_double_harris_seeded_long_run_validation(min_early_growth_factor=1.0)
    with pytest.raises(ValueError, match="min_max_growth_factor"):
        run_periodic_double_harris_seeded_long_run_validation(min_max_growth_factor=1.0)
    with pytest.raises(ValueError, match="max_relative_energy_increase"):
        run_periodic_double_harris_seeded_long_run_validation(
            max_relative_energy_increase=-1.0
        )


def test_periodic_double_harris_convergence_gate() -> None:
    result = run_periodic_double_harris_convergence_validation(
        resolutions=(16, 18),
        dt_values=(0.02, 0.01),
        reference_resolution=16,
        t_end=6.0,
        fit_window=(0.0, 3.0),
        max_relative_growth_rate_spread=2.0,
    )
    assert result.diagnostics["schema"] == PERIODIC_DOUBLE_HARRIS_CONVERGENCE_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.case_kind.shape == (4,)
    assert result.resolution.shape == result.dt.shape
    assert result.fitted_early_growth_rate.shape == result.dt.shape
    assert np.isfinite(result.fitted_early_growth_rate).all()
    assert np.isfinite(result.max_growth_factor).all()
    assert set(result.case_kind.tolist()) == {"resolution", "timestep"}
    assert np.all(result.fitted_early_growth_rate > 0.0)
    assert np.all(result.max_growth_factor > 1.0)


def test_periodic_double_harris_convergence_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="resolutions"):
        run_periodic_double_harris_convergence_validation(resolutions=(16,))
    with pytest.raises(ValueError, match="resolutions"):
        run_periodic_double_harris_convergence_validation(resolutions=(6, 16))
    with pytest.raises(ValueError, match="unique"):
        run_periodic_double_harris_convergence_validation(resolutions=(16, 16))
    with pytest.raises(ValueError, match="dt_values"):
        run_periodic_double_harris_convergence_validation(dt_values=(0.01,))
    with pytest.raises(ValueError, match="positive"):
        run_periodic_double_harris_convergence_validation(dt_values=(0.02, 0.0))
    with pytest.raises(ValueError, match="unique"):
        run_periodic_double_harris_convergence_validation(dt_values=(0.02, 0.02))
    with pytest.raises(ValueError, match="reference_resolution"):
        run_periodic_double_harris_convergence_validation(reference_resolution=6)
    with pytest.raises(ValueError, match="reference_dt"):
        run_periodic_double_harris_convergence_validation(reference_dt=0.0)
    with pytest.raises(ValueError, match="t_end"):
        run_periodic_double_harris_convergence_validation(t_end=0.0)
    with pytest.raises(ValueError, match="save_interval"):
        run_periodic_double_harris_convergence_validation(save_interval=0.0)
    with pytest.raises(ValueError, match="must not exceed"):
        run_periodic_double_harris_convergence_validation(save_interval=9.0, t_end=8.0)
    with pytest.raises(ValueError, match="fit_window"):
        run_periodic_double_harris_convergence_validation(fit_window=(2.0, 1.0))
    with pytest.raises(ValueError, match="must not exceed"):
        run_periodic_double_harris_convergence_validation(fit_window=(0.0, 9.0))
    with pytest.raises(ValueError, match="at least three"):
        run_periodic_double_harris_convergence_validation(
            t_end=2.0,
            fit_window=(0.0, 1.0),
            save_interval=1.0,
        )
    with pytest.raises(ValueError, match="min_saved_samples"):
        run_periodic_double_harris_convergence_validation(min_saved_samples=2)
    with pytest.raises(ValueError, match="min_early_growth_rate"):
        run_periodic_double_harris_convergence_validation(min_early_growth_rate=0.0)
    with pytest.raises(ValueError, match="min_early_growth_factor"):
        run_periodic_double_harris_convergence_validation(min_early_growth_factor=1.0)
    with pytest.raises(ValueError, match="min_max_growth_factor"):
        run_periodic_double_harris_convergence_validation(min_max_growth_factor=1.0)
    with pytest.raises(ValueError, match="max_relative_energy_increase"):
        run_periodic_double_harris_convergence_validation(
            max_relative_energy_increase=-1.0
        )
    with pytest.raises(ValueError, match="max_relative_growth_rate_spread"):
        run_periodic_double_harris_convergence_validation(
            max_relative_growth_rate_spread=0.0
        )
    with pytest.raises(ValueError, match="max_relative_max_growth_spread"):
        run_periodic_double_harris_convergence_validation(
            max_relative_max_growth_spread=0.0
        )


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


def test_write_periodic_double_harris_nonlinear_growth_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_periodic_double_harris_nonlinear_growth_validation(
        tmp_path,
        shape=(8, 8),
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("Nonlinear periodic")
    history = np.load(tmp_path / "periodic_double_harris_nonlinear_growth.npz")
    assert history["schema"] == PERIODIC_DOUBLE_HARRIS_NONLINEAR_GROWTH_SCHEMA
    assert history["time"].shape == history["perturbation_norm"].shape
    assert history["growth_factor"] > 2.0
    assert history["base_initial_psi"].shape == (8, 8)
    assert (
        tmp_path / "figures" / "periodic_double_harris_nonlinear_growth.png"
    ).stat().st_size > 0

    outdir = tmp_path / "cli-double-harris"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "double-harris-growth",
            "--outdir",
            str(outdir),
            "--nx",
            "8",
            "--ny",
            "8",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation.json").exists()


def test_write_periodic_double_harris_seeded_long_run_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_periodic_double_harris_seeded_long_run_validation(
        tmp_path,
        shape=(16, 16),
        t_end=10.0,
        save_every=100,
        fit_window=(0.0, 6.0),
        min_max_growth_factor=1.2,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("Longer seeded")
    history = np.load(tmp_path / "periodic_double_harris_seeded_long_run.npz")
    assert history["schema"] == PERIODIC_DOUBLE_HARRIS_SEEDED_LONG_RUN_SCHEMA
    assert history["time"].shape == history["perturbation_norm"].shape
    assert history["total_energy"].shape == history["time"].shape
    assert history["current_density_linf"].shape == history["time"].shape
    assert history["perturbed_psi"].shape[1:] == (16, 16)
    assert (
        tmp_path / "figures" / "periodic_double_harris_seeded_long_run.png"
    ).stat().st_size > 0

    outdir = tmp_path / "cli-double-harris-long"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "double-harris-long-run",
            "--outdir",
            str(outdir),
            "--nx",
            "16",
            "--ny",
            "16",
            "--t-end",
            "10",
            "--save-every",
            "100",
            "--fit-stop",
            "6",
            "--min-max-growth-factor",
            "1.2",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation.json").exists()


def test_write_periodic_double_harris_seeded_long_run_movies(tmp_path) -> None:
    manifest_path, validation = write_periodic_double_harris_seeded_long_run_validation(
        tmp_path,
        shape=(16, 16),
        t_end=10.0,
        save_every=100,
        fit_window=(0.0, 6.0),
        min_max_growth_factor=1.2,
        movies=True,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert (tmp_path / "figures" / "periodic_double_harris_flux.gif").stat().st_size > 0
    assert (
        tmp_path / "figures" / "periodic_double_harris_current.gif"
    ).stat().st_size > 0


def test_write_periodic_double_harris_convergence_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_periodic_double_harris_convergence_validation(
        tmp_path,
        resolutions=(16, 18),
        dt_values=(0.02, 0.01),
        reference_resolution=16,
        t_end=6.0,
        fit_window=(0.0, 3.0),
        max_relative_growth_rate_spread=2.0,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["references"]["scope"].startswith("Tiny deterministic")
    history = np.load(tmp_path / "periodic_double_harris_convergence.npz")
    assert history["schema"] == PERIODIC_DOUBLE_HARRIS_CONVERGENCE_SCHEMA
    assert history["case_kind"].shape == (4,)
    assert history["resolution"].shape == history["dt"].shape
    assert history["fitted_early_growth_rate"].shape == history["dt"].shape
    assert (
        tmp_path / "figures" / "periodic_double_harris_convergence.png"
    ).stat().st_size > 0

    outdir = tmp_path / "cli-double-harris-convergence"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "double-harris-convergence",
            "--outdir",
            str(outdir),
            "--resolutions",
            "16,18",
            "--dt-values",
            "0.02,0.01",
            "--reference-resolution",
            "16",
            "--t-end",
            "6",
            "--fit-stop",
            "3",
            "--max-relative-growth-rate-spread",
            "2.0",
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
