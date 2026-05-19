from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    DECAYING_MHD_TURBULENCE_SCHEMA,
    FORCED_TURBULENT_RECONNECTION_READINESS_SCHEMA,
    FORCED_TURBULENT_RECONNECTION_SCHEMA,
    assess_forced_turbulent_reconnection_readiness,
    run_decaying_mhd_turbulence_validation,
    run_forced_turbulent_reconnection_validation,
    write_decaying_mhd_turbulence_validation,
    write_forced_turbulent_reconnection_readiness_report,
    write_forced_turbulent_reconnection_validation,
)
from mhx.benchmarks.turbulence import (
    _critical_flux_separation,
    _high_k_fraction,
    _sample_indices,
    _write_scalar_movie,
)
from mhx.cli.main import app


def test_decaying_mhd_turbulence_validation_gate() -> None:
    result = run_decaying_mhd_turbulence_validation(
        shape=(16, 16),
        resistivity=2.0e-2,
        viscosity=2.0e-2,
        t_end=0.5,
        save_every=10,
    )

    assert result.schema == DECAYING_MHD_TURBULENCE_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.psi.shape == result.omega.shape == result.current_density.shape
    assert result.psi.shape[0] == result.time.size
    assert result.total_energy[-1] < result.total_energy[0]
    assert result.diagnostics["current_linf_growth"] > 0.0


def test_forced_turbulent_reconnection_validation_gate() -> None:
    result = run_forced_turbulent_reconnection_validation(
        shape=(16, 16),
        t_end=1.0,
        save_every=10,
        max_relative_energy_growth=10.0,
    )

    assert result.schema == FORCED_TURBULENT_RECONNECTION_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.reconnection_proxy is not None
    assert result.reconnection_rate_proxy is not None
    assert np.isfinite(result.reconnection_rate_proxy).all()
    assert result.diagnostics["reconnection_proxy_change"] > 0.0


def test_turbulence_writers_and_cli(tmp_path) -> None:
    decay_manifest, decay_validation = write_decaying_mhd_turbulence_validation(
        tmp_path / "decay",
        shape=(16, 16),
        t_end=0.5,
        save_every=10,
        movies=True,
    )
    forced_manifest, forced_validation = write_forced_turbulent_reconnection_validation(
        tmp_path / "forced",
        shape=(16, 16),
        t_end=1.0,
        save_every=10,
        max_relative_energy_growth=10.0,
        movies=True,
    )

    assert decay_validation["passed"] is True
    assert forced_validation["passed"] is True
    assert json.loads((tmp_path / "decay" / "diagnostics.json").read_text())["schema"] == (
        DECAYING_MHD_TURBULENCE_SCHEMA
    )
    assert json.loads((tmp_path / "forced" / "diagnostics.json").read_text())["schema"] == (
        FORCED_TURBULENT_RECONNECTION_SCHEMA
    )
    for manifest in (decay_manifest, forced_manifest):
        payload = json.loads(manifest.read_text())
        assert payload["claim_level"] == "validation"
        assert (manifest.parent / payload["outputs"]["history"]).exists()
        assert (manifest.parent / payload["outputs"]["current_movie"]).exists()
        assert (manifest.parent / payload["outputs"]["flux_movie"]).exists()

    runner = CliRunner()
    cli_decay = runner.invoke(
        app,
        [
            "benchmark",
            "decaying-turbulence",
            "--outdir",
            str(tmp_path / "cli_decay"),
            "--nx",
            "16",
            "--ny",
            "16",
            "--t-end",
            "0.5",
        ],
    )
    assert cli_decay.exit_code == 0, cli_decay.stdout
    cli_forced = runner.invoke(
        app,
        [
            "benchmark",
            "forced-turbulent-reconnection",
            "--outdir",
            str(tmp_path / "cli_forced"),
            "--nx",
            "16",
            "--ny",
            "16",
            "--t-end",
            "1.0",
            "--save-every",
            "10",
            "--max-relative-energy-growth",
            "10.0",
        ],
    )
    assert cli_forced.exit_code == 0, cli_forced.stdout


def test_forced_turbulent_reconnection_readiness_report_and_cli(tmp_path) -> None:
    run_dir = tmp_path / "forced"
    write_forced_turbulent_reconnection_validation(
        run_dir,
        shape=(16, 16),
        t_end=1.0,
        save_every=10,
        max_relative_energy_growth=10.0,
        movies=True,
    )

    assessment = assess_forced_turbulent_reconnection_readiness(
        run_dir,
        require_movies=True,
        min_history_samples=5,
        min_t_end=1.0,
        max_relative_energy_growth=10.0,
    )
    assert assessment.promotion_ready is True
    assert assessment.diagnostics["schema"] == FORCED_TURBULENT_RECONNECTION_READINESS_SCHEMA
    assert assessment.validation["checks"]["nontrivial_reconnection_proxy_or_rate"] is True

    manifest_path, validation = write_forced_turbulent_reconnection_readiness_report(
        run_dir,
        require_movies=True,
        min_history_samples=5,
        min_t_end=1.0,
        max_relative_energy_growth=10.0,
    )
    assert validation["passed"] is True
    assert manifest_path == run_dir / "readiness" / "manifest.json"
    assert (run_dir / "readiness" / "promotion_readiness.json").exists()
    assert (run_dir / "readiness" / "validation.json").exists()
    assert (run_dir / "readiness" / "figures" / "promotion_matrix.png").stat().st_size > 0
    artifact_manifest = json.loads((run_dir / "readiness" / "artifact_manifest.json").read_text())
    assert {record["path"] for record in artifact_manifest["files"]} >= {
        "manifest.json",
        "validation.json",
        "promotion_readiness.json",
        "figures/promotion_matrix.png",
    }

    runner = CliRunner()
    cli_ready = runner.invoke(
        app,
        [
            "benchmark",
            "forced-turbulent-reconnection-readiness-check",
            str(run_dir),
            "--require-movies",
            "--min-history-samples",
            "5",
            "--min-t-end",
            "1.0",
            "--max-relative-energy-growth",
            "10.0",
        ],
    )
    assert cli_ready.exit_code == 0, cli_ready.stdout

    cli_blocked = runner.invoke(
        app,
        [
            "benchmark",
            "forced-turbulent-reconnection-readiness-check",
            str(run_dir),
            "--min-history-samples",
            "999",
            "--min-t-end",
            "1.0",
            "--max-relative-energy-growth",
            "10.0",
        ],
    )
    assert cli_blocked.exit_code == 1
    assert "forced-turbulent-reconnection-readiness-check failed" in cli_blocked.output


def test_turbulence_rejects_invalid_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="shape"):
        run_decaying_mhd_turbulence_validation(shape=(7, 8))
    with pytest.raises(ValueError, match="resistivity"):
        run_decaying_mhd_turbulence_validation(resistivity=0.0)
    with pytest.raises(ValueError, match="viscosity"):
        run_decaying_mhd_turbulence_validation(viscosity=0.0)
    with pytest.raises(ValueError, match="dt"):
        run_decaying_mhd_turbulence_validation(dt=0.0)
    with pytest.raises(ValueError, match="at least four"):
        run_decaying_mhd_turbulence_validation(t_end=0.01)
    with pytest.raises(ValueError, match="integer multiple"):
        run_decaying_mhd_turbulence_validation(t_end=0.501)
    with pytest.raises(ValueError, match="save_every"):
        run_decaying_mhd_turbulence_validation(save_every=0)
    with pytest.raises(ValueError, match="at least two"):
        run_decaying_mhd_turbulence_validation(t_end=0.04, save_every=100)
    with pytest.raises(ValueError, match="min_relative_energy_drop"):
        run_decaying_mhd_turbulence_validation(min_relative_energy_drop=-1.0)
    with pytest.raises(ValueError, match="max_relative_energy_growth"):
        run_decaying_mhd_turbulence_validation(max_relative_energy_growth=-1.0)
    with pytest.raises(ValueError, match="min_current_linf_growth"):
        run_decaying_mhd_turbulence_validation(min_current_linf_growth=-1.0)
    with pytest.raises(ValueError, match="max_magnetic_divergence_linf"):
        run_decaying_mhd_turbulence_validation(max_magnetic_divergence_linf=0.0)
    with pytest.raises(ValueError, match="width"):
        run_forced_turbulent_reconnection_validation(width=0.0)
    with pytest.raises(ValueError, match="perturbation_amplitude"):
        run_forced_turbulent_reconnection_validation(perturbation_amplitude=0.0)
    with pytest.raises(ValueError, match="turbulent amplitudes"):
        run_forced_turbulent_reconnection_validation(turbulent_flux_amplitude=-1.0)
    with pytest.raises(ValueError, match="forcing_amplitude"):
        run_forced_turbulent_reconnection_validation(forcing_amplitude=-1.0)
    with pytest.raises(ValueError, match="min_history_samples"):
        assess_forced_turbulent_reconnection_readiness(tmp_path, min_history_samples=0)
    with pytest.raises(ValueError, match="min_t_end"):
        assess_forced_turbulent_reconnection_readiness(tmp_path, min_t_end=0.0)
    with pytest.raises(ValueError, match="min_reconnection_proxy_change"):
        assess_forced_turbulent_reconnection_readiness(
            tmp_path,
            min_reconnection_proxy_change=-1.0,
        )
    with pytest.raises(ValueError, match="min_abs_reconnection_rate_proxy"):
        assess_forced_turbulent_reconnection_readiness(
            tmp_path,
            min_abs_reconnection_rate_proxy=-1.0,
        )
    with pytest.raises(ValueError, match="max_relative_energy_growth"):
        assess_forced_turbulent_reconnection_readiness(
            tmp_path,
            max_relative_energy_growth=-1.0,
        )


def test_turbulence_helper_branches(tmp_path) -> None:
    assert np.array_equal(_sample_indices(4, 10), np.arange(4))
    assert np.array_equal(_sample_indices(10, 4), np.array([0, 3, 6, 9]))
    assert _high_k_fraction(np.zeros((8, 8))) == 0.0
    assert _critical_flux_separation(np.zeros((8, 8)), lengths=(1.0, 1.0)) == 0.0

    fields = np.linspace(0.0, 1.0, 3 * 8 * 8).reshape(3, 8, 8)
    movie_path = _write_scalar_movie(
        fields,
        tmp_path / "scalar.gif",
        cmap="viridis",
        symmetric=False,
        max_frames=2,
    )
    assert movie_path.stat().st_size > 0
