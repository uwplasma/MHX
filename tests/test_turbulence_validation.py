from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    DECAYING_MHD_TURBULENCE_SCHEMA,
    FORCED_TURBULENT_RECONNECTION_SCHEMA,
    run_decaying_mhd_turbulence_validation,
    run_forced_turbulent_reconnection_validation,
    write_decaying_mhd_turbulence_validation,
    write_forced_turbulent_reconnection_validation,
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


def test_turbulence_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        run_decaying_mhd_turbulence_validation(shape=(7, 8))
    with pytest.raises(ValueError, match="integer multiple"):
        run_decaying_mhd_turbulence_validation(t_end=0.501)
    with pytest.raises(ValueError, match="width"):
        run_forced_turbulent_reconnection_validation(width=0.0)
    with pytest.raises(ValueError, match="forcing_amplitude"):
        run_forced_turbulent_reconnection_validation(forcing_amplitude=-1.0)
