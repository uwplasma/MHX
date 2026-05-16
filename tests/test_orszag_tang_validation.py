from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    ORSZAG_TANG_VORTEX_SCHEMA,
    orszag_tang_initial_state,
    run_orszag_tang_vortex_validation,
    write_orszag_tang_vortex_validation,
)
from mhx.cli.main import app
from mhx.config import MeshConfig
from mhx.grids import CartesianGrid


def test_orszag_tang_initial_condition_matches_classic_modes() -> None:
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=(16, 16), lower=(0.0, 0.0), upper=(2.0 * np.pi, 2.0 * np.pi))
    )
    x, y = grid.mesh()
    state = orszag_tang_initial_state(grid)

    np.testing.assert_allclose(
        np.asarray(state.psi),
        np.asarray(np.cos(y) + 0.5 * np.cos(2.0 * x)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(state.omega),
        np.asarray(-np.cos(x) - np.cos(y)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_orszag_tang_vortex_validation_gate() -> None:
    result = run_orszag_tang_vortex_validation(shape=(16, 16), t_end=1.0)

    assert result.diagnostics["schema"] == ORSZAG_TANG_VORTEX_SCHEMA
    assert result.validation["passed"] is True
    assert all(result.validation["checks"].values())
    assert result.psi.shape == result.omega.shape == result.current_density.shape
    assert result.psi.shape[0] == result.time.size
    assert result.current_high_k_growth > 0.0
    assert result.vorticity_high_k_growth > 0.0
    assert result.total_energy[-1] < result.total_energy[0]
    assert result.final_magnetic_divergence_linf <= 1.0e-10


def test_orszag_tang_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        run_orszag_tang_vortex_validation(shape=(7, 8))
    with pytest.raises(ValueError, match="resistivity"):
        run_orszag_tang_vortex_validation(resistivity=0.0)
    with pytest.raises(ValueError, match="viscosity"):
        run_orszag_tang_vortex_validation(viscosity=0.0)
    with pytest.raises(ValueError, match="dt"):
        run_orszag_tang_vortex_validation(dt=0.0)
    with pytest.raises(ValueError, match="t_end"):
        run_orszag_tang_vortex_validation(t_end=0.0)
    with pytest.raises(ValueError, match="integer multiple"):
        run_orszag_tang_vortex_validation(t_end=1.001)
    with pytest.raises(ValueError, match="divisible"):
        run_orszag_tang_vortex_validation(t_end=1.0, save_every=17)
    with pytest.raises(ValueError, match="save_every"):
        run_orszag_tang_vortex_validation(save_every=0)
    with pytest.raises(ValueError, match="at least two"):
        run_orszag_tang_vortex_validation(t_end=0.02, save_every=100)
    with pytest.raises(ValueError, match="min_relative_energy_drop"):
        run_orszag_tang_vortex_validation(min_relative_energy_drop=-1.0)
    with pytest.raises(ValueError, match="max_relative_energy_growth"):
        run_orszag_tang_vortex_validation(max_relative_energy_growth=-1.0)
    with pytest.raises(ValueError, match="min_current_high_k_growth"):
        run_orszag_tang_vortex_validation(min_current_high_k_growth=-1.0)
    with pytest.raises(ValueError, match="min_vorticity_high_k_growth"):
        run_orszag_tang_vortex_validation(min_vorticity_high_k_growth=-1.0)
    with pytest.raises(ValueError, match="max_magnetic_divergence_linf"):
        run_orszag_tang_vortex_validation(max_magnetic_divergence_linf=0.0)


def test_write_orszag_tang_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_orszag_tang_vortex_validation(
        tmp_path,
        shape=(16, 16),
        t_end=1.0,
        save_every=20,
        movies=True,
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert json.loads((tmp_path / "diagnostics.json").read_text())["schema"] == (
        ORSZAG_TANG_VORTEX_SCHEMA
    )
    history = np.load(tmp_path / "orszag_tang_vortex.npz")
    assert str(history["schema"]) == ORSZAG_TANG_VORTEX_SCHEMA
    assert history["psi"].shape == history["omega"].shape
    assert (tmp_path / "figures" / "orszag_tang_summary.png").stat().st_size > 0
    assert (tmp_path / "figures" / "orszag_tang_flux.gif").stat().st_size > 0
    assert (tmp_path / "figures" / "orszag_tang_current.gif").stat().st_size > 0
    assert (tmp_path / "figures" / "orszag_tang_vorticity.gif").stat().st_size > 0

    outdir = tmp_path / "cli"
    cli_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "orszag-tang",
            "--outdir",
            str(outdir),
            "--nx",
            "16",
            "--ny",
            "16",
            "--t-end",
            "1.0",
            "--movies",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "manifest.json").exists()
