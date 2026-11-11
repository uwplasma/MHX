"""Run the turbulence and Sweet-Parker example scripts with tiny settings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _example_runner import run_example

pytestmark = pytest.mark.slow


EXAMPLES = Path(__file__).parents[1] / "examples"
TURBULENCE = EXAMPLES / "turbulence"
SWEET_PARKER = EXAMPLES / "sweet_parker"


def _assert_files(folder: Path, names: list[str]) -> None:
    for name in names:
        path = folder / name
        assert path.exists(), f"missing {path}"
        assert path.stat().st_size > 0, f"empty {path}"


@pytest.fixture(scope="module")
def turbulence_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny forced-turbulence run whose npz the other turbulence scripts read."""
    workdir = tmp_path_factory.mktemp("turbulence")
    outdir = workdir / "out"
    monkeypatch = pytest.MonkeyPatch()
    try:
        namespace = run_example(
            TURBULENCE / "benchmark_turbulent_spectrum.py",
            workdir,
            {"OUTDIR": outdir, "NX": 32, "NY": 32, "DT": 0.01, "T_END": 0.2, "SAVE_EVERY": 10},
            monkeypatch=monkeypatch,
        )
    finally:
        monkeypatch.undo()
    assert "passed" in namespace
    return outdir


def test_benchmark_turbulent_spectrum_writes_outputs(turbulence_run: Path) -> None:
    _assert_files(
        turbulence_run,
        ["turbulent_spectrum.npz", "diagnostics.json", "validation.json", "manifest.json"],
    )
    with np.load(turbulence_run / "turbulent_spectrum.npz") as data:
        assert data["psi"].shape[1:] == (32, 32)
        assert len(data["time"]) >= 2


def test_analyze_spectrum(turbulence_run: Path, tmp_path: Path, monkeypatch) -> None:
    namespace = run_example(
        TURBULENCE / "analyze_spectrum.py",
        tmp_path,
        {"NPZ": turbulence_run / "turbulent_spectrum.npz", "K_MIN": 2.0, "K_MAX": 8.0},
        monkeypatch=monkeypatch,
    )
    _assert_files(turbulence_run, ["custom_spectrum_analysis.png"])
    assert np.isfinite(namespace["slope"])


def test_animate_turbulence(turbulence_run: Path, tmp_path: Path, monkeypatch) -> None:
    namespace = run_example(
        TURBULENCE / "animate_turbulence.py",
        tmp_path,
        {"NPZ": turbulence_run / "turbulent_spectrum.npz", "PROGRESS_EVERY": 1},
        monkeypatch=monkeypatch,
    )
    _assert_files(turbulence_run, ["turbulence_jz_evolution.gif"])
    assert len(namespace["frames"]) == namespace["num_frames"]


def test_render_turbulence_2d(turbulence_run: Path, tmp_path: Path, monkeypatch) -> None:
    outdir = tmp_path / "dynamic"
    namespace = run_example(
        TURBULENCE / "render_turbulence_2d.py",
        tmp_path,
        {
            "NPZ": turbulence_run / "turbulent_spectrum.npz",
            "OUTDIR": str(outdir),
            "KMIN": 2,
            "KMAX": 8,
            "KPLOT": 12,
            "FRAMES": 3,
        },
        monkeypatch=monkeypatch,
    )
    _assert_files(
        outdir, ["turbulence_2d_dynamic.gif", "turbulence_2d_final.png", "slope_history.csv"]
    )
    assert len(namespace["indices"]) == 3


@pytest.fixture(scope="module")
def collapse_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("collapse")
    outdir = workdir / "run"
    monkeypatch = pytest.MonkeyPatch()
    try:
        namespace = run_example(
            SWEET_PARKER / "collapse_check.py",
            workdir,
            {
                "OUTDIR": outdir,
                "NX": 64,
                "NY": 32,
                "T_END": 2.0,
                "SAVE_INTERVAL": 0.5,
                "SNAPSHOTS": 3,
                "GIF_INTERVAL": 1.0,
                "HOLD_EQUILIBRIUM": True,
            },
            monkeypatch=monkeypatch,
        )
    finally:
        monkeypatch.undo()
    assert "checks" in namespace["summary"]
    return outdir


def test_collapse_check_writes_outputs(collapse_run: Path) -> None:
    _assert_files(
        collapse_run,
        [
            "summary.json",
            "histories.npz",
            "histories.png",
            "snapshots.npz",
            "current_snapshots.png",
            "domain.gif",
        ],
    )
    summary = json.loads((collapse_run / "summary.json").read_text())
    assert summary["grid"] == [64, 32]
    assert summary["hold_equilibrium"] is True
    assert summary["t_final"] == pytest.approx(2.0)


def test_collapse_check_replot(collapse_run: Path, tmp_path: Path, monkeypatch) -> None:
    (collapse_run / "histories.png").unlink()
    run_example(
        SWEET_PARKER / "collapse_check.py", tmp_path, {"REPLOT": collapse_run},
        monkeypatch=monkeypatch,
    )
    _assert_files(collapse_run, ["histories.png", "summary.json"])


def test_plot_scaling(tmp_path: Path, monkeypatch) -> None:
    run_dirs = []
    for index, (eta, lundquist) in enumerate([(2.0e-3, 500.0), (1.0e-3, 1000.0)]):
        run_dir = tmp_path / f"run{index}"
        run_dir.mkdir()
        metrics = {
            "lundquist_window": lundquist,
            "normalized_rate_window": 0.5 / np.sqrt(lundquist),
            "aspect_ratio_window": 2.0 * np.sqrt(lundquist),
            "outflow_over_alfven": 0.8,
        }
        (run_dir / "summary.json").write_text(
            json.dumps({"resistivity": eta, "viscosity": eta, "metrics": metrics})
        )
        run_dirs.append(str(run_dir))
    output = tmp_path / "scaling.png"
    namespace = run_example(
        SWEET_PARKER / "plot_scaling.py",
        tmp_path,
        {"RUN_DIRS": run_dirs, "OUTPUT": output},
        monkeypatch=monkeypatch,
    )
    _assert_files(tmp_path, ["scaling.png"])
    assert len(namespace["points"]) == 2
