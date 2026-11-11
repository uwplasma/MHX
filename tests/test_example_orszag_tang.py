"""Run the Orszag--Tang example scripts with tiny inputs and check their outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _example_runner import run_example

EXAMPLES = Path(__file__).parents[1] / "examples"
OT = EXAMPLES / "orszag_tang"


def _non_empty(*paths: Path) -> None:
    for path in paths:
        assert path.is_file(), path
        assert path.stat().st_size > 0, path


def test_vencels_kinetic_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    ns = run_example(
        OT / "vencels_kinetic_comparison.py",
        tmp_path,
        {"OUTDIR": out, "NX": 16, "NY": 16, "T_END": 0.1, "SAVE_EVERY": 5},
        monkeypatch=monkeypatch,
    )

    _non_empty(out / "kinetic_comparison.png", out / "manifest.json")
    assert ns["result"]["time"].shape == (5,)
    assert np.isfinite(ns["result"]["total_energy"]).all()
    assert ns["result"]["final_divergence"] < 1.0e-8


def test_autodiff_optimization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    ns = run_example(
        OT / "autodiff_optimization.py",
        tmp_path,
        {
            "OUTDIR": out,
            "NX": 16,
            "NY": 16,
            "DT": 0.005,
            "T_END": 0.04,
            "SAVE_EVERY": 2,
            "OPT_STEPS": 2,
        },
        monkeypatch=monkeypatch,
    )

    _non_empty(out / "kinetic_comparison.png")
    assert len(ns["history"]) == 2
    assert all(np.isfinite(loss) for _, loss, _, _ in ns["history"])
    assert ns["final_eta"] > 0.0 and ns["final_nu"] > 0.0


def test_autodiff_optimization_rejects_partial_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        run_example(
            OT / "autodiff_optimization.py",
            tmp_path,
            {"OUTDIR": tmp_path / "out", "NX": 16, "NY": 16, "T_END": 0.05, "SAVE_EVERY": 20},
            monkeypatch=monkeypatch,
        )


def test_optimize_initial_conditions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("optax")
    out = tmp_path / "out"
    ns = run_example(
        OT / "optimize_initial_conditions.py",
        tmp_path,
        {
            "OUTDIR": out,
            "RESOLUTION": 16,
            "DT": 0.05,
            "T_END": 0.2,
            "OPT_STEPS": 2,
            "SCAN_ETA": True,
            "SCAN_P2": True,
            "SCAN_POINTS": 3,
        },
        monkeypatch=monkeypatch,
    )

    _non_empty(
        out / "initial_condition_opt_loss.png",
        out / "optimization_process.gif",
        out / "time_evolution_comparison.gif",
        out / "scan_eta_vs_loss.png",
        out / "scan_p2_vs_loss.png",
        out / "scan_results.npz",
    )
    assert len(ns["loss_history"]) == 2
    assert np.isfinite(ns["loss_history"]).all()
    assert ns["theta"].shape == (4,)


def test_benchmark_inviscid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    ns = run_example(
        OT / "benchmark_inviscid.py",
        tmp_path,
        {
            "OUTDIR": out,
            "NX": 16,
            "NY": 16,
            "DT": 0.005,
            "T_END": 0.04,
            "SAVE_EVERY": 2,
            "MOVIES": True,
        },
        monkeypatch=monkeypatch,
    )

    _non_empty(
        out / "diagnostics.json",
        out / "validation.json",
        out / "manifest.json",
        out / "inviscid_orszag_tang.npz",
        out / "figures" / "inviscid_orszag_tang_summary.png",
        out / "figures" / "inviscid_orszag_tang_flux.gif",
        out / "figures" / "inviscid_orszag_tang_current.gif",
        out / "figures" / "inviscid_orszag_tang_vorticity.gif",
    )
    assert ns["checks"]["finite_arrays"]
    assert ns["total"].shape == (5,)
    assert ns["relative_energy_error"] < 1.0e-3


def test_benchmark_inviscid_validates_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="divisible"):
        run_example(
            OT / "benchmark_inviscid.py",
            tmp_path,
            {"OUTDIR": tmp_path / "out", "NX": 16, "NY": 16, "T_END": 0.04, "DT": 0.005,
             "SAVE_EVERY": 3},
            monkeypatch=monkeypatch,
        )


def _write_trajectory(path: Path, n: int = 12, frames: int = 3) -> None:
    rng = np.random.default_rng(0)
    shape = (frames, 3, n, n, n)
    np.savez(
        path,
        times=np.linspace(1.0, 2.0, frames),
        velocity=rng.normal(size=shape),
        magnetic=rng.normal(size=shape),
        config=np.array(str({"equilibrium": "orszag_tang_3d"})),
    )


def test_track_orszag_tang_3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    npz = tmp_path / "trajectory.npz"
    _write_trajectory(npz)
    ns = run_example(
        OT / "track_orszag_tang_3d.py",
        tmp_path,
        {"NPZ": npz, "KMIN": 2, "KMAX": 4, "GIF_FRAMES": 2, "T_SNAP": 2.0},
        monkeypatch=monkeypatch,
    )

    track = tmp_path / "track"
    _non_empty(
        track / "energy.png",
        track / "spectrum.png",
        track / "history.csv",
        track / "spectrum.csv",
        track / "cube_final.png",
        track / "cube.gif",
    )
    assert ns["total"] == 4  # three saved states plus the rebuilt t = 0
    assert ns["kinetic"].shape == (4,)
    assert np.isfinite(ns["slope"])
    assert ns["w0"] == pytest.approx(3.92)  # E_K(0) = 2, E_M(0) = 1.92 at beta = 0.8


def test_track_orszag_tang_3d_magnetic_field_faces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "trajectory.npz"
    _write_trajectory(npz)
    run_example(
        OT / "track_orszag_tang_3d.py",
        tmp_path,
        {"NPZ": npz, "OUTDIR": tmp_path / "out", "FIELD": "B", "KMIN": 2, "GIF_FRAMES": 1},
        monkeypatch=monkeypatch,
    )

    _non_empty(tmp_path / "out" / "cube_final.png")


def test_run_orszag_tang(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    ns = run_example(
        EXAMPLES / "run_orszag_tang.py",
        tmp_path,
        {"OUTDIR": out, "NX": 16, "NY": 16, "T_END": 0.1, "SAVE_EVERY": 5, "MOVIES": False},
        monkeypatch=monkeypatch,
    )

    _non_empty(out / "manifest.json")
    assert isinstance(ns["validation"]["passed"], bool)


def test_run_linear_tearing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ns = run_example(
        EXAMPLES / "run_linear_tearing.py",
        tmp_path,
        {"CONFIG": str(EXAMPLES / "linear_tearing.toml")},
        monkeypatch=monkeypatch,
    )

    assert isinstance(ns["diagnostics"], dict)
    assert ns["diagnostics"]
