"""Run gallery scripts with tiny overrides of their input constants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _example_runner import run_example

GALLERY = Path(__file__).parents[1] / "examples" / "gallery"


def test_gallery_cpu_parallel(tmp_path, monkeypatch) -> None:
    run_example(
        GALLERY / "04_cpu_parallel.py",
        tmp_path,
        {
            "OUTDIR": tmp_path / "out",
            "SHAPE": (32, 32),
            "T_END": 0.05,
            "DEVICE_COUNT": 2,
        },
        monkeypatch=monkeypatch,
    )

    assert (tmp_path / "out" / "summary.png").exists()


def test_gallery_strong_scaling(tmp_path, monkeypatch) -> None:
    import jax

    if len(jax.devices()) < 2:
        pytest.skip("needs at least two logical devices (set by tests/conftest.py)")
    run_example(
        GALLERY / "06_strong_scaling.py",
        tmp_path,
        {"SIDE": 32, "STEPS": 2, "SAMPLES": 1, "CASES": 2, "OUTDIR": tmp_path / "out"},
        monkeypatch=monkeypatch,
    )

    measurements = json.loads((tmp_path / "out" / "measurements.json").read_text())
    assert measurements["device_counts"][0] == 1
    assert (tmp_path / "out" / "strong_scaling.png").exists()


def test_gallery_gradient(tmp_path, monkeypatch, capsys) -> None:
    import jax

    x64 = jax.config.jax_enable_x64
    try:
        run_example(
            GALLERY / "08_gradient.py",
            tmp_path,
            {"N": 16, "STEPS": 4},
            monkeypatch=monkeypatch,
        )
    finally:
        jax.config.update("jax_enable_x64", x64)

    assert "relative error" in capsys.readouterr().out


def test_gallery_compressible_orszag_tang(tmp_path, monkeypatch) -> None:
    run_example(
        GALLERY / "10_compressible_orszag_tang.py",
        tmp_path,
        {
            "OUTDIR": tmp_path / "out",
            "SHAPE": (16, 16, 4),
            "T_END": 0.02,
            "SAVE_EVERY": 10,
        },
        monkeypatch=monkeypatch,
    )

    assert (tmp_path / "out" / "summary.png").exists()


def test_gallery_turbulence_3d(tmp_path, monkeypatch, capsys) -> None:
    import jax

    # The script is written for float32 (complex64 states); the suite enables x64 globally.
    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        run_example(
            GALLERY / "11_turbulence_3d.py",
            tmp_path,
            {"N": 8, "T_END": 0.1, "SAVE_EVERY": 10, "OUTDIR": tmp_path / "out"},
            monkeypatch=monkeypatch,
        )
    finally:
        jax.config.update("jax_enable_x64", x64)

    assert (tmp_path / "out" / "trajectory.npz").exists()
    assert (tmp_path / "out" / "spectrum.png").exists()
    assert "Warning: No frames found" in capsys.readouterr().out
