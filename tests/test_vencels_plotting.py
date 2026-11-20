"""Tests for the Vencels fluid-versus-kinetic comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from mhx.benchmarks.vencels import VENCELS_CSV, VENCELS_DOMAIN_SIZE, run_orszag_tang
from mhx.plotting import plot_kinetic_comparison


@pytest.fixture(scope="module")
def tiny_run() -> dict:
    return run_orszag_tang(
        shape=(16, 16), resistivity=1.0e-2, viscosity=1.0e-2, dt=5.0e-3, t_end=0.1, save_every=5
    )


def test_plot_kinetic_comparison_writes_png(tiny_run: dict, tmp_path: Path) -> None:
    path = plot_kinetic_comparison(
        tiny_run, tmp_path / "nested" / "kinetic.png", VENCELS_CSV, VENCELS_DOMAIN_SIZE
    )

    assert path == tmp_path / "nested" / "kinetic.png"
    assert path.stat().st_size > 0


def test_plot_kinetic_comparison_without_reference_csv(tiny_run: dict, tmp_path: Path) -> None:
    path = plot_kinetic_comparison(
        tiny_run, tmp_path / "no_ref.png", tmp_path / "missing.csv", VENCELS_DOMAIN_SIZE
    )

    assert path.stat().st_size > 0
