"""The Kelvin--Helmholtz gallery example runs end to end at a tiny size."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _example_runner import run_example

SCRIPT = Path(__file__).parents[1] / "examples" / "gallery" / "14_kelvin_helmholtz.py"


def test_kelvin_helmholtz_example_writes_history_and_figure(tmp_path, monkeypatch) -> None:
    outdir = tmp_path / "out"
    namespace = run_example(
        SCRIPT,
        tmp_path,
        {
            "OUTDIR": outdir,
            "NX": 16,
            "NY": 32,
            "RE": 1.0e3,
            "DT": 1.0e-2,
            "T_END": 0.4,
            "SAVE_EVERY": 10,
            "SNAPSHOT_TIMES": (0.2, 0.4),
        },
        monkeypatch=monkeypatch,
    )

    assert (outdir / "kh_incompressible.npz").stat().st_size > 0
    assert (outdir / "dye_snapshots.png").stat().st_size > 0
    with np.load(outdir / "kh_incompressible.npz") as data:
        assert data["dye"].shape == (4, 16, 32)
        # A spectral dye overshoots 0 and 1 slightly (Gibbs), so allow a small margin.
        assert np.all(data["dye"] >= -0.1)
        assert np.all(data["dye"] <= 1.1)
        assert np.isfinite(data["entropy"]).all()
        assert np.allclose(data["times"], [0.1, 0.2, 0.3, 0.4])
    assert namespace["viscosity"] == 2.0 / 1.0e3
