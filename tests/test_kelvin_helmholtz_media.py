from __future__ import annotations

from pathlib import Path

import numpy as np
from examples.media.kelvin_helmholtz import simulate_config


def test_kelvin_helmholtz_media_source_smoke(tmp_path: Path) -> None:
    source = simulate_config(
        config={
            "shape": (8, 16),
            "dt": 2.0e-3,
            "t_end": 4.0e-3,
            "save_every": 2,
            "viscosity": 1.0e-3,
        },
        outdir=tmp_path,
        preset="test",
    )

    with np.load(source, allow_pickle=False) as data:
        assert data["time"].shape == (2,)
        assert data["dye"].shape == (2, 8, 16)
        assert data["omega"].shape == data["dye"].shape
        assert np.isfinite(data["dye"]).all()
        assert np.isfinite(data["omega"]).all()
