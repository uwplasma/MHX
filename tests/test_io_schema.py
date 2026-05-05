from __future__ import annotations

import json

import pytest

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.config import MeshConfig, RunConfig, TimeConfig
from mhx.io import read_reduced_mhd_trajectory_npz, write_reduced_mhd_trajectory_npz


def test_reduced_mhd_npz_roundtrip(tmp_path) -> None:
    cfg = RunConfig(mesh=MeshConfig(shape=(8, 8)), time=TimeConfig(t1=0.02, dt=0.01))
    trajectory, diagnostics = run_linear_tearing_smoke(cfg)
    path = write_reduced_mhd_trajectory_npz(
        tmp_path / "trajectory.npz",
        trajectory=trajectory,
        config=cfg.to_dict(),
        diagnostics=diagnostics,
    )
    loaded, loaded_diagnostics = read_reduced_mhd_trajectory_npz(path)
    assert loaded.times.shape == trajectory.times.shape
    assert loaded.states.psi.shape == trajectory.states.psi.shape
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        assert str(data["api_version"]) == "v1"
        assert str(data["schema"]) == "mhx.reduced_mhd.trajectory.v1"
    assert loaded_diagnostics["final_total_energy"] == pytest.approx(
        diagnostics["final_total_energy"]
    )


def test_reduced_mhd_npz_rejects_unknown_schema(tmp_path) -> None:
    import numpy as np

    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        schema="bad.schema",
        time=[],
        psi=[],
        omega=[],
        diagnostics_json=json.dumps({}),
    )
    with pytest.raises(ValueError, match="unsupported trajectory schema"):
        read_reduced_mhd_trajectory_npz(path)


def test_reduced_mhd_npz_rejects_unsupported_api_version(tmp_path) -> None:
    import numpy as np

    path = tmp_path / "bad-api.npz"
    np.savez_compressed(
        path,
        schema="mhx.reduced_mhd.trajectory.v1",
        api_version="v999",
        time=[],
        psi=[],
        omega=[],
        diagnostics_json=json.dumps({}),
    )
    with pytest.raises(ValueError, match="unsupported API version"):
        read_reduced_mhd_trajectory_npz(path)
