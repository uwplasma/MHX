"""Versioned trajectory NPZ schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mhx._version import __version__
from mhx.state import ReducedMHDState, ReducedMHDTrajectory

REDUCED_MHD_TRAJECTORY_SCHEMA = "mhx.reduced_mhd.trajectory.v1"


def write_reduced_mhd_trajectory_npz(
    path: str | Path,
    *,
    trajectory: ReducedMHDTrajectory,
    config: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Path:
    """Write a reduced-MHD trajectory using the stable v1 NPZ schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(REDUCED_MHD_TRAJECTORY_SCHEMA),
        mhx_version=np.asarray(__version__),
        time=np.asarray(trajectory.times),
        psi=np.asarray(trajectory.states.psi),
        omega=np.asarray(trajectory.states.omega),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        diagnostics_json=np.asarray(json.dumps(diagnostics, sort_keys=True)),
    )
    return output_path


def read_reduced_mhd_trajectory_npz(
    path: str | Path,
) -> tuple[ReducedMHDTrajectory, dict[str, Any]]:
    """Read a reduced-MHD v1 NPZ trajectory and diagnostics."""
    with np.load(Path(path), allow_pickle=False) as data:
        schema = str(data["schema"])
        if schema != REDUCED_MHD_TRAJECTORY_SCHEMA:
            raise ValueError(f"unsupported trajectory schema {schema!r}")
        trajectory = ReducedMHDTrajectory(
            times=data["time"],
            states=ReducedMHDState(psi=data["psi"], omega=data["omega"]),
        )
        diagnostics = json.loads(str(data["diagnostics_json"]))
    return trajectory, diagnostics
