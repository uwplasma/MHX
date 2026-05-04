from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_fast_simulate_then_figures(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"

    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_DISABLE_JIT", "1")

    _run(
        [
            sys.executable,
            "-m",
            "mhx.cli.main",
            "simulate",
            "--fast",
            "--equilibrium",
            "original",
            "--eta",
            "1e-3",
            "--nu",
            "1e-3",
            "--outdir",
            str(run_dir),
        ],
        cwd=repo,
        env=env,
    )

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "solution_final.npz").exists()

    data = np.load(run_dir / "solution_final.npz", allow_pickle=True)
    for key in ["ts", "E_kin", "E_mag", "gamma_fit", "complexity_final", "Az_final_mid"]:
        assert key in data.files
    assert np.isfinite(np.array(data["gamma_fit"])).all()
    assert np.isfinite(np.array(data["complexity_final"])).all()

    _run(
        [sys.executable, "-m", "mhx.cli.main", "figures", "--run", str(run_dir)],
        cwd=repo,
        env=env,
    )

    assert (run_dir / "figures" / "energy.png").exists()
    assert (run_dir / "figures" / "az_midplane.png").exists()

