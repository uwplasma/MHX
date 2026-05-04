from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def config_yaml(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def logs_txt(self) -> Path:
        return self.run_dir / "logs.txt"

    @property
    def history_npz(self) -> Path:
        return self.run_dir / "history.npz"

    @property
    def solution_initial_npz(self) -> Path:
        return self.run_dir / "solution_initial.npz"

    @property
    def solution_mid_npz(self) -> Path:
        return self.run_dir / "solution_mid.npz"

    @property
    def solution_final_npz(self) -> Path:
        return self.run_dir / "solution_final.npz"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"


def _timestamp() -> str:
    # Local time, filesystem-friendly.
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(
    *,
    root: Path | str = "outputs/runs",
    tag: str = "run",
    timestamp: Optional[str] = None,
    exist_ok: bool = False,
) -> RunPaths:
    root_path = Path(root)
    ts = _timestamp() if timestamp is None else timestamp
    run_dir = root_path / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir)

