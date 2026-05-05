"""Generate documentation figures for numerical-validation pages."""

from __future__ import annotations

import shutil
from pathlib import Path

from mhx.benchmarks import write_reconnection_scaling_validation, write_resistive_decay_validation


def main() -> None:
    """Regenerate deterministic validation figures used by the docs."""
    run_dir = Path("outputs/docs_validation/resistive_decay")
    scaling_run_dir = Path("outputs/docs_validation/reconnection_scaling")
    decay_docs_dir = Path("docs/_static/validation/exact_decay")
    scaling_docs_dir = Path("docs/_static/validation/reconnection_scaling")
    write_resistive_decay_validation(run_dir)
    write_reconnection_scaling_validation(scaling_run_dir)
    decay_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decay_amplitude.png", "decay_energy.png", "decay_relative_error.png"):
        shutil.copy2(run_dir / "figures" / name, decay_docs_dir / name)
    scaling_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fkr_scaling.png", "plasmoid_scaling.png", "ideal_tearing_scaling.png"):
        shutil.copy2(scaling_run_dir / "figures" / name, scaling_docs_dir / name)


if __name__ == "__main__":
    main()
