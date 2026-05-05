"""Generate documentation figures for numerical-validation pages."""

from __future__ import annotations

import shutil
from pathlib import Path

from mhx.benchmarks import (
    write_reconnection_scaling_validation,
    write_resistive_decay_validation,
    write_timing_benchmark,
)


def main() -> None:
    """Regenerate deterministic validation figures used by the docs."""
    run_dir = Path("outputs/docs_validation/resistive_decay")
    scaling_run_dir = Path("outputs/docs_validation/reconnection_scaling")
    timing_run_dir = Path("outputs/docs_validation/timing")
    decay_docs_dir = Path("docs/_static/validation/exact_decay")
    scaling_docs_dir = Path("docs/_static/validation/reconnection_scaling")
    timing_docs_dir = Path("docs/_static/performance")
    write_resistive_decay_validation(run_dir)
    write_reconnection_scaling_validation(scaling_run_dir)
    write_timing_benchmark(timing_run_dir, repeats=1, warmups=0)
    decay_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decay_amplitude.png", "decay_energy.png", "decay_relative_error.png"):
        shutil.copy2(run_dir / "figures" / name, decay_docs_dir / name)
    scaling_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fkr_scaling.png", "plasmoid_scaling.png", "ideal_tearing_scaling.png"):
        shutil.copy2(scaling_run_dir / "figures" / name, scaling_docs_dir / name)
    timing_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        timing_run_dir / "figures" / "timing_summary.png",
        timing_docs_dir / "timing_summary.png",
    )


if __name__ == "__main__":
    main()
