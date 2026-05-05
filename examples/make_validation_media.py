"""Generate documentation figures for numerical-validation pages."""

from __future__ import annotations

import shutil
from pathlib import Path

from mhx.benchmarks import (
    write_fkr_window_validation,
    write_linearized_rhs_validation,
    write_reconnection_scaling_validation,
    write_resistive_decay_validation,
    write_timing_benchmark,
)


def main() -> None:
    """Regenerate deterministic validation figures used by the docs."""
    run_dir = Path("outputs/docs_validation/resistive_decay")
    scaling_run_dir = Path("outputs/docs_validation/reconnection_scaling")
    fkr_window_run_dir = Path("outputs/docs_validation/fkr_window")
    linearized_run_dir = Path("outputs/docs_validation/linearized_rhs")
    timing_run_dir = Path("outputs/docs_validation/timing")
    decay_docs_dir = Path("docs/_static/validation/exact_decay")
    scaling_docs_dir = Path("docs/_static/validation/reconnection_scaling")
    fkr_window_docs_dir = Path("docs/_static/validation/fkr_window")
    linearized_docs_dir = Path("docs/_static/validation/linearized_rhs")
    timing_docs_dir = Path("docs/_static/performance")
    write_resistive_decay_validation(run_dir)
    write_reconnection_scaling_validation(scaling_run_dir)
    write_fkr_window_validation(fkr_window_run_dir)
    write_linearized_rhs_validation(linearized_run_dir)
    write_timing_benchmark(timing_run_dir, repeats=1, warmups=0)
    decay_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decay_amplitude.png", "decay_energy.png", "decay_relative_error.png"):
        shutil.copy2(run_dir / "figures" / name, decay_docs_dir / name)
    scaling_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fkr_scaling.png", "plasmoid_scaling.png", "ideal_tearing_scaling.png"):
        shutil.copy2(scaling_run_dir / "figures" / name, scaling_docs_dir / name)
    fkr_window_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        fkr_window_run_dir / "figures" / "fkr_constant_psi_window.png",
        fkr_window_docs_dir / "fkr_constant_psi_window.png",
    )
    linearized_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        linearized_run_dir / "figures" / "linearized_rhs_errors.png",
        linearized_docs_dir / "linearized_rhs_errors.png",
    )
    timing_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        timing_run_dir / "figures" / "timing_summary.png",
        timing_docs_dir / "timing_summary.png",
    )


if __name__ == "__main__":
    main()
