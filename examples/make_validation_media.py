"""Generate documentation figures for numerical-validation pages."""

from __future__ import annotations

import shutil
from pathlib import Path

from mhx.benchmarks import write_resistive_decay_validation


def main() -> None:
    """Regenerate deterministic validation figures used by the docs."""
    run_dir = Path("outputs/docs_validation/resistive_decay")
    docs_dir = Path("docs/_static/validation/exact_decay")
    write_resistive_decay_validation(run_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decay_amplitude.png", "decay_energy.png", "decay_relative_error.png"):
        shutil.copy2(run_dir / "figures" / name, docs_dir / name)


if __name__ == "__main__":
    main()
