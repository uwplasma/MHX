"""Run a restartable Rutherford production-campaign chunk in FAST mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from mhx.campaigns import (
    WalltimePolicy,
    write_rutherford_production_execution,
    write_rutherford_production_plan,
)
from mhx.runtime import configure_jax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/examples/rutherford_chunk"))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--movies", action="store_true")
    args = parser.parse_args()

    configure_jax(enable_x64=True)
    write_rutherford_production_plan(
        args.outdir,
        shape=(8, 8),
        dt=1.0e-2,
        target_saved_frames=120,
        min_production_resolution=8,
        walltime_policy=WalltimePolicy(
            max_walltime_hours=1.0,
            seconds_per_step_estimate=0.1,
            checkpoint_interval_minutes=1.0,
            preemption_margin_minutes=1.0,
        ),
    )
    manifest_path, validation = write_rutherford_production_execution(
        args.outdir,
        max_steps=args.steps,
        write_movies=args.movies,
    )
    print(f"wrote {manifest_path}")
    print(f"validation passed: {validation['passed']}")


if __name__ == "__main__":
    main()
