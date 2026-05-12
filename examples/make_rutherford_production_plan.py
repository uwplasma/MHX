"""Write a non-expensive Rutherford production planning bundle.

This example is safe for laptops and CI because it does not run the nonlinear
PDE. It writes the reviewer-facing walltime/checkpoint/resume contract used by
the restartable production executor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mhx.campaigns import (
    WalltimePolicy,
    write_rutherford_production_plan,
    write_rutherford_resume_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/examples/rutherford_production_plan"),
        help="Directory for production-plan artifacts.",
    )
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--target-saved-frames", type=int, default=120)
    parser.add_argument("--max-walltime-hours", type=float, default=1.0)
    parser.add_argument("--seconds-per-step-estimate", type=float, default=2.0)
    args = parser.parse_args()

    manifest_path, validation = write_rutherford_production_plan(
        args.outdir,
        shape=(args.nx, args.ny),
        dt=args.dt,
        target_saved_frames=args.target_saved_frames,
        walltime_policy=WalltimePolicy(
            max_walltime_hours=args.max_walltime_hours,
            seconds_per_step_estimate=args.seconds_per_step_estimate,
            checkpoint_interval_minutes=10.0,
            preemption_margin_minutes=5.0,
        ),
    )
    resume_path, _ = write_rutherford_resume_plan(args.outdir)
    print(f"manifest: {manifest_path}")
    print(f"resume_plan: {resume_path}")
    print(f"validation_passed: {validation['passed']}")


if __name__ == "__main__":
    main()
