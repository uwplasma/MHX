"""Orchestrate explicit simulation and staged rendering for all curated media."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .common import SOURCE_ROOT
except ImportError:  # Direct script execution.
    from common import SOURCE_ROOT

HERE = Path(__file__).resolve().parent
CASES = {
    "decaying_turbulence": ("decaying_turbulence.py", "decaying_mhd_turbulence.npz"),
    "double_harris_reconnection": (
        "double_harris_reconnection.py",
        "plasmoids_trajectory.npz",
    ),
    "forced_turbulence_2d": ("forced_turbulence_2d.py", "turbulent_spectrum.npz"),
    "orszag_tang_2d": ("orszag_tang_2d.py", "orszag_tang_vortex.npz"),
    "orszag_tang_3d": ("orszag_tang_3d.py", "trajectory.npz"),
    "kelvin_helmholtz": ("kelvin_helmholtz.py", "kelvin_helmholtz.npz"),
    "turbulence_3d": ("turbulence_3d.py", "trajectory.npz"),
}


def _selected_cases(values: list[str] | None) -> list[str]:
    if not values:
        return list(CASES)
    unknown = sorted(set(values) - set(CASES))
    if unknown:
        raise ValueError(f"unknown media cases: {', '.join(unknown)}")
    return values


def status(*, preset: str, cases: list[str]) -> bool:
    """Print source availability without running simulations or renderers."""
    complete = True
    for case in cases:
        _, source_name = CASES[case]
        source = SOURCE_ROOT / preset / case / source_name
        available = source.is_file()
        complete &= available
        state = "ready" if available else "missing"
        print(f"{case:28s} {state:7s} {source}")
    return complete


def invoke(*, command: str, preset: str, cases: list[str]) -> None:
    """Invoke one standardized command for each selected case."""
    for case in cases:
        script_name, source_name = CASES[case]
        command_line = [sys.executable, str(HERE / script_name), command, "--preset", preset]
        if command == "render":
            source = SOURCE_ROOT / preset / case / source_name
            command_line.extend(("--source", str(source)))
        print(f"[{case}] {' '.join(command_line)}", flush=True)
        subprocess.run(command_line, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "simulate", "render", "run"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--preset", choices=("preview", "final"), default="preview")
        command_parser.add_argument("--case", action="append", dest="cases", choices=CASES)
        if command in {"simulate", "run"}:
            command_parser.add_argument("--allow-expensive", action="store_true")
    args = parser.parse_args()
    cases = _selected_cases(args.cases)
    if args.command == "status":
        raise SystemExit(0 if status(preset=args.preset, cases=cases) else 1)
    if args.command in {"simulate", "run"}:
        if args.preset == "final" and not args.allow_expensive:
            raise SystemExit("final simulations require --allow-expensive")
        invoke(command="simulate", preset=args.preset, cases=cases)
    if args.command in {"render", "run"}:
        invoke(command="render", preset=args.preset, cases=cases)


if __name__ == "__main__":
    main()
