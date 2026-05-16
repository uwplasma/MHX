"""Run the reduced-MHD Orszag--Tang nonlinear example with movies."""

from __future__ import annotations

import argparse
from pathlib import Path

from mhx.benchmarks import write_orszag_tang_vortex_validation
from mhx.runtime import configure_jax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/examples/orszag_tang"))
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--eta", type=float, default=1.0e-2)
    parser.add_argument("--nu", type=float, default=1.0e-2)
    parser.add_argument("--dt", type=float, default=5.0e-3)
    parser.add_argument("--t-end", type=float, default=6.0)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--no-movies", action="store_true")
    args = parser.parse_args()

    configure_jax(enable_x64=True)
    manifest_path, validation = write_orszag_tang_vortex_validation(
        args.outdir,
        shape=(args.nx, args.ny),
        resistivity=args.eta,
        viscosity=args.nu,
        dt=args.dt,
        t_end=args.t_end,
        save_every=args.save_every,
        movies=not args.no_movies,
    )
    print(f"wrote {manifest_path}")
    print(f"passed={validation['passed']}")


if __name__ == "__main__":
    main()
