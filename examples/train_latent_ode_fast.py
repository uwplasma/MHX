"""Fit the deterministic MHX random-feature latent ODE on a FAST dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from mhx.neural_ode import write_neural_ode_training_bundle
from mhx.runtime import configure_jax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/examples/latent_ode_fast"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6)
    args = parser.parse_args()

    configure_jax(enable_x64=True)
    manifest_path, validation = write_neural_ode_training_bundle(
        args.outdir,
        shape=(8, 8),
        seeds=(0, 1, 2, 3),
        steps=args.steps,
        hidden_size=args.hidden_size,
    )
    print(f"wrote {manifest_path}")
    print(f"validation passed: {validation['passed']}")


if __name__ == "__main__":
    main()
