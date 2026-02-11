from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    # Core media
    run([sys.executable, "examples/make_fast_media.py"])

    # Inverse design / reachable region
    run([sys.executable, "examples/make_inverse_design_media.py"])

    # Latent ODE fit
    run([sys.executable, "examples/latent_ode_fast.py"])

    # Reachable region figures (grid/inverse comparison)
    run([sys.executable, "mhd_tearing_inverse_design_figures.py"])


if __name__ == "__main__":
    main()
