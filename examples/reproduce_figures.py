from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], *, env: dict | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    start = time.time()
    manifest = {"outputs": [], "commands": []}
    manifest["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        manifest["git_rev"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        manifest["git_rev"] = "unknown"

    # Core media
    run([sys.executable, "examples/make_fast_media.py"])
    manifest["commands"].append("examples/make_fast_media.py")

    # Inverse design / reachable region
    run([sys.executable, "examples/make_inverse_design_media.py"])
    manifest["commands"].append("examples/make_inverse_design_media.py")

    # Latent ODE fit
    run([sys.executable, "examples/latent_ode_fast.py"])
    manifest["commands"].append("examples/latent_ode_fast.py")

    # Reachable region figures (grid/inverse comparison)
    env_fast = os.environ.copy()
    env_fast.setdefault("MHX_FIGURES_FAST", "1")
    env_fast.setdefault("MHX_SCAN_N_ETA", "3")
    env_fast.setdefault("MHX_SCAN_N_NU", "3")
    run([sys.executable, "mhd_tearing_inverse_design_figures.py"], env=env_fast)
    manifest["commands"].append("mhd_tearing_inverse_design_figures.py")

    # Timing table
    run([sys.executable, "examples/benchmark_timings.py"])
    manifest["commands"].append("examples/benchmark_timings.py")

    # Collect outputs
    for path in [
        "docs/_static/energy.png",
        "docs/_static/az_midplane.gif",
        "docs/_static/fig_reachable_heatmap.png",
        "docs/_static/fig_reachable_region.png",
        "docs/_static/fig_cost_history.png",
        "docs/_static/latent_ode_fit.png",
        "docs/_static/latent_ode_ablation.rst",
        "docs/_static/timing_table.rst",
        "outputs/figures/fig_reachable_heatmaps_forcefree.png",
        "outputs/figures/fig_inverse_vs_grid_forcefree.png",
        "outputs/benchmarks/timing_table.json",
        "outputs/benchmarks/latent_ode_ablation.json",
    ]:
        if Path(path).exists():
            manifest["outputs"].append(path)

    manifest["elapsed_sec"] = time.time() - start
    out = Path("outputs") / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
