from __future__ import annotations

import hashlib
import json
import os
import shutil
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

    # Latent ODE experiment with baselines
    run([sys.executable, "examples/latent_ode_experiment.py"])
    manifest["commands"].append("examples/latent_ode_experiment.py")

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
        "docs/_static/latent_ode_experiment.png",
        "docs/_static/latent_ode_experiment.rst",
        "outputs/figures/fig_reachable_heatmaps_forcefree.png",
        "outputs/figures/fig_inverse_vs_grid_forcefree.png",
        "outputs/benchmarks/timing_table.json",
        "outputs/benchmarks/latent_ode_ablation.json",
        "outputs/benchmarks/latent_ode_experiment.json",
    ]:
        if Path(path).exists():
            manifest["outputs"].append(path)

    # Bundle run configs from this session
    run_configs_dir = Path("outputs/run_configs")
    run_configs_dir.mkdir(parents=True, exist_ok=True)
    run_config_paths = []
    for run_dir in sorted(Path("outputs/runs").glob("*")):
        if run_dir.is_dir() and run_dir.stat().st_mtime >= start:
            cfg = run_dir / "config.yaml"
            if cfg.exists():
                dst = run_configs_dir / f"{run_dir.name}_config.yaml"
                shutil.copy2(cfg, dst)
                run_config_paths.append(str(dst))
    manifest["run_configs"] = run_config_paths

    # Hashes for outputs (sha256)
    hashes = {}
    for path in manifest["outputs"]:
        p = Path(path)
        if p.exists():
            h = hashlib.sha256()
            h.update(p.read_bytes())
            hashes[path] = h.hexdigest()
    manifest["hashes"] = hashes

    manifest["elapsed_sec"] = time.time() - start
    out = Path("outputs") / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
