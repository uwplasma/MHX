from __future__ import annotations

import dataclasses
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics


def _make_config(size: str) -> TearingSimConfig:
    if size == "fast":
        return TearingSimConfig.fast("original")
    if size == "medium":
        return TearingSimConfig(
            Nx=32,
            Ny=32,
            Nz=1,
            t1=1.0,
            n_frames=12,
            dt0=5e-4,
            progress=False,
            jit=False,
            check_finite=True,
            equilibrium_mode="original",
        )
    if size == "prod":
        return TearingSimConfig(
            Nx=48,
            Ny=48,
            Nz=1,
            t1=1.5,
            n_frames=16,
            dt0=5e-4,
            progress=False,
            jit=False,
            check_finite=True,
            equilibrium_mode="original",
        )
    raise ValueError(f"Unknown size: {size}")


def _scale_for_ci(cfg: TearingSimConfig) -> TearingSimConfig:
    if os.getenv("MHX_BENCH_FAST"):
        return TearingSimConfig.fast(cfg.equilibrium_mode)
    return cfg


def run_case(size: str, jit: bool) -> dict:
    cfg = _scale_for_ci(_make_config(size))
    cfg = dataclasses.replace(cfg, jit=jit)

    tracemalloc.start()
    start = time.perf_counter()
    res = _run_tearing_simulation_and_diagnostics(
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        Nz=cfg.Nz,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        Lz=cfg.Lz,
        nu=cfg.nu,
        eta=cfg.eta,
        B0=cfg.B0,
        a=cfg.a,
        B_g=cfg.B_g,
        eps_B=cfg.eps_B,
        t0=cfg.t0,
        t1=cfg.t1,
        n_frames=cfg.n_frames,
        dt0=cfg.dt0,
        equilibrium_mode=cfg.equilibrium_mode,
        progress=False,
        jit=jit,
        check_finite=cfg.check_finite,
        diagnostics=["energies", "theory"],
    )
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "size": size,
        "jit": bool(jit),
        "Nx": cfg.Nx,
        "Ny": cfg.Ny,
        "n_frames": cfg.n_frames,
        "t1": cfg.t1,
        "elapsed_sec": float(elapsed),
        "peak_mem_mb": float(peak) / (1024.0**2),
        "gamma_fit": float(np.array(res["gamma_fit"])),
    }


def main() -> None:
    sizes = ["fast", "medium", "prod"]
    rows = []
    for size in sizes:
        for jit in (False, True):
            rows.append(run_case(size, jit))

    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        ".. list-table:: Benchmark matrix (FAST/medium/prod × jit)",
        "   :header-rows: 1",
        "",
        "   * - size",
        "     - jit",
        "     - Nx",
        "     - n_frames",
        "     - t1",
        "     - elapsed (s)",
        "     - peak mem (MB)",
    ]
    for row in rows:
        lines.extend(
            [
                f"   * - {row['size']}",
                f"     - {row['jit']}",
                f"     - {row['Nx']}",
                f"     - {row['n_frames']}",
                f"     - {row['t1']}",
                f"     - {row['elapsed_sec']:.3f}",
                f"     - {row['peak_mem_mb']:.1f}",
            ]
        )

    static_dir = Path("docs/_static")
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "benchmark_matrix.rst").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
