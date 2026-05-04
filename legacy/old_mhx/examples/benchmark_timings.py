from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics


def _time_case(label: str, cfg: TearingSimConfig, *, jit: bool) -> dict:
    rss_mb = None
    try:
        import psutil  # type: ignore

        rss_mb = psutil.Process().memory_info().rss / 1e6
    except Exception:
        rss_mb = None

    tracemalloc.start()
    start = time.perf_counter()
    _run_tearing_simulation_and_diagnostics(
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
        check_finite=True,
    )
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "case": label,
        "Nx": cfg.Nx,
        "Ny": cfg.Ny,
        "t1": cfg.t1,
        "n_frames": cfg.n_frames,
        "jit": jit,
        "elapsed_sec": elapsed,
        "rss_mb": rss_mb,
        "tracemalloc_peak_mb": peak / 1e6,
    }


def _write_table(rows: list[dict], path: Path) -> None:
    lines = [
        ".. list-table:: Timing table",
        "   :header-rows: 1",
        "",
        "   * - Case",
        "     - Nx",
        "     - Ny",
        "     - t1",
        "     - n_frames",
        "     - jit",
        "     - elapsed_sec",
    ]
    for row in rows:
        lines.extend(
            [
                f"   * - {row['case']}",
                f"     - {row['Nx']}",
                f"     - {row['Ny']}",
                f"     - {row['t1']}",
                f"     - {row['n_frames']}",
                f"     - {row['jit']}",
                f"     - {row['elapsed_sec']:.3f}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true", help="Include a full production-sized run (may be slow).")
    parser.add_argument("--jit", action="store_true", help="Enable JIT for timing runs.")
    args = parser.parse_args()

    rows = []
    cfg_fast = TearingSimConfig.fast("original")
    rows.append(_time_case("fast", cfg_fast, jit=args.jit))

    cfg_small = TearingSimConfig.fast("original")
    cfg_small = cfg_small.__class__(**{**cfg_small.as_dict(), "Nx": 32, "Ny": 32, "t1": 1.0, "n_frames": 12})
    rows.append(_time_case("small", cfg_small, jit=args.jit))

    if args.production:
        cfg_prod = TearingSimConfig(equilibrium_mode="original")
        rows.append(_time_case("production", cfg_prod, jit=args.jit))

    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_table(rows, Path("docs/_static/timing_table.rst"))


if __name__ == "__main__":
    main()
