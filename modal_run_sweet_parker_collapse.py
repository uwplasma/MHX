"""Run the Sweet-Parker X-point collapse pre-test on Modal.

Usage (from the MHX_final repo root)::

    modal run modal_run_sweet_parker_collapse.py
    modal run modal_run_sweet_parker_collapse.py --ny 768 --ly 25.132741228718345
    modal run modal_run_sweet_parker_collapse.py --no-hold

Runs ``examples/sweet_parker/collapse_check.py`` on an H100 (default ``Lx = 8*pi``
with ``--hold-equilibrium``). Outputs land in the persistent ``mhx-outputs``
volume under
``sweet_parker/collapse_eta<eta>_<nx>x<ny>_lx<lx>_ly<ly>_seed<seed>[_hold]/``; the pull
commands are printed when the run finishes.
"""

from __future__ import annotations

import math
from pathlib import Path

import modal

app = modal.App("mhx-sweet-parker-collapse")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "jaxtyping",
        "matplotlib",
        "numpy",
        "rich",
        "solvax",
        "imageio",
        "typer",
    )
    .add_local_dir("src", remote_path="/root/mhx/src", copy=True)
    .add_local_dir(
        "examples/sweet_parker",
        remote_path="/root/mhx/examples/sweet_parker",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/root/mhx/src",
            "JAX_ENABLE_X64": "1",
            "JAX_PLATFORM_NAME": "cuda",
        }
    )
)

volume = modal.Volume.from_name("mhx-outputs", create_if_missing=True)
OUTPUT_FILES = (
    "summary.json",
    "histories.png",
    "current_snapshots.png",
    "domain.gif",
    "histories.npz",
    "snapshots.npz",
)


def _run_name(
    eta: float, nx: int, ny: int, lx: float, ly: float, seed: float, hold: bool
) -> str:
    suffix = "_hold" if hold else ""
    return f"collapse_eta{eta:g}_{nx}x{ny}_lx{lx:.4g}_ly{ly:.4g}_seed{seed:g}{suffix}"


@app.function(
    image=image,
    gpu="H100",
    volumes={"/outputs": volume},
    timeout=3600,
)
def run_collapse(
    nx: int = 3072,
    ny: int = 384,
    eta: float = 2.0e-3,
    t_end: float = 300.0,
    seed: float = 1.0e-3,
    lx: float = 8.0 * math.pi,
    ly: float = 4.0 * math.pi,
    hold: bool = True,
    extra_args: str = "",
) -> str:
    import os
    import subprocess
    import sys
    import time

    outdir = Path("/outputs/sweet_parker") / _run_name(eta, nx, ny, lx, ly, seed, hold)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",  # unbuffered, so progress streams to the Modal logs
        "/root/mhx/examples/sweet_parker/collapse_check.py",
        "--nx", str(nx),
        "--ny", str(ny),
        "--eta", str(eta),
        "--t-end", str(t_end),
        "--seed", str(seed),
        "--lx", repr(lx),
        "--ly", repr(ly),
        *(["--hold-equilibrium"] if hold else []),
        f"--outdir={outdir}",
        *extra_args.split(),
    ]
    print(f"[modal] running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=os.environ.copy())
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"collapse_check failed with exit code {result.returncode}")
    return f"Wall-clock: {time.perf_counter() - t0:.1f} s  |  outputs: {outdir}"


@app.local_entrypoint()
def main(
    nx: int = 3072,
    ny: int = 384,
    eta: float = 2.0e-3,
    t_end: float = 300.0,
    seed: float = 1.0e-3,
    lx: float = 8.0 * math.pi,
    ly: float = 4.0 * math.pi,
    hold: bool = True,
    extra_args: str = "",
):
    summary = run_collapse.remote(nx, ny, eta, t_end, seed, lx, ly, hold, extra_args)
    name = _run_name(eta, nx, ny, lx, ly, seed, hold)
    print(f"\nDone.  {summary}")
    print(
        "\nPull results (make the folder first, then one file at a time):\n"
        f"  mkdir -p outputs/sweet_parker/{name}\n"
        + "".join(
            f"  modal volume get mhx-outputs sweet_parker/{name}/{f} "
            f"outputs/sweet_parker/{name}/{f}\n"
            for f in OUTPUT_FILES
        )
    )
