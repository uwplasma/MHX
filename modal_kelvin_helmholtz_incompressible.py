"""Run the incompressible Kelvin--Helmholtz problem on Modal and extract dye snapshots at t = 2, 4, 6, 8.

The NERSC incompressible branch: 1024 x 2048 grid, Re = 1e6 (nu = 2u/Re = 2e-6, u = 1),
dt = 2.5e-4, saved every 0.1 time units. Zero for ``nx``, ``ny``, ``re``, ``dt`` or
``save_every`` means the default above.

Usage (from the MHX_final repo root; this bills Modal GPU time):

    modal run modal_kelvin_helmholtz_incompressible.py

Results land in the ``mhx-outputs`` volume under ``kh_incompressible_modal/``. Pull the
snapshots, not the full trajectory:

    mkdir -p outputs/kh_incompressible_modal
    modal volume get mhx-outputs kh_incompressible_modal/dye_snapshots_t2_4_6_8.png outputs/kh_incompressible_modal/dye_snapshots_t2_4_6_8.png
    modal volume get mhx-outputs kh_incompressible_modal/dye_snapshots_t2_4_6_8.npz outputs/kh_incompressible_modal/dye_snapshots_t2_4_6_8.npz
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("mhx-kh-incompressible")

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
    .add_local_file("examples/__init__.py", remote_path="/root/mhx/examples/__init__.py", copy=True)
    .add_local_file(
        "examples/kelvin_helmholtz.py",
        remote_path="/root/mhx/examples/kelvin_helmholtz.py",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/root/mhx/src:/root/mhx",
            "JAX_ENABLE_X64": "1",
            "JAX_PLATFORM_NAME": "cuda",
        }
    )
)

volume = modal.Volume.from_name("mhx-outputs", create_if_missing=True)
INCOMPRESSIBLE_OUTDIR = Path("/outputs/kh_incompressible_modal")
SNAPSHOT_TIMES = (2.0, 4.0, 6.0, 8.0)


def _plot_snapshots(dye, times, indices, path: Path) -> None:
    """Save dye snapshots at the given indices as one figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4), constrained_layout=True)
    for ax, index in zip(axes, indices):
        image = ax.imshow(
            dye[index].T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 2.0),
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"dye, t = {times[index]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image, ax=axes, shrink=0.8, label="dye concentration c")
    fig.savefig(path, dpi=150)
    plt.close(fig)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/outputs": volume},
    timeout=60 * 60 * 4,
)
def run_kh_incompressible(
    nx: int = 1024,
    ny: int = 2048,
    re: float = 1.0e6,
    flow_speed: float = 1.0,
    dt: float = 2.5e-4,
    t_end: float = 8.0,
    save_every: int = 400,
) -> str:
    """Run the incompressible Kelvin--Helmholtz problem and extract dye snapshots."""
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from mhx.runtime import configure_jax

    configure_jax(enable_x64=True)

    from examples.kelvin_helmholtz import KelvinHelmholtzConfig, run_kelvin_helmholtz_dye

    # nu = L * Delta_u / Re with Delta_u = 2u and L = 1.
    viscosity = 2.0 * flow_speed / re
    cfg = KelvinHelmholtzConfig(
        shape=(nx, ny),
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        viscosity=viscosity,
        resistivity=0.0,
        flow_speed=flow_speed,
    )
    print(
        f"Running incompressible KH: grid {nx}x{ny}, Re={re:.2e}, nu={viscosity:.2e}, "
        f"dt={dt:.2e}, steps={cfg.steps}, save_every={save_every}",
        flush=True,
    )
    result = run_kelvin_helmholtz_dye(cfg)
    result.trajectory.times.block_until_ready()

    times = np.asarray(result.trajectory.times)
    dye = np.asarray(result.trajectory.states.dye)
    omega = np.asarray(result.trajectory.states.mhd.omega)
    entropy = np.asarray(result.entropy)

    INCOMPRESSIBLE_OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INCOMPRESSIBLE_OUTDIR / "kh_incompressible.npz",
        times=times,
        dye=dye,
        omega=omega,
        entropy=entropy,
    )

    indices = [int(np.argmin(np.abs(times - t))) for t in SNAPSHOT_TIMES]
    used_times = times[indices]
    np.savez_compressed(
        INCOMPRESSIBLE_OUTDIR / "dye_snapshots_t2_4_6_8.npz",
        times=used_times,
        dye=dye[indices],
    )
    _plot_snapshots(dye, times, indices, INCOMPRESSIBLE_OUTDIR / "dye_snapshots_t2_4_6_8.png")

    volume.commit()
    lines = [f"requested {t} -> used t = {u:.4f}" for t, u in zip(SNAPSHOT_TIMES, used_times)]
    summary = (
        f"saved {len(times)} snapshots; min dye entropy {entropy.min():.4f}; "
        f"finite: {bool(np.isfinite(dye).all())}\n" + "\n".join(lines)
    )
    print(summary, flush=True)
    return summary


@app.local_entrypoint()
def main(
    nx: int = 0,
    ny: int = 0,
    re: float = 0.0,
    t_end: float = 8.0,
    dt: float = 0.0,
    save_every: int = 0,
):
    """Submit the run and return at once. Zero for nx, ny, re, dt or save_every means the default.

    The run is spawned, not awaited, so exiting this command (or pressing Ctrl+C)
    does not cancel it. Follow it with ``modal app logs <app-id>``.
    """
    call = run_kh_incompressible.spawn(
        nx or 1024,
        ny or 2048,
        re or 1.0e6,
        1.0,
        dt or 2.5e-4,
        t_end,
        save_every or 400,
    )
    folder = "kh_incompressible_modal"

    print(f"\nSubmitted incompressible run. Function call ID: {call.object_id}")
    print("Check progress with: modal app logs <app-id>   (app ID is shown by `modal app list`)")
    print("\nWhen it finishes, pull the snapshots:")
    print(f"  mkdir -p outputs/{folder}")
    for name in ("dye_snapshots_t2_4_6_8.png", "dye_snapshots_t2_4_6_8.npz"):
        print(f"  modal volume get mhx-outputs {folder}/{name} outputs/{folder}/{name}")
