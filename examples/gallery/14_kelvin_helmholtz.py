"""Run the incompressible Kelvin--Helmholtz roll-up and save dye snapshots.

A double shear layer, perturbed with a single mode, rolls up while a passive
dye follows the flow. The script runs the periodic problem with
``mhx.benchmarks.kelvin_helmholtz``, prints the dye entropy, and saves the full
history plus a figure of the dye at the requested times. The defaults are a
small laptop run; the poster comparison against Dedalus used ``NX, NY =
1024, 2048``, ``RE = 1e6``, ``DT = 2.5e-4`` and ``T_END = 8`` on a GPU.

Run from the repository root:

    python examples/gallery/14_kelvin_helmholtz.py
"""

import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from mhx.benchmarks.kelvin_helmholtz import KelvinHelmholtzConfig, run_kelvin_helmholtz_dye
from mhx.runtime import configure_jax

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/gallery/kelvin_helmholtz")
NX = 128  # grid points across the unit-length periodic direction
NY = 256  # grid points along the length-2 direction
RE = 1.0e4  # Reynolds number: nu = 2 u / Re with shear velocity jump 2u and L = 1
FLOW_SPEED = 1.0  # shear velocity u
DT = 1.0e-3  # time step
T_END = 4.0  # final time
SAVE_EVERY = 100  # steps between saved states
SNAPSHOT_TIMES = (1.0, 2.0, 3.0, 4.0)  # times to show in the figure

# --- Run -------------------------------------------------------------------
configure_jax(enable_x64=True)
viscosity = 2.0 * FLOW_SPEED / RE
config = KelvinHelmholtzConfig(
    shape=(NX, NY),
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
    viscosity=viscosity,
    resistivity=0.0,
    flow_speed=FLOW_SPEED,
)
print(
    f"Kelvin-Helmholtz: grid {NX}x{NY}, Re={RE:.2e}, nu={viscosity:.2e}, "
    f"dt={DT:.2e}, {config.steps} steps, saving every {SAVE_EVERY}",
    flush=True,
)
start = time.perf_counter()
result = run_kelvin_helmholtz_dye(config)
jax.block_until_ready(result.trajectory.times)
print(f"Run finished in {time.perf_counter() - start:.1f} s", flush=True)

# --- Results ---------------------------------------------------------------
times = np.asarray(result.trajectory.times)
dye = np.asarray(result.trajectory.states.dye)
omega = np.asarray(result.trajectory.states.mhd.omega)
entropy = np.asarray(result.entropy)
print(f"Saved {len(times)} states; dye entropy {entropy[0]:.4f} -> {entropy[-1]:.4f}")
print(f"All values finite: {bool(np.isfinite(dye).all())}")

OUTDIR.mkdir(parents=True, exist_ok=True)
history_path = OUTDIR / "kh_incompressible.npz"
np.savez_compressed(history_path, times=times, dye=dye, omega=omega, entropy=entropy)

indices = [int(np.argmin(np.abs(times - t))) for t in SNAPSHOT_TIMES]
for requested, index in zip(SNAPSHOT_TIMES, indices, strict=True):
    print(f"Snapshot t = {requested:g}: using saved t = {times[index]:.4f}")
figure, axes = plt.subplots(
    1, len(indices), figsize=(4 * len(indices), 4), constrained_layout=True
)
axes = np.atleast_1d(axes)
for axis, index in zip(axes, indices, strict=True):
    image = axis.imshow(
        dye[index].T,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 2.0),
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_title(f"dye, t = {times[index]:.3f}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
figure.colorbar(image, ax=axes, shrink=0.8, label="dye concentration c")
figure_path = OUTDIR / "dye_snapshots.png"
figure.savefig(figure_path, dpi=150)
plt.close(figure)

print(f"History: {history_path}")
print(f"Figure:  {figure_path}")
