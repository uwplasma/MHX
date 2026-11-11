"""Hybrid forced-turbulent-reconnection run on the plasmoid-chain domain.

Combines the forced-turbulent-reconnection setup (``mhx.benchmarks.turbulence``)
with the long thin two-sheet domain of ``plasmoids.py``:

- Domain and sheets from the plasmoid chain: ``2*SHEET_SEP x 8*pi`` with two thin
  Harris sheets.
- Seed, forcing and rendered field from forced-turbulent reconnection: a seeded
  tearing mode plus broadband turbulence, a persistent large-scale vorticity
  source applied every step, and an out-of-plane current density ``j_z`` map with
  flux contours. RK4 steps are followed by a 2/3-rule dealiasing filter and
  advanced in blocks of SAVE_EVERY steps with ``jax.lax.scan``.

Writes to OUTDIR: ``forced_turbulence_trajectory.npz`` (``time``, ``psi``,
``current_density``) and ``forced_turbulent_reconnection.gif``.

Runtime: the defaults (256 x 1024, 12000 steps) take several minutes on a GPU
and much longer on a CPU. For a quick check use NX = 64, NY = 256, T_END = 2. To
scale up raise NX, NY and T_END. Run from the repository root:

    python examples/plasmoid_chain/forced_turbulence.py
"""

import time
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mhx.benchmarks.turbulence import _broadband_scalar_field, turbulent_initial_state
from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import laplacian, spectral_wavenumbers
from mhx.physics.equilibria import PeriodicDoubleHarrisEquilibrium
from mhx.runtime import configure_jax
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import rk4_step

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/forced_turbulence")
NX = 256  # grid points in x
NY = 1024  # grid points in y
SHEET_SEP = 12.57  # separation of the two Harris sheets; Lx = 2*SHEET_SEP
WIDTH = 0.32  # half-width of each Harris sheet
T_END = 60.0  # end time
DT = 0.005  # time step
SAVE_EVERY = 200  # steps per block / between saved frames
ETA = 1.5e-3  # resistivity (magnetic diffusivity)
NU = 1.5e-3  # viscosity
PERTURBATION_AMPLITUDE = 1.0e-2  # seeded tearing-mode flux amplitude
TURBULENT_FLUX_AMPLITUDE = 1.5e-2  # broadband turbulent flux amplitude
TURBULENT_FLOW_AMPLITUDE = 1.5e-2  # broadband turbulent flow amplitude
FORCING_AMPLITUDE = 2.0e-3  # amplitude of the persistent vorticity source
SEED = 11  # RNG seed for the turbulence and the forcing
GIF_FRAMES = 18  # maximum number of frames in the GIF


def advance_block(state, step, steps):
    """Advance ``steps`` full steps with ``jax.lax.scan``."""

    def body(carry, _):
        return step(carry), None

    return jax.lax.scan(body, state, None, length=steps)[0]


# --- Run -------------------------------------------------------------------
configure_jax(enable_x64=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

Lx, Ly = 2.0 * SHEET_SEP, 8.0 * np.pi
grid = CartesianGrid.from_mesh_config(
    MeshConfig(shape=(NX, NY), lower=(0.0, 0.0), upper=(Lx, Ly))
)
print(f"Forced turbulent reconnection: {NX}x{NY} on [{Lx:.3f}, {Ly:.3f}], width {WIDTH}")
print(f"  eta = {ETA:.3g}, nu = {NU:.3g}, forcing amplitude = {FORCING_AMPLITUDE:.3g}, seed {SEED}")

# Sheet + turbulence composite initial state (forced-TR recipe).
sheet = PeriodicDoubleHarrisEquilibrium(
    width=WIDTH,
    perturbation_amplitude=PERTURBATION_AMPLITUDE,
    perturbation_mode=(0, 1),
).initial_state(grid)
turb = turbulent_initial_state(
    grid,
    seed=SEED,
    flux_amplitude=TURBULENT_FLUX_AMPLITUDE,
    flow_amplitude=TURBULENT_FLOW_AMPLITUDE,
    kmin=1,
    kmax=4,
)
state = ReducedMHDState(psi=sheet.psi + turb.psi, omega=sheet.omega + turb.omega)

# Persistent large-scale vorticity source (forced-TR recipe).
forcing_stream = _broadband_scalar_field(grid, seed=SEED + 2027, kmin=1, kmax=3)
forcing_omega = FORCING_AMPLITUDE * laplacian(jnp.asarray(forcing_stream), lengths=grid.lengths)
params = ReducedMHDParams(resistivity=ETA, viscosity=NU)

# 2/3-rule dealiasing filter applied after every RK4 step.
kx = spectral_wavenumbers(grid.shape[0], grid.lengths[0])
ky = spectral_wavenumbers(grid.shape[1], grid.lengths[1])
cutoff_x = (2.0 / 3.0) * np.pi * grid.shape[0] / grid.lengths[0]
cutoff_y = (2.0 / 3.0) * np.pi * grid.shape[1] / grid.lengths[1]
dealias_mask = (jnp.abs(kx)[:, None] < cutoff_x) & (jnp.abs(ky)[None, :] < cutoff_y)


def rhs(s):
    base = reduced_mhd_rhs(s, params, lengths=grid.lengths)
    return ReducedMHDState(psi=base.psi, omega=base.omega + forcing_omega)


def step(s):
    s = rk4_step(s, rhs, DT)
    return ReducedMHDState(
        psi=jnp.real(jnp.fft.ifftn(dealias_mask * jnp.fft.fftn(s.psi))),
        omega=jnp.real(jnp.fft.ifftn(dealias_mask * jnp.fft.fftn(s.omega))),
    )


num_steps = int(T_END / DT)
num_blocks = num_steps // SAVE_EVERY
if num_blocks < 1:
    raise ValueError("T_END is too short for SAVE_EVERY")
print(f"{num_steps} steps in {num_blocks} blocks of {SAVE_EVERY}; compiling...", flush=True)

compile_start = time.perf_counter()
advance = jax.jit(advance_block, static_argnames=("step", "steps"))
advance_executable = advance.lower(state, step=step, steps=SAVE_EVERY).compile()
print(f"Compile time: {time.perf_counter() - compile_start:.3f} s", flush=True)

psi_frames = [np.asarray(state.psi)]
t_history = [0.0]
run_seconds = 0.0
for block in range(1, num_blocks + 1):
    t0 = time.perf_counter()
    state = advance_executable(state)
    jax.block_until_ready(state.psi)
    elapsed = time.perf_counter() - t0
    run_seconds += elapsed
    psi_frames.append(np.asarray(state.psi))
    t_history.append(block * SAVE_EVERY * DT)
    print(f"  block {block}/{num_blocks} done in {elapsed:.1f}s", flush=True)
print(f"Run time: {run_seconds:.3f} s", flush=True)

# --- Results ---------------------------------------------------------------
psi_series = np.asarray(psi_frames, dtype=np.float64)
current_series = np.asarray(
    [-np.asarray(laplacian(f, lengths=grid.lengths), dtype=np.float64) for f in psi_series]
)
trajectory = OUTDIR / "forced_turbulence_trajectory.npz"
np.savez_compressed(
    trajectory,
    time=np.array(t_history),
    psi=psi_series,
    current_density=current_series,
    nx=NX,
    ny=NY,
    lx=Lx,
    ly=Ly,
    eta=ETA,
    forcing_amplitude=FORCING_AMPLITUDE,
    seed=SEED,
)

# Compact GIF: j_z map + flux contours.
frames = []
indices = np.unique(np.linspace(0, len(t_history) - 1, GIF_FRAMES, dtype=int))
for idx in indices:
    jz = current_series[idx]
    psi = psi_series[idx]
    vmax = max(float(np.percentile(np.abs(jz), 99.0)), np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=80)
    ax.imshow(jz.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=(0.0, Lx, 0.0, Ly), aspect="auto")
    levels = np.linspace(float(np.percentile(psi, 5.0)), float(np.percentile(psi, 95.0)), 18)
    ax.contour(psi.T, levels=levels, colors="black", linewidths=0.35)
    ax.set_title(f"Forced turbulent reconnection, t={t_history[idx]:.0f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
gif = OUTDIR / "forced_turbulent_reconnection.gif"
imageio.mimsave(gif, frames, duration=90, loop=0, palettesize=32)

print(f"Final max |j_z| = {float(np.max(np.abs(current_series[-1]))):.3e}")
print(f"Saved {trajectory}")
print(f"Saved {gif}")
