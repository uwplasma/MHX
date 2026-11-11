"""Inviscid Orszag--Tang reduced-MHD vortex energy conservation benchmark.

Evolves the periodic inviscid Orszag--Tang vortex on a 2 pi box with the Arakawa
reduced-MHD right-hand side and RK4, then checks that the total energy is conserved
and the magnetic field stays divergence free. It writes diagnostics.json,
validation.json, an optional NPZ history, a summary figure, optional GIFs and a
manifest under OUTDIR, and prints whether the gates passed. The default
128 x 128 run (4000 steps) takes about a minute on a laptop CPU; lower NX, NY and
T_END for a quick look, raise NX and NY (and lower DT) to scale up. Run from the
repository root:

    python examples/orszag_tang/benchmark_inviscid.py
"""

import json
import time
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from mhx.benchmarks.orszag_tang import orszag_tang_initial_state
from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_divergence_linf, magnetic_energy
from mhx.equations.arakawa_reduced_mhd import arakawa_reduced_mhd_rhs
from mhx.equations.reduced_mhd import current_density
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.plotting import sample_indices
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/inviscid_orszag_tang_output")
NX = 128  # grid points in x
NY = 128  # grid points in y
DT = 5.0e-4  # time step
T_END = 2.0  # end time (an integer multiple of DT)
SAVE_EVERY = 40  # steps between saved states
MOVIES = False  # also write flux, current and vorticity GIFs
SAVE_NPZ = True  # save the full history (heavy) as an NPZ file
MAX_RELATIVE_ENERGY_ERROR = 1.0e-6  # gate on |E(T) - E(0)| / E(0)
MAX_MAGNETIC_DIVERGENCE_LINF = 1.0e-10  # gate on the final max |div B|

SCHEMA = "mhx.validation.inviscid_orszag_tang.v1"
DOMAIN = (2.0 * np.pi, 2.0 * np.pi)


def write_scalar_movie(fields, path, *, cmap, symmetric, max_frames=36):
    """Write a GIF of the scalar field frames ``fields[t, x, y]``."""
    indices = sample_indices(fields.shape[0], max_frames)
    values = np.asarray(fields)[indices]
    if symmetric:
        vmax = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        vmin = -vmax
    else:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    colormap = colormaps[cmap]
    frames = []
    for field in values:
        normalized = np.clip((field.T - vmin) / (vmax - vmin), 0.0, 1.0)
        frames.append((255.0 * colormap(normalized)[..., :3]).astype(np.uint8))
    imageio.mimsave(path, frames, duration=90, loop=0, palettesize=48)
    return path


# --- Run -------------------------------------------------------------------
if min(NX, NY) < 8:
    raise ValueError("NX and NY must be at least 8 in each periodic direction")
if DT <= 0.0 or T_END <= 0.0:
    raise ValueError("DT and T_END must be positive")
steps = int(round(T_END / DT))
if steps < 4:
    raise ValueError("T_END / DT must produce at least four RK4 steps")
if not np.isclose(steps * DT, T_END):
    raise ValueError("T_END must be an integer multiple of DT")
if SAVE_EVERY < 1:
    raise ValueError("SAVE_EVERY must be >= 1")
if steps % SAVE_EVERY != 0:
    raise ValueError("T_END / DT must be divisible by SAVE_EVERY")
if steps // SAVE_EVERY < 2:
    raise ValueError("configuration must save at least two non-initial samples")

print("Running Inviscid Orszag-Tang Benchmark...")
print(f"Grid: {NX}x{NY} | dt: {DT} | t_end: {T_END} ({steps} steps) | Movies: {MOVIES}")
print(f"Output directory: {OUTDIR.resolve()}; devices: {jax.devices()}", flush=True)
OUTDIR.mkdir(parents=True, exist_ok=True)
figure_dir = OUTDIR / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(NX, NY), lower=(0.0, 0.0), upper=DOMAIN))
initial_state = orszag_tang_initial_state(grid)
params = ReducedMHDParams(resistivity=0.0, viscosity=0.0)


def rhs(state: ReducedMHDState) -> ReducedMHDState:
    return arakawa_reduced_mhd_rhs(state, params, lengths=grid.lengths)


start = time.perf_counter()
trajectory = evolve_rk4(initial_state, rhs, dt=DT, steps=steps, save_every=SAVE_EVERY)
jax.block_until_ready(trajectory.states.psi)
run_seconds = time.perf_counter() - start  # includes JIT compilation
print(f"Integration time (including compile): {run_seconds:.2f} s", flush=True)

time_axis = np.concatenate(([0.0], np.asarray(trajectory.times, dtype=np.float64)))
psi = np.concatenate(
    (
        np.asarray(initial_state.psi, dtype=np.float64)[None, ...],
        np.asarray(trajectory.states.psi, dtype=np.float64),
    ),
    axis=0,
)
omega = np.concatenate(
    (
        np.asarray(initial_state.omega, dtype=np.float64)[None, ...],
        np.asarray(trajectory.states.omega, dtype=np.float64),
    ),
    axis=0,
)
current = np.asarray(
    [np.asarray(current_density(frame, lengths=grid.lengths)) for frame in psi],
    dtype=np.float64,
)
states = tuple(
    ReducedMHDState(psi=jnp.asarray(psi[i]), omega=jnp.asarray(omega[i]))
    for i in range(time_axis.size)
)
magnetic = np.asarray(
    [float(magnetic_energy(state, lengths=grid.lengths)) for state in states],
    dtype=np.float64,
)
kinetic = np.asarray(
    [float(kinetic_energy(state, lengths=grid.lengths)) for state in states],
    dtype=np.float64,
)
total = magnetic + kinetic
relative_energy_error = float(abs((total[-1] - total[0]) / total[0]))
final_divergence = float(magnetic_divergence_linf(states[-1], lengths=grid.lengths))

checks = {
    "finite_arrays": bool(
        np.isfinite(psi).all() and np.isfinite(omega).all() and np.isfinite(total).all()
    ),
    "energy_conserved": relative_energy_error < MAX_RELATIVE_ENERGY_ERROR,
    "magnetic_divergence_preserved": final_divergence <= MAX_MAGNETIC_DIVERGENCE_LINF,
}
passed = all(checks.values())
diagnostics = {
    "schema": SCHEMA,
    "shape": [NX, NY],
    "domain": [0.0, DOMAIN[0], 0.0, DOMAIN[1]],
    "resistivity": 0.0,
    "viscosity": 0.0,
    "dt": DT,
    "t_end": T_END,
    "steps": steps,
    "save_every": SAVE_EVERY,
    "samples": int(time_axis.size),
    "initial_total_energy": float(total[0]),
    "final_total_energy": float(total[-1]),
    "relative_energy_error": relative_energy_error,
    "final_magnetic_divergence_linf": final_divergence,
    "references": {
        "classic_test": "Inviscid Orszag--Tang vortex energy conservation check.",
    },
}
validation = {
    "schema": "mhx.validation.inviscid_orszag_tang.gates.v1",
    "passed": passed,
    "checks": checks,
    "thresholds": {
        "max_relative_energy_error": MAX_RELATIVE_ENERGY_ERROR,
        "max_magnetic_divergence_linf": MAX_MAGNETIC_DIVERGENCE_LINF,
    },
    "diagnostics": diagnostics,
}

# --- Results ---------------------------------------------------------------
print(f"samples: {time_axis.size}, initial energy {total[0]:.8f}, final energy {total[-1]:.8f}")
print(f"relative energy error: {relative_energy_error:.3e} (gate {MAX_RELATIVE_ENERGY_ERROR:g})")
print(f"final magnetic divergence (Linf): {final_divergence:.3e} "
      f"(gate {MAX_MAGNETIC_DIVERGENCE_LINF:g})")

diagnostics_path = OUTDIR / "diagnostics.json"
validation_path = OUTDIR / "validation.json"
diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
outputs = {"diagnostics": diagnostics_path.name, "validation": validation_path.name}

if SAVE_NPZ:
    history_path = OUTDIR / "inviscid_orszag_tang.npz"
    np.savez_compressed(
        history_path,
        schema=SCHEMA,
        time=time_axis,
        psi=psi,
        omega=omega,
        current_density=current,
        magnetic_energy=magnetic,
        kinetic_energy=kinetic,
        total_energy=total,
        relative_energy_error=relative_energy_error,
        final_magnetic_divergence_linf=final_divergence,
    )
    outputs["history"] = history_path.name

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
axes[0].plot(time_axis, magnetic, label=r"$E_B$", color="navy", linewidth=2)
axes[0].plot(time_axis, kinetic, label=r"$E_K$", color="crimson", linewidth=2)
axes[0].plot(time_axis, total, label=r"$E$", color="purple", linestyle="--", linewidth=2)
axes[0].set_title("Inviscid Orszag--Tang Energy Conservation")
axes[0].set_xlabel("time")
axes[0].set_ylabel("energy")
axes[0].legend(frameon=False, fontsize="small")

axes[1].plot(time_axis, (total - total[0]) / total[0], color="red", linewidth=2)
axes[1].set_title("Relative Energy Error: $(E(t) - E(0)) / E(0)$")
axes[1].set_xlabel("time")
axes[1].set_ylabel("Relative Error")

# Clip the color scale to 3 standard deviations so intense current sheets do not
# wash out the whole background.
vmax = max(3.0 * float(np.std(current[-1])), 1.0)
im = axes[2].imshow(
    current[-1].T, cmap="jet", origin="lower", vmin=-vmax, vmax=vmax,
    extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
)
axes[2].set_title("Final Current Density ($j_z$)")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(im, ax=axes[2], shrink=0.75)
summary_path = figure_dir / "inviscid_orszag_tang_summary.png"
fig.savefig(summary_path, dpi=180)
plt.close(fig)
outputs["summary"] = str(summary_path.relative_to(OUTDIR))

if MOVIES:
    print("Writing GIFs ...", flush=True)
    for name, fields, cmap, symmetric in (
        ("flux", psi, "viridis", False),
        ("current", current, "RdBu_r", True),
        ("vorticity", omega, "RdBu_r", True),
    ):
        movie = write_scalar_movie(
            fields, figure_dir / f"inviscid_orszag_tang_{name}.gif", cmap=cmap,
            symmetric=symmetric,
        )
        outputs[f"{name}_movie"] = str(movie.relative_to(OUTDIR))

write_manifest(
    OUTDIR / "manifest.json",
    config=diagnostics,
    outputs=outputs,
    claim_level="validation",
    claim_scope="Inviscid reduced-MHD Orszag--Tang vortex energy conservation validation.",
)
print(f"wrote {OUTDIR / 'manifest.json'}")
print(f"passed={passed}")
if not passed:
    print("Failed checks:")
    for check, passed_check in checks.items():
        if not passed_check:
            print(f"  - {check}")
