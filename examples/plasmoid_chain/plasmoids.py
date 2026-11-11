"""Many-plasmoid reduced-MHD tearing example (periodic double Harris sheet).

Two thin Harris current sheets in a long periodic box (Lx = 2*SHEET_SEP,
Ly = 8*pi) are seeded with a coherent tearing mode or broadband noise and
advanced with RK4 plus a 2/3-rule dealiasing filter and hyper-resistivity. The
run is split into blocks of SAVE_EVERY steps, each advanced with
``jax.lax.scan``, and one frame is stored per block.

How to get MANY plasmoids:

1. Long, thin sheets: the tearing wavelength is ``lambda ~ 10*WIDTH``, so
   ``WIDTH = 0.05`` fits ~50 islands along each ``Ly = 8*pi`` sheet.
2. Broadband seeding: ``SEED_TYPE = "noise"`` excites every tearing mode at
   once, so all grow simultaneously into a chain.
3. High global Lundquist number ``S_L = Ly*vA/eta``: with ``ETA = 5e-5``,
   ``S_L ~ 5e5``, far above the plasmoid-instability onset (``S ~ 1e4``).
4. Hyper-resistivity ``-eta4*laplacian^2`` damps grid-scale noise without
   touching the low-k tearing modes. With ``ETA4 = None`` it is auto-scaled so
   the grid-scale damping rate ``eta4*k_max^4`` is ~100/s at any resolution.
5. Two sheets give two chains plus sheet-sheet interaction.

The nonlinear phase thins the sheets towards the grid scale, where
pseudo-spectral codes blow up without dealiasing, so a 2/3-rule spectral filter
is applied after every RK4 step.

Writes to OUTDIR: ``plasmoids.gif`` (coloured field map + flux contours),
``plasmoids_final.png``, ``plasmoids_diagnostics.png`` / ``.npz`` (energies,
max |J_z|, y-mode amplitudes), ``plasmoids_final_state.npz`` and
``plasmoids_trajectory.npz`` (raw psi and current density per saved frame).

Runtime: the defaults (512 x 1024, 30000 steps) take tens of minutes on a GPU
and much longer on a CPU. For a quick check use NX = 128, NY = 256, T_END = 5.
To scale up raise NX, NY and T_END (``eta4`` re-scales automatically); when
raising SHEET_SEP keep roughly ``NX ~ 80*SHEET_SEP``. Set CROP = (0.1, 0.4) to
zoom on one sheet, FIELD = "jz" or "island" to colour another field. This is an
exploratory demonstration, not a production Sweet-Parker plasmoid-chain claim.
Run from the repository root:

    python examples/plasmoid_chain/plasmoids.py
"""

import time
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_energy, mode_amplitude
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import laplacian, spectral_wavenumbers
from mhx.physics.equilibria import PeriodicDoubleHarrisEquilibrium
from mhx.physics.terms import HyperResistivityTerm
from mhx.runtime import configure_jax
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import rk4_step

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/examples/plasmoids")
NX = 512  # grid points in x
NY = 1024  # grid points in y
SHEET_SEP = 12.566370614359172  # separation of the two Harris sheets (4*pi); Lx = 2*SHEET_SEP
T_END = 60.0  # end time
DT = 0.002  # time step
SAVE_EVERY = 250  # steps per block / between saved frames
ETA = 5.0e-5  # resistivity (magnetic diffusivity)
NU = 5.0e-5  # viscosity
ETA4 = None  # hyper-resistivity and hyper-viscosity; None auto-scales with resolution
WIDTH = 0.05  # half-width of each Harris sheet
SEED_TYPE = "single"  # "single" coherent Fourier mode or "noise" broadband white noise
MODE_Y = 1  # y mode index of the coherent seed; wavelength = Ly / MODE_Y
PERTURBATION_AMP = None  # coherent flux amplitude; None means 0.01 * WIDTH
NOISE_AMP = 2.0e-3  # white-noise seed amplitude (SEED_TYPE = "noise")
SEED = 42  # RNG seed for reproducible noise
CROP = None  # x-fraction crop of the frames, e.g. (0.10, 0.40); None shows the full box
FIELD = "psi"  # coloured field: "psi", "jz" (current density) or "island" (psi minus y-mean)
SCALE = "frame"  # colour limits: "frame" (per-frame percentile) or "fixed" (across all frames)
TRACKED_Y_MODES = (1, 2, 4, 8, 16, 32)  # y-harmonics monitored for multi-mode growth


# --- Helpers ---------------------------------------------------------------
def advance_block(state, step, steps):
    """Advance ``steps`` full steps with ``jax.lax.scan`` (memory-safe blocks)."""

    def body(carry, _):
        return step(carry), None

    return jax.lax.scan(body, state, None, length=steps)[0]


def render_field(state, grid):
    """Return the 2-D array to colour for the chosen FIELD."""
    psi = np.asarray(state.psi)
    if FIELD == "psi":
        return psi
    if FIELD == "jz":
        return -np.asarray(laplacian(state.psi, lengths=grid.lengths))
    return psi - psi.mean(axis=1, keepdims=True)  # island


def colour_limit(values):
    """Symmetric colour limit from the 1st/99th percentiles (1 if degenerate)."""
    lo, hi = np.percentile(values, [1.0, 99.0])
    limit = max(abs(float(lo)), abs(float(hi)))
    return limit if np.isfinite(limit) and limit != 0.0 else 1.0


def render_frame(field, grid, time_value, limit, label):
    """Draw a coloured field map + flux contours; return an RGB array."""
    nx = field.shape[0]
    i0, i1 = 0, nx
    if CROP is not None:
        i0 = int(CROP[0] * nx)
        i1 = max(int(CROP[1] * nx), i0 + 1)
        field = field[i0:i1, :]
    dx = grid.spacing[0]
    extent = (i0 * dx, i1 * dx, 0.0, grid.lengths[1])
    aspect = grid.lengths[1] / grid.lengths[0]
    fig, ax = plt.subplots(figsize=(4.2, 4.2 * aspect), constrained_layout=True)
    image = ax.imshow(
        field.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
        extent=extent, aspect="auto",
    )
    fig.colorbar(image, ax=ax, shrink=0.9, label=label)
    # Contours must share the imshow extent/origin, otherwise they use array indices.
    ax.contour(
        field.T, levels=48, colors="black", linewidths=0.4, alpha=0.6,
        extent=extent, origin="lower",
    )
    ax.set_title(f"t = {time_value:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


# --- Run -------------------------------------------------------------------
configure_jax(enable_x64=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

if CROP is not None and not (0.0 <= CROP[0] < CROP[1] <= 1.0):
    raise ValueError("CROP must be two increasing fractions in [0, 1]")
if SHEET_SEP <= 0.0:
    raise ValueError("SHEET_SEP must be positive")
if MODE_Y < 1:
    raise ValueError("MODE_Y must be a positive integer")
if PERTURBATION_AMP is not None and PERTURBATION_AMP < 0.0:
    raise ValueError("PERTURBATION_AMP must be non-negative")

Lx, Ly = 2.0 * SHEET_SEP, 8.0 * np.pi
grid = CartesianGrid.from_mesh_config(
    MeshConfig(shape=(NX, NY), lower=(0.0, 0.0), upper=(Lx, Ly))
)

# Hyper-resistivity: the default makes the grid-scale damping rate eta4*k_max^4 ~100/s.
k_max = max(np.pi * NX / Lx, np.pi * NY / Ly)
eta4 = 100.0 / k_max**4 if ETA4 is None else ETA4

lambda_island = 10.0 * WIDTH  # lambda ~ 2*pi/k, k*a ~ 0.6
expected_per_sheet = max(1, int(round(Ly / lambda_island)))
s_lundquist = Ly / ETA  # v_A ~ 1 in these units
print("=" * 70)
print(f"Double-Harris many-plasmoid run: {NX}x{NY} on [{Lx:.3f}, {Ly:.3f}]")
print(f"  sheet width a      = {WIDTH:.4g}  (a/Lx = {WIDTH / Lx:.4g})")
print(f"  sheet separation   = {SHEET_SEP:.3g}  (sheets at x = Lx/4 and 3Lx/4)")
print(f"  resistivity eta    = {ETA:.3g}  ->  S_L = Ly/eta ~ {s_lundquist:.2g}")
print(f"  hyper-resistivity  = {eta4:.3g}  (grid damping ~100/s at k_max={k_max:.3g})")
print(
    f"  tearing wavelength ~ {lambda_island:.3g}  ->  "
    f"~{expected_per_sheet} islands per sheet (2 sheets)"
)
print("=" * 70, flush=True)

# mode_x = 2 is periodic and has extrema at both current sheets.
coherent_amplitude = 0.01 * WIDTH if PERTURBATION_AMP is None else PERTURBATION_AMP
equilibrium = PeriodicDoubleHarrisEquilibrium(
    width=WIDTH,
    amplitude=1.0,
    perturbation_amplitude=coherent_amplitude if SEED_TYPE == "single" else 0.0,
    perturbation_mode=(2, MODE_Y),
)
base = equilibrium.initial_state(grid)
if SEED_TYPE == "noise":
    noise = NOISE_AMP * jax.random.normal(jax.random.PRNGKey(SEED), base.psi.shape)
    state = ReducedMHDState(psi=base.psi + noise, omega=base.omega)
else:
    state = base
    print(
        f"  perturbation = {coherent_amplitude:.3g} cos(4*pi*x/Lx) "
        f"cos(2*pi*{MODE_Y}*y/Ly); lambda_y={Ly / MODE_Y:.3g}"
    )

# 2/3-rule dealiasing: reduced_mhd_rhs does not dealias its nonlinear products, and
# aliased energy piles up at the Nyquist scale as the sheets thin (NaN otherwise).
kx = spectral_wavenumbers(grid.shape[0], grid.lengths[0])
ky = spectral_wavenumbers(grid.shape[1], grid.lengths[1])
cutoff_x = (2.0 / 3.0) * np.pi * grid.shape[0] / grid.lengths[0]
cutoff_y = (2.0 / 3.0) * np.pi * grid.shape[1] / grid.lengths[1]
dealias_mask = (jnp.abs(kx)[:, None] < cutoff_x) & (jnp.abs(ky)[None, :] < cutoff_y)

params = ReducedMHDParams(resistivity=ETA, viscosity=NU)
hyper_term = HyperResistivityTerm(eta4=eta4, nu4=eta4)


def rhs(s):
    return reduced_mhd_rhs(s, params, lengths=grid.lengths, terms=(hyper_term,))


def step(s):
    """One RK4 step followed by the dealiasing filter."""
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

field_label = {"psi": r"$\psi$", "jz": r"$j_z$", "island": r"$\Delta\psi$"}[FIELD]
field_frames = [render_field(state, grid)]
psi_frames = [np.asarray(state.psi)]
t_history = [0.0]
ek_history = [float(kinetic_energy(state, lengths=grid.lengths))]
eb_history = [float(magnetic_energy(state, lengths=grid.lengths))]
jz0 = np.abs(np.asarray(current_density(state.psi, lengths=grid.lengths)))
jzmax_history = [float(np.max(jz0))]
mode_history = [np.array([float(mode_amplitude(state, mode=(0, m))) for m in TRACKED_Y_MODES])]

run_seconds = 0.0
for block in range(1, num_blocks + 1):
    start = time.perf_counter()
    state = advance_executable(state)
    jax.block_until_ready(state.psi)
    elapsed = time.perf_counter() - start
    run_seconds += elapsed

    time_value = block * SAVE_EVERY * DT
    if not np.isfinite(np.asarray(state.psi)).all():
        print(f"NaN detected at block {block}; simulation crashed - stop.", flush=True)
        break

    field_frames.append(render_field(state, grid))
    psi_frames.append(np.asarray(state.psi))
    jz = np.abs(np.asarray(current_density(state.psi, lengths=grid.lengths)))
    t_history.append(time_value)
    ek_history.append(float(kinetic_energy(state, lengths=grid.lengths)))
    eb_history.append(float(magnetic_energy(state, lengths=grid.lengths)))
    jzmax_history.append(float(np.max(jz)))
    mode_history.append(
        np.array([float(mode_amplitude(state, mode=(0, m))) for m in TRACKED_Y_MODES])
    )
    print(
        f"block {block:4d}/{num_blocks} | t = {time_value:7.2f} | "
        f"{elapsed:6.2f}s | max|Jz| = {float(np.max(jz)):.3e}",
        flush=True,
    )

print(f"Run time: {run_seconds:.3f} s", flush=True)

# --- Results ---------------------------------------------------------------
# Render after the full run so SCALE = "fixed" can span all frames.
fixed_limit = None
if SCALE == "fixed":
    fixed_limit = colour_limit(np.asarray([f for f in field_frames if np.isfinite(f).all()]))
frames = []
for idx, field in enumerate(field_frames):
    limit = colour_limit(field) if SCALE == "frame" else fixed_limit
    frames.append(render_frame(field, grid, t_history[idx], limit, field_label))

gif_path = OUTDIR / "plasmoids.gif"
imageio.mimsave(gif_path, frames, duration=0.1)
plt.imsave(OUTDIR / "plasmoids_final.png", frames[-1])

diag_path = OUTDIR / "plasmoids_diagnostics.npz"
np.savez(
    diag_path,
    time=np.array(t_history),
    kinetic_energy=np.array(ek_history),
    magnetic_energy=np.array(eb_history),
    jz_max=np.array(jzmax_history),
    y_mode_amplitudes=np.array(mode_history),
    y_modes=np.array(TRACKED_Y_MODES),
    nx=NX,
    ny=NY,
    lx=Lx,
    ly=Ly,
    width=WIDTH,
    eta=ETA,
    eta4=eta4,
    seed=SEED,
)
state_path = OUTDIR / "plasmoids_final_state.npz"
np.savez(
    state_path,
    psi=np.asarray(state.psi),
    omega=np.asarray(state.omega),
    nx=NX,
    ny=NY,
    lx=Lx,
    ly=Ly,
)

# Raw flux and current over the saved frames, so any field can be re-rendered.
psi_series = np.asarray(psi_frames, dtype=np.float64)
current_series = np.asarray(
    [-np.asarray(laplacian(f, lengths=grid.lengths), dtype=np.float64) for f in psi_series]
)
trajectory_path = OUTDIR / "plasmoids_trajectory.npz"
np.savez_compressed(
    trajectory_path,
    time=np.array(t_history),
    psi=psi_series,
    current_density=current_series,
    nx=NX,
    ny=NY,
    lx=Lx,
    ly=Ly,
    eta=ETA,
    eta4=eta4,
    seed=SEED,
    field=FIELD,
)

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(t_history, eb_history, "r-", label=r"$E_B$")
axs[0].plot(t_history, ek_history, "b-", label=r"$E_K$")
axs[0].set_ylabel("energy")
axs[0].legend(frameon=False)
axs[1].semilogy(t_history, jzmax_history, "k-")
axs[1].set_ylabel(r"max $|J_z|$")
axs[1].set_xlabel("time")
fig.tight_layout()
diag_png = OUTDIR / "plasmoids_diagnostics.png"
fig.savefig(diag_png, dpi=160)
plt.close(fig)

print("=" * 70)
print(
    "Y-mode amplitudes at the end: "
    f"{dict(zip(TRACKED_Y_MODES, np.round(mode_history[-1], 6), strict=False))}"
)
for written in (gif_path, OUTDIR / "plasmoids_final.png", diag_png, diag_path, state_path,
                trajectory_path):
    print(f"Saved {written}")
print("=" * 70)
