"""Run forced 3D incompressible MHD turbulence (paper-matching setup).

Models the balanced-Elsässer forced turbulence of equation (25) from the
3D MHD turbulence paper:

- cubic periodic domain L = 2π
- forcing band n⊥ ∈ [1, 2] with |nz| ≤ 1 (where n = k L/(2π))
- independent Gaussian white-noise realisations for each Elsässer field z±,
  with amplitude A = 0.048
- hyper-resistivity η (−∇²)^r with η = 6.0, r = 2 (η k⁴ damping)
- integration to a few Alfvén crossing times
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

from mhx.equations import mhd3d
from mhx.simulation3d import MHD3DResult
from mhx.state.mhd3d import MHD3DParams, MHD3DState, MHD3DTrajectory
from mhx.time_integrators.exponential import etdrk4_step

# --- settings (edit here) -------------------------------------------------
GRID_SIZE = int(os.environ.get("MHX_TURBULENCE_3D_N", "64"))
SHAPE = (GRID_SIZE,) * 3       # cubic periodic L = 2π  [paper: 64³]
AMP_FORCE = 0.048             # forcing amplitude  [paper: 0.048]
DISSIPATION_ORDER = 2         # hyper-viscosity k⁴
RESISTIVITY = 6.0 / (21**4)   # η  [paper: 6.0 at k_max=21]
VISCOSITY = 6.0 / (21**4)     # matched to resistivity
DT = float(os.environ.get("MHX_TURBULENCE_3D_DT", "2.0e-3"))
T_END = float(os.environ.get("MHX_TURBULENCE_3D_T_END", "200.0"))
SAVE_EVERY = int(os.environ.get("MHX_TURBULENCE_3D_SAVE_EVERY", "500"))
SEED = 11
K_FIT_MIN = 3                 # lower edge of the resolved inertial-range fit
K_FIT_MAX = 16                # below the k≈21 de-aliasing/hyper-dissipation rolloff

output = Path(os.environ.get("MHX_TURBULENCE_3D_OUTDIR", "outputs/gallery/turbulence_3d"))
output.parent.mkdir(parents=True, exist_ok=True)

# --- 1. System Setup ------------------------------------------------------
lengths = (2.0 * jnp.pi,) * 3
params = MHD3DParams(
    viscosity=VISCOSITY,
    resistivity=RESISTIVITY,
    dissipation_order=DISSIPATION_ORDER,
)
k = mhd3d.wavevectors(SHAPE, lengths)
dealias_mask = mhd3d.two_thirds_mask_rfft(SHAPE)
decay = mhd3d.decay_rates(params, k)

def nonlinear(state: MHD3DState) -> MHD3DState:
    return mhd3d.mhd3d_nonlinear(
        state, params, shape=SHAPE, k=k, mask=dealias_mask
    )

# --- 2. Forcing Mask ------------------------------------------------------
kx, ky, kz = jnp.meshgrid(
    jnp.fft.fftfreq(SHAPE[0], d=1.0/SHAPE[0]),
    jnp.fft.fftfreq(SHAPE[1], d=1.0/SHAPE[1]),
    jnp.fft.rfftfreq(SHAPE[2], d=1.0/SHAPE[2]),
    indexing="ij"
)
n_perp_sq = kx**2 + ky**2
# n_perp in [1, 2] implies n_perp_sq in [1, 4]
band_mask = (n_perp_sq >= 1.0 - 1e-5) & (n_perp_sq <= 4.0 + 1e-5) & (jnp.abs(kz) <= 1.0 + 1e-5)
forcing_mask = jnp.where(band_mask, 1.0, 0.0)[None, ...]

# Forcing z± independently with A gives v and b independent noise with A/sqrt(2)
# MHX uses unnormalized FFTs, so spectral fields are O(N^3). We must scale the noise.
noise_scale = (AMP_FORCE / jnp.sqrt(2.0)) * jnp.sqrt(DT) * float(np.prod(SHAPE))

# --- 3. Time Stepping Loop ------------------------------------------------
def step_fn(carry: MHD3DState, key: jax.Array) -> tuple[MHD3DState, MHD3DState]:
    state = carry
    # Deterministic step
    state_next = etdrk4_step(state, nonlinear, decay, DT)

    # Stochastic white-noise step
    key_v, key_b = jax.random.split(key)
    noise_v = jax.random.normal(key_v, state_next.v_hat.shape, dtype=state_next.v_hat.dtype)
    noise_b = jax.random.normal(key_b, state_next.b_hat.shape, dtype=state_next.b_hat.dtype)

    force_v_hat = noise_v * forcing_mask * noise_scale
    force_b_hat = noise_b * forcing_mask * noise_scale

    # Project to divergence-free
    force_v_hat = mhd3d.project(force_v_hat, k)
    force_b_hat = mhd3d.project(force_b_hat, k)

    state_next = MHD3DState(
        v_hat=state_next.v_hat + force_v_hat,
        b_hat=state_next.b_hat + force_b_hat,
    )
    return state_next, state_next


def chunk_fn(carry: MHD3DState, key_chunk: jax.Array) -> tuple[MHD3DState, MHD3DState]:
    keys = jax.random.split(key_chunk, SAVE_EVERY)
    state_next, _ = jax.lax.scan(step_fn, carry, keys)
    return state_next, state_next


@jax.jit
def run_simulation(state0: MHD3DState, keys_chunks: jax.Array) -> tuple[MHD3DState, MHD3DState]:
    return jax.lax.scan(chunk_fn, state0, keys_chunks)

# --- 4. Execution ---------------------------------------------------------
steps = int(round(T_END / DT))
chunks = steps // SAVE_EVERY

print(f"Running stochastic turbulence for {T_END} time units ({steps} steps)...")
key0 = jax.random.PRNGKey(SEED)
keys_chunks = jax.random.split(key0, chunks)

# Start from a zero state
zero_hat = jnp.zeros((3, SHAPE[0], SHAPE[1], SHAPE[2] // 2 + 1), dtype=jnp.complex64)
state0 = MHD3DState(v_hat=zero_hat, b_hat=zero_hat)

t0 = time.perf_counter()
compiled_simulation = run_simulation.lower(state0, keys_chunks).compile()
compile_seconds = time.perf_counter() - t0

t0 = time.perf_counter()
final_state, saved = compiled_simulation(state0, keys_chunks)
jax.block_until_ready(final_state)
run_seconds = time.perf_counter() - t0

times = jnp.arange(1, chunks + 1) * DT * SAVE_EVERY
trajectory = MHD3DTrajectory(times=times, states=saved)

# --- 5. Diagnostics & Time-Averaging --------------------------------------
def _shell_spectrum(field_hat, k, weight, shape: tuple[int, int, int]):
    per_mode = (
        0.5
        * np.asarray(weight)
        * np.sum(np.abs(np.asarray(field_hat)) ** 2, axis=0)
        / float(np.prod(shape) ** 2)
    )
    radii = np.round(np.sqrt(np.sum(np.asarray(k) ** 2, axis=0))).astype(int)
    dk = 2.0 * np.pi / float(shape[0])
    max_radius = int(np.max(radii))
    k_1d = np.arange(1, max_radius + 1)
    power = np.asarray([np.sum(per_mode[radii == radius]) / dk for radius in k_1d])
    return k_1d, power

weight = mhd3d.parseval_weight(SHAPE)

# Time average from t=180 to t=200
t_min, t_max = 180.0, 200.0
valid_indices = np.asarray(jnp.where((times >= t_min) & (times <= t_max))[0])

mag_spectra = []
kin_spectra = []
for idx in valid_indices:
    b_hat_i = jax.tree.map(lambda leaf, i=idx: leaf[i], saved).b_hat
    v_hat_i = jax.tree.map(lambda leaf, i=idx: leaf[i], saved).v_hat
    k_1d, mag_spec = _shell_spectrum(b_hat_i, k, weight, SHAPE)
    _, kin_spec = _shell_spectrum(v_hat_i, k, weight, SHAPE)
    mag_spectra.append(mag_spec)
    kin_spectra.append(kin_spec)

if mag_spectra:
    magnetic_spectrum = np.mean(np.stack(mag_spectra), axis=0)
    kinetic_spectrum = np.mean(np.stack(kin_spectra), axis=0)
    print(
        f"Computed time-averaged spectrum over {len(mag_spectra)} frames "
        f"from t={t_min} to t={t_max}."
    )
else:
    print(f"Warning: No frames found between {t_min} and {t_max}. Using final state only.")
    k_1d, magnetic_spectrum = _shell_spectrum(final_state.b_hat, k, weight, SHAPE)
    _, kinetic_spectrum = _shell_spectrum(final_state.v_hat, k, weight, SHAPE)


# Build a mock Result to use mhx's native print_summary and save
final_energy = mhd3d.energies(final_state, shape=SHAPE)
diagnostics = {
    "initial_total_energy": 0.0,
    "final_total_energy": float(final_energy["total"]),
    "final_kinetic_energy": float(final_energy["kinetic"]),
    "final_magnetic_energy": float(final_energy["magnetic"]),
    "final_cross_helicity": float(final_energy["cross_helicity"]),
    "final_magnetic_divergence_linf": float(mhd3d.divergence_linf(final_state.b_hat, k)),
    "final_velocity_divergence_linf": float(mhd3d.divergence_linf(final_state.v_hat, k)),
}
config = {
    "equations": "mhd3d",
    "shape": list(SHAPE),
    "viscosity": VISCOSITY,
    "resistivity": RESISTIVITY,
    "dt": DT,
    "t_end": T_END,
    "save_every": SAVE_EVERY,
    "integrator": "etdrk4_stochastic",
}
result = MHD3DResult(
    trajectory=trajectory,
    shape=SHAPE,
    parameters=params,
    config=config,
    diagnostics=diagnostics,
    compile_seconds=compile_seconds,
    run_seconds=run_seconds,
    device_count=1,
)

result.print_summary()
result.save(output)

# --- 6. Plotting ----------------------------------------------------------
figure, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
axis.loglog(k_1d, magnetic_spectrum, "o-", color="teal", label="magnetic")
axis.loglog(k_1d, kinetic_spectrum, "s-", color="firebrick", label="kinetic")

fit_mask = (
    (k_1d >= K_FIT_MIN)
    & (k_1d <= K_FIT_MAX)
    & np.isfinite(magnetic_spectrum)
    & (magnetic_spectrum > 0.0)
)
if not np.any(fit_mask):
    raise RuntimeError("No positive magnetic-spectrum samples in the Kolmogorov fit range")

# Fit only the normalization of the fixed Kolmogorov exponent in the marked band.
log_scale = np.mean(
    np.log(magnetic_spectrum[fit_mask])
    + (5.0 / 3.0) * np.log(k_1d[fit_mask])
)
kolmogorov_scale = float(np.exp(log_scale))
k_fit = np.linspace(float(K_FIT_MIN), float(K_FIT_MAX), 100)
axis.loglog(
    k_fit,
    kolmogorov_scale * k_fit ** (-5.0 / 3.0),
    "k--",
    label=rf"Kolmogorov fit, $k^{{-5/3}}$ ({K_FIT_MIN} ≤ k ≤ {K_FIT_MAX})",
)
for boundary in (K_FIT_MIN, K_FIT_MAX):
    axis.axvline(boundary, color="black", linestyle=":", linewidth=1.2)

axis.set_title(f"Time-averaged spectra (t={t_min} to {t_max})")
axis.set_xlabel(r"wavenumber $k$")
axis.set_ylabel(r"energy density $E(k)$")
axis.legend(frameon=False)
figure.savefig(output / "spectrum.png", dpi=180)
plt.close(figure)

# Animate nine orthogonal slices through the saved turbulence trajectory.
# Rows are slice normals; columns are fixed locations at 25%, 50%, and 75%.
animation_frames = np.unique(
    np.linspace(0, chunks - 1, min(chunks, 80), dtype=int)
)
magnetic_magnitudes = []
for frame_index in animation_frames:
    b_hat = np.asarray(saved.b_hat[frame_index])
    magnetic_field = np.fft.irfftn(b_hat, s=SHAPE, axes=(-3, -2, -1)).real
    magnetic_magnitudes.append(np.sqrt(np.sum(magnetic_field**2, axis=0)))

slice_indices = tuple(
    int(round(fraction * (SHAPE[0] - 1))) for fraction in (0.25, 0.50, 0.75)
)
slice_percentages = ("25%", "50%", "75%")
slice_specs = (
    ("x", "y", "z", lambda field, index: field[index, :, :].T),
    ("y", "x", "z", lambda field, index: field[:, index, :].T),
    ("z", "x", "y", lambda field, index: field[:, :, index].T),
)
color_max = float(np.percentile(np.stack(magnetic_magnitudes), 99.5))
if not np.isfinite(color_max) or color_max <= 0.0:
    color_max = 1.0

figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
initial_field = magnetic_magnitudes[0]
images = []
for row, (normal, horizontal, vertical, extract) in enumerate(slice_specs):
    for column, (index, percentage) in enumerate(
        zip(slice_indices, slice_percentages, strict=True)
    ):
        axis = axes[row, column]
        image = axis.imshow(
            extract(initial_field, index), origin="lower", cmap="magma",
            vmin=0.0, vmax=color_max, aspect="equal",
        )
        images.append((image, extract, index))
        axis.set_title(f"{normal} = {percentage}")
        axis.set_xlabel(horizontal)
        axis.set_ylabel(vertical)

title = figure.suptitle("Magnetic-field magnitude: t = 0.0", fontsize=15)
figure.colorbar(image, ax=axes, shrink=0.82, label="B magnitude")

def update_slice_animation(frame_number: int):
    field = magnetic_magnitudes[frame_number]
    for image, extract, index in images:
        image.set_data(extract(field, index))
    title.set_text(
        f"Magnetic-field magnitude: t = {float(times[animation_frames[frame_number]]):.1f}"
    )
    return [image for image, _, _ in images]

slice_animation = FuncAnimation(
    figure, update_slice_animation, frames=len(animation_frames), interval=100, blit=False
)
static_slice_path = output / "slices.png"
if static_slice_path.exists():
    static_slice_path.unlink()
slice_animation.save(output / "slices.gif", writer=PillowWriter(fps=10), dpi=110)
plt.close(figure)

# Map the three unique periodic faces onto adjoining planes of an isometric cube.
coordinates = np.linspace(0.0, 1.0, SHAPE[0])
face_normalizer = Normalize(vmin=0.0, vmax=color_max)
face_colormap = plt.get_cmap("magma")

def draw_isometric_faces(axis, field):
    y, z = np.meshgrid(coordinates, coordinates, indexing="ij")
    x, z_for_y = np.meshgrid(coordinates, coordinates, indexing="ij")
    x_for_z, y_for_z = np.meshgrid(coordinates, coordinates, indexing="ij")
    face_surfaces = (
        (np.zeros_like(y), y, z, field[0, :, :]),
        (x, np.zeros_like(x), z_for_y, field[:, 0, :]),
        (x_for_z, y_for_z, np.zeros_like(x_for_z), field[:, :, 0]),
    )
    for x_face, y_face, z_face, values in face_surfaces:
        axis.plot_surface(
            x_face, y_face, z_face,
            facecolors=face_colormap(face_normalizer(values)),
            rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_zlim(0.0, 1.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=-25.0, azim=225.0)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")

figure = plt.figure(figsize=(7, 7), constrained_layout=True)
axis = figure.add_subplot(projection="3d")
draw_isometric_faces(axis, initial_field)
cube_title = figure.suptitle("Periodic boundary faces on an isometric cube: t = 0.0")
figure.colorbar(
    plt.cm.ScalarMappable(norm=face_normalizer, cmap=face_colormap),
    ax=axis, shrink=0.72, pad=0.08, label="B magnitude",
)
figure.text(0.5, 0.02, "Faces: x = 0 (y-z), y = 0 (x-z), z = 0 (x-y)", ha="center")

def update_cube_animation(frame_number: int):
    axis.clear()
    draw_isometric_faces(axis, magnetic_magnitudes[frame_number])
    cube_title.set_text(
        "Periodic boundary faces on an isometric cube: "
        f"t = {float(times[animation_frames[frame_number]]):.1f}"
    )
    return axis.collections

cube_animation = FuncAnimation(
    figure, update_cube_animation, frames=len(animation_frames), interval=100, blit=False
)
flat_face_path = output / "outer_faces.gif"
if flat_face_path.exists():
    flat_face_path.unlink()
cube_animation.save(output / "outer_faces_cube.gif", writer=PillowWriter(fps=10), dpi=110)
plt.close(figure)

print(f"Data and spectrum plot saved to: {output}")
