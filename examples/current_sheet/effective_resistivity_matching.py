"""Infer a 2D effective-resistivity field eta(x,y) from sparse probe time series.

Synthetic-data inverse problem (diffusion-coefficient field inversion):

* A Gaussian magnetic structure (initial flux ``psi0``) is seeded in a periodic
  box. Its current density ``j_z = -laplacian(psi)`` decays and spreads under the
  resistive diffusion term ``eta(x, y) * laplacian(psi)`` in the reduced-MHD RHS.
  The blob is large enough that its current flows across the whole anomalous
  region, so ``eta*j_z`` directly samples the resistivity structure everywhere
  in it.
* The truth effective-resistivity profile ``eta(x, y)`` is an H-shaped anomalous
  region built from three Gaussian bars (two vertical bars and a horizontal
  crossbar) centered on the structure. The forward solver generates sparse
  "probe" time series of ``psi``, ``j_z`` and ``eta*j_z`` (the non-ideal field
  ``E_non-ideal = eta j``) at fixed locations over a time window, and Gaussian
  noise is added. With ``PROBE_STYLE = "full"`` the probes are a uniform grid
  spanning the box (covering the whole H); ``"center"`` uses a tight cluster
  around the structure.
* The inverse problem: starting from uniform resistivity (anomalous region
  wiped), reverse-mode AD propagates error gradients back through the full RK4
  trajectory and Adam (optax) optimizes the H parameters so the solver
  reproduces the observed time series.

Writes to OUTDIR: ``loss_history.png``, ``probe_fits.png``, ``eta_fields.png``
(recovered H against the truth), ``evolution.gif`` (target vs uniform vs
optimized ``j_z``) and ``effective_resistivity_matching.npz``.

Because this repository ships no external PIC data, the reference is generated
with the same solver (self-consistency test). Set TARGET_NPZ to a file with
``times`` and ``probes`` arrays (probes has 3 channels: psi, j, eta*j) to use a
reference from disk instead.

Runtime: the defaults (64 x 64, 1600 RK4 steps per forward run, 120 Adam steps)
take several minutes on a CPU. For a quick check use NX = 32, T_END = 2,
OPT_STEPS = 20; to scale up raise NX, N_OBS and OPT_STEPS. Requires ``optax``.
Run from the repository root:

    python examples/current_sheet/effective_resistivity_matching.py
"""

import time
from functools import partial
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import current_density, poisson_bracket, stream_function
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import laplacian
from mhx.state import ReducedMHDState

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/examples/effective_resistivity_matching")
NX = 64  # grid points in x
NY = None  # grid points in y (None: same as NX)
LX = 6.283185307179586  # box length in x (2*pi)
LY = 6.283185307179586  # box length in y (2*pi)
ETA0 = 2.0e-2  # uniform background resistivity
NU = 1.0e-2  # uniform viscosity
BLOB_AMP = 0.6  # initial Gaussian flux amplitude
BLOB_SIGMA = 1.1  # initial Gaussian flux width
DT = 5.0e-3  # time step
T_END = 8.0  # end time of the observation window
N_OBS = 20  # observation intervals
PROBE_STYLE = "full"  # "full" = uniform probe grid across the box; "center" = tight cluster
N_FULL_PROBES = 10  # probes per axis for the "full" style
PROBE_EXTENT = 4  # probe grid half-extent in probe spacings for the "center" style
H_A = 2.5e-2  # truth H amplitude
H_D = 0.55  # truth H bar half-separation
H_SW = 0.16  # truth H bar width
H_SH = 0.70  # truth H bar half-length
H_SL = 0.65  # truth H crossbar half-length
H_ST = 0.16  # truth H crossbar thickness
NOISE = 1.0e-3  # relative observation noise (0 disables)
SEED = 0  # RNG seed for the observation noise
W_PROBE = 1.0  # weight of the probe-data term
REG = 1.0e-3  # width prior weight
OPT_STEPS = 120  # Adam iterations
LR = 1.0e-2  # Adam learning rate
TARGET_NPZ = None  # optional NPZ with times/probes reference data instead of synthetic truth

# Enable double precision for the inverse-problem accuracy.
jax.config.update("jax_enable_x64", True)

# H-shaped anomalous-region parameters:
#   A   total amplitude scaling the three bars
#   dx, dy  center offset from the structure center
#   d   half separation of the two vertical bars
#   sw  horizontal width of the vertical bars
#   sh  vertical half-length of the vertical bars
#   sL  horizontal half-length of the crossbar
#   st  vertical thickness of the crossbar
H_PARAM_NAMES = ("A", "dx", "dy", "d", "sw", "sh", "sL", "st")


# ---------------------------------------------------------------------------
# Forward model with a spatially varying resistivity
# ---------------------------------------------------------------------------


def gaussian_blob_state(
    grid: CartesianGrid, amplitude: float, sigma: float, center: tuple[float, float]
) -> ReducedMHDState:
    """Return a localized Gaussian flux structure with zero initial flow."""
    x, y = grid.mesh()
    radius2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
    psi = amplitude * jnp.exp(-radius2 / (2.0 * sigma**2))
    return ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))


def field_eta_rhs(
    state: ReducedMHDState,
    eta_field: jnp.ndarray,
    viscosity: float,
    lengths: tuple[float, float],
) -> ReducedMHDState:
    r"""Reduced-MHD RHS with a 2D resistivity field.

    ``psi_t + [phi, psi] = eta(x, y) nabla^2 psi`` and
    ``omega_t + [phi, omega] = [psi, nabla^2 psi] + nu nabla^2 omega``.
    """
    phi = stream_function(state.omega, lengths=lengths)
    lap_psi = laplacian(state.psi, lengths=lengths)
    lap_omega = laplacian(state.omega, lengths=lengths)
    dpsi = -poisson_bracket(phi, state.psi, lengths=lengths) + eta_field * lap_psi
    domega = (
        -poisson_bracket(phi, state.omega, lengths=lengths)
        + poisson_bracket(state.psi, lap_psi, lengths=lengths)
        + viscosity * lap_omega
    )
    return ReducedMHDState(psi=dpsi, omega=domega)


@partial(jax.checkpoint, static_argnums=(3, 4, 5))
def advance_block(
    state: ReducedMHDState,
    eta_field: jnp.ndarray,
    viscosity: float,
    lengths: tuple[float, float],
    dt: float,
    steps: int,
) -> ReducedMHDState:
    """Advance ``steps`` explicit RK4 steps with a fixed ``eta_field``."""

    def step_fn(carry: ReducedMHDState, _: object) -> tuple[ReducedMHDState, None]:
        s = carry
        k1 = field_eta_rhs(s, eta_field, viscosity, lengths)
        s2 = ReducedMHDState(psi=s.psi + 0.5 * dt * k1.psi, omega=s.omega + 0.5 * dt * k1.omega)
        k2 = field_eta_rhs(s2, eta_field, viscosity, lengths)
        s3 = ReducedMHDState(psi=s.psi + 0.5 * dt * k2.psi, omega=s.omega + 0.5 * dt * k2.omega)
        k3 = field_eta_rhs(s3, eta_field, viscosity, lengths)
        s4 = ReducedMHDState(psi=s.psi + dt * k3.psi, omega=s.omega + dt * k3.omega)
        k4 = field_eta_rhs(s4, eta_field, viscosity, lengths)
        psi_next = s.psi + (dt / 6.0) * (k1.psi + 2.0 * k2.psi + 2.0 * k3.psi + k4.psi)
        omega_next = s.omega + (dt / 6.0) * (k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega)
        return ReducedMHDState(psi=psi_next, omega=omega_next), None

    final, _ = jax.lax.scan(step_fn, state, None, length=steps)
    return final


def record_states(
    eta_field: jnp.ndarray,
    viscosity: float,
    lengths: tuple[float, float],
    dt: float,
    steps_per_obs: int,
    num_obs: int,
    psi0: jnp.ndarray,
    omega0: jnp.ndarray,
) -> list[ReducedMHDState]:
    """Advance the solver and return states at every observation time (incl. t=0)."""
    state = ReducedMHDState(psi=psi0, omega=omega0)
    recorded = [state]
    for _ in range(num_obs):
        state = advance_block(state, eta_field, viscosity, lengths, dt, steps_per_obs)
        recorded.append(state)
    return recorded


def simulate_observables(
    eta_field: jnp.ndarray,
    viscosity: float,
    lengths: tuple[float, float],
    dt: float,
    steps_per_obs: int,
    num_obs: int,
    probe_xy: jnp.ndarray,
    psi0: jnp.ndarray,
    omega0: jnp.ndarray,
) -> jnp.ndarray:
    """Return probe readings ``(num_obs + 1, K, 3)`` with channels [psi, j, eta*j]."""
    recorded = record_states(
        eta_field, viscosity, lengths, dt, steps_per_obs, num_obs, psi0, omega0
    )
    psi_stack = jnp.stack([s.psi for s in recorded])
    j_stack = jnp.stack([current_density(s.psi, lengths=lengths) for s in recorded])
    psi_probes = jax.vmap(lambda f: bilinear_sample(f, probe_xy, lengths))(psi_stack)
    j_probes = jax.vmap(lambda f: bilinear_sample(f, probe_xy, lengths))(j_stack)
    eta_probes = bilinear_sample(eta_field, probe_xy, lengths)
    eta_j_probes = eta_probes[None, :] * j_probes
    return jnp.stack([psi_probes, j_probes, eta_j_probes], axis=-1)


def simulate_fields(
    eta_field: jnp.ndarray,
    viscosity: float,
    lengths: tuple[float, float],
    dt: float,
    steps_per_obs: int,
    num_obs: int,
    psi0: jnp.ndarray,
    omega0: jnp.ndarray,
) -> jnp.ndarray:
    """Return the ``psi`` stack at observation times (for post-processing)."""
    recorded = record_states(
        eta_field, viscosity, lengths, dt, steps_per_obs, num_obs, psi0, omega0
    )
    return jnp.stack([s.psi for s in recorded])


# ---------------------------------------------------------------------------
# H-shaped anomalous-resistivity region
# ---------------------------------------------------------------------------


def build_eta_field(
    theta: jnp.ndarray,
    mesh: tuple[jnp.ndarray, jnp.ndarray],
    lengths: tuple[float, float],
    det_center: tuple[float, float],
    eta0: float,
    sigma_min: float,
) -> jnp.ndarray:
    r"""Return ``eta0 + A * H(x, y)`` with ``H`` built from three Gaussian bars.

    ``H = exp(-(xr^2 / 2 sw^2 + (y-cy)^2 / 2 sh^2))`` for each of the two
    vertical bars (centered at ``cx +/- d``) plus the horizontal crossbar
    ``exp(-((x-cx)^2 / 2 sL^2 + (y-cy)^2 / 2 st^2))``.  ``A`` is clipped
    non-negative, the geometry widths are floored at ``sigma_min``, and center
    offsets are clamped.  All raw parameters have O(1) gradients near the
    wiped start.
    """
    amplitude = jnp.clip(theta[0], min=0.0)
    dx = jnp.clip(theta[1], -lengths[0] / 4.0, lengths[0] / 4.0)
    dy = jnp.clip(theta[2], -lengths[1] / 4.0, lengths[1] / 4.0)
    d = jnp.clip(theta[3], min=sigma_min)
    sw = jnp.clip(theta[4], min=sigma_min)
    sh = jnp.clip(theta[5], min=sigma_min)
    sL = jnp.clip(theta[6], min=sigma_min)
    st = jnp.clip(theta[7], min=sigma_min)

    cx = det_center[0] + dx
    cy = det_center[1] + dy
    x, y = mesh

    def vertical_bar(center_x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-(((x - center_x) ** 2) / (2.0 * sw**2) + ((y - cy) ** 2) / (2.0 * sh**2)))

    crossbar = jnp.exp(-(((x - cx) ** 2) / (2.0 * sL**2) + ((y - cy) ** 2) / (2.0 * st**2)))
    h_shape = vertical_bar(cx - d) + vertical_bar(cx + d) + crossbar
    return eta0 + amplitude * h_shape


# ---------------------------------------------------------------------------
# Observation operator and cost
# ---------------------------------------------------------------------------


def bilinear_sample(
    field: jnp.ndarray, points: jnp.ndarray, lengths: tuple[float, float]
) -> jnp.ndarray:
    """Bilinearly interpolate ``field`` at ``points`` with periodic wrap."""
    nx, ny = field.shape
    fx = points[:, 0] / lengths[0] * nx - 0.5
    fy = points[:, 1] / lengths[1] * ny - 0.5
    x0 = jnp.floor(fx).astype(jnp.int32)
    y0 = jnp.floor(fy).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    tx = fx - x0
    ty = fy - y0
    x0 = x0 % nx
    x1 = x1 % nx
    y0 = y0 % ny
    y1 = y1 % ny
    f00 = field[x0, y0]
    f10 = field[x1, y0]
    f01 = field[x0, y1]
    f11 = field[x1, y1]
    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def make_probe_grid(
    center: tuple[float, float], grid: CartesianGrid, extent: int = 4, spacing: float = 2.0
) -> jnp.ndarray:
    """Return a ``(2*extent + 1)^2`` probe grid around ``center``."""
    dx, dy = grid.spacing
    offsets_x = jnp.arange(-extent, extent + 1) * spacing * dx
    offsets_y = jnp.arange(-extent, extent + 1) * spacing * dy
    ox, oy = jnp.meshgrid(offsets_x, offsets_y, indexing="ij")
    return jnp.stack([center[0] + ox.ravel(), center[1] + oy.ravel()], axis=-1)


def make_full_probe_grid(grid: CartesianGrid, n: int = 10) -> jnp.ndarray:
    """Return an ``n x n`` probe grid spanning most of the periodic box."""
    fraction = jnp.linspace(0.08, 0.92, n)
    ox, oy = jnp.meshgrid(fraction, fraction, indexing="ij")
    return jnp.stack(
        [ox.ravel() * grid.lengths[0], oy.ravel() * grid.lengths[1]], axis=-1
    )


def loss_fn(
    theta: jnp.ndarray,
    mesh: tuple[jnp.ndarray, jnp.ndarray],
    lengths: tuple[float, float],
    det_center: tuple[float, float],
    eta0: float,
    sigma_min: float,
    viscosity: float,
    dt: float,
    steps_per_obs: int,
    num_obs: int,
    probe_xy: jnp.ndarray,
    psi0: jnp.ndarray,
    omega0: jnp.ndarray,
    probes_obs: jnp.ndarray,
    probes_sigma: jnp.ndarray,
    w_probe: float,
    reg_lambda: float,
) -> jnp.ndarray:
    """Weighted mismatch of the sparse probe time series plus a width prior."""
    eta_field = build_eta_field(theta, mesh, lengths, det_center, eta0, sigma_min)
    probes_sim = simulate_observables(
        eta_field, viscosity, lengths, dt, steps_per_obs, num_obs, probe_xy, psi0, omega0
    )
    diff = (probes_sim - probes_obs) / probes_sigma[None, None, :]
    data_term = jnp.mean(diff**2)
    widths = jnp.stack([jnp.clip(theta[i], min=sigma_min) for i in (4, 5, 6, 7)])
    reg = reg_lambda * jnp.sum((widths - 0.3) ** 2 / 0.09)
    return w_probe * data_term + reg


# ---------------------------------------------------------------------------
# Setup and verification helpers
# ---------------------------------------------------------------------------


def add_noise(
    probes_clean: jnp.ndarray, noise_rel: float, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Add per-channel Gaussian noise and return ``(obs, clean, probes_sigma)``."""
    if noise_rel <= 0.0:
        return probes_clean, probes_clean, jnp.ones((3,), dtype=jnp.float64)
    probes_sigma = jnp.maximum(jnp.std(probes_clean, axis=(0, 1)), 1.0e-12)
    key = jax.random.PRNGKey(seed)
    noise = probes_sigma[None, None, :] * jax.random.normal(key, probes_clean.shape)
    return probes_clean + noise, probes_clean, probes_sigma


def truth_theta(
    a: float, d: float, sw: float, sh: float, sL: float, st: float
) -> jnp.ndarray:
    """Return the truth H parameter vector, centered on the structure."""
    return jnp.asarray([a, 0.0, 0.0, d, sw, sh, sL, st], dtype=jnp.float64)


def initial_theta() -> jnp.ndarray:
    """Wiped start: nearly zero amplitude, centered, generic geometry."""
    return jnp.asarray([1.0e-4, 0.0, 0.0, 0.6, 0.3, 0.3, 0.3, 0.3], dtype=jnp.float64)


def decode_params(theta: jnp.ndarray, det_center: tuple[float, float],
                  lengths: tuple[float, float]) -> dict[str, float]:
    """Return physical H parameters for reporting."""
    return {
        "A": float(jnp.clip(theta[0], min=0.0)),
        "cx": det_center[0] + float(jnp.clip(theta[1], -lengths[0] / 4.0, lengths[0] / 4.0)),
        "cy": det_center[1] + float(jnp.clip(theta[2], -lengths[1] / 4.0, lengths[1] / 4.0)),
        "d": float(jnp.clip(theta[3], min=0.0)),
        "sw": float(jnp.clip(theta[4], min=0.0)),
        "sh": float(jnp.clip(theta[5], min=0.0)),
        "sL": float(jnp.clip(theta[6], min=0.0)),
        "st": float(jnp.clip(theta[7], min=0.0)),
    }


def render_evolution_gif(
    jz_target: np.ndarray,
    jz_uniform: np.ndarray,
    jz_optimized: np.ndarray,
    times: np.ndarray,
    lengths: tuple[float, float],
    path: Path,
    max_frames: int = 16,
) -> Path:
    """Render a target / uniform / optimized ``j_z`` evolution GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = jz_target.shape[0]
    indices = np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))
    vmax = float(
        max(np.max(np.abs(jz_target)), np.max(np.abs(jz_uniform)), np.max(np.abs(jz_optimized)))
    )
    vmax = max(vmax, np.finfo(np.float64).tiny)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    extent = (0.0, lengths[0], 0.0, lengths[1])
    images = []
    for axis, (label, stack) in zip(
        axes.flat,
        (
            ("target", jz_target),
            ("uniform eta", jz_uniform),
            ("optimized eta", jz_optimized),
        ),
        strict=True,
    ):
        im = axis.imshow(
            stack[0].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
        )
        axis.set_title(label)
        images.append(im)
    fig.colorbar(images[0], ax=axes, shrink=0.85, label=r"$j_z$")

    frames = []
    for index in indices:
        for im, stack in zip(images, (jz_target, jz_uniform, jz_optimized), strict=True):
            im.set_data(stack[index].T)
        fig.suptitle(f"t = {times[index]:.2f}")
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            (height, width, 4)
        )
        frames.append(buffer.copy())
    plt.close(fig)
    imageio.mimsave(path, frames, duration=140, loop=0)
    return path



# --- Run -------------------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)

ny = NX if NY is None else NY
grid = CartesianGrid.from_mesh_config(
    MeshConfig(shape=(NX, ny), lower=(0.0, 0.0), upper=(LX, LY))
)
mesh = grid.mesh()
lengths = grid.lengths
dx, dy = grid.spacing
sigma_min = 0.5 * min(dx, dy)
steps_per_obs = max(1, int(round((T_END / N_OBS) / DT)))
num_obs = N_OBS
times = DT * steps_per_obs * np.arange(num_obs + 1)
det_center = (0.5 * LX, 0.5 * LY)
if PROBE_STYLE == "full":
    probe_xy = make_full_probe_grid(grid, N_FULL_PROBES)
else:
    probe_xy = make_probe_grid(det_center, grid, extent=PROBE_EXTENT)
center_probe = int(jnp.argmin(jnp.sum((probe_xy - jnp.asarray(det_center)) ** 2, axis=-1)))

initial_state = gaussian_blob_state(grid, BLOB_AMP, BLOB_SIGMA, det_center)
psi0 = initial_state.psi
omega0 = initial_state.omega

print("=== Effective resistivity eta(x,y) inversion (H-shaped region) ===")
print(f"grid {grid.shape}, structure center ({det_center[0]:.4f}, {det_center[1]:.4f}), "
      f"{probe_xy.shape[0]} probes, {num_obs} obs intervals, "
      f"{steps_per_obs} RK4 steps each, actual t_end = {times[-1]:.3f}", flush=True)

sim_jit = jax.jit(
    partial(
        simulate_observables,
        viscosity=NU,
        lengths=lengths,
        dt=DT,
        steps_per_obs=steps_per_obs,
        num_obs=num_obs,
        probe_xy=probe_xy,
        psi0=psi0,
        omega0=omega0,
    )
)

# 1. Reference observables (synthetic "kinetic-like" data).
theta_true = truth_theta(H_A, H_D, H_SW, H_SH, H_SL, H_ST)
theta_uniform = initial_theta().at[0].set(0.0)
theta0 = initial_theta()

t0 = time.perf_counter()
if TARGET_NPZ is not None:
    reference = np.load(TARGET_NPZ)
    probes_obs = jnp.asarray(reference["probes"], dtype=jnp.float64)
    probes_sigma = jnp.maximum(jnp.std(probes_obs, axis=(0, 1)), 1.0e-12)
    print(f"loaded reference from {TARGET_NPZ}")
else:
    eta_true_field = build_eta_field(theta_true, mesh, lengths, det_center, ETA0, sigma_min)
    probes_clean = jax.block_until_ready(sim_jit(eta_true_field))
    probes_obs, probes_clean, probes_sigma = add_noise(probes_clean, NOISE, SEED)
    print(f"reference observations ({time.perf_counter() - t0:.1f} s incl. compile): "
          f"probes shape {tuple(probes_obs.shape)}, "
          f"per-channel sigma = {np.asarray(probes_sigma)}", flush=True)

# 2. Differentiable loss and its exact gradient (reverse-mode AD through every
#    RK4 step of the whole observation window).
loss_val_grad = jax.jit(
    jax.value_and_grad(
        partial(
            loss_fn,
            mesh=mesh,
            lengths=lengths,
            det_center=det_center,
            eta0=ETA0,
            sigma_min=sigma_min,
            viscosity=NU,
            dt=DT,
            steps_per_obs=steps_per_obs,
            num_obs=num_obs,
            probe_xy=probe_xy,
            psi0=psi0,
            omega0=omega0,
            probes_obs=probes_obs,
            probes_sigma=probes_sigma,
            w_probe=W_PROBE,
            reg_lambda=REG,
        )
    )
)

t0 = time.perf_counter()
initial_loss, _ = jax.block_until_ready(loss_val_grad(theta0))
print(f"loss with wiped anomalous region (uniform eta): {float(initial_loss):.6e} "
      f"(compile + first gradient {time.perf_counter() - t0:.1f} s)", flush=True)

# 3. Adam optimization of the H parameters.
optimizer = optax.adam(learning_rate=LR)
opt_state = optimizer.init(theta0)
theta = theta0
loss_history = []
t0 = time.perf_counter()
for step in range(1, OPT_STEPS + 1):
    loss, grads = loss_val_grad(theta)
    updates, opt_state = optimizer.update(grads, opt_state)
    theta = optax.apply_updates(theta, updates)
    loss_history.append(float(loss))
    if step == 1 or step % 10 == 0 or step == OPT_STEPS:
        p = decode_params(theta, det_center, lengths)
        print(f"step {step:03d} | loss {loss:.6e} | A={p['A']:.2e} d={p['d']:.3f} "
              f"sw={p['sw']:.3f} sh={p['sh']:.3f} sL={p['sL']:.3f} st={p['st']:.3f} "
              f"| {time.perf_counter() - t0:.1f} s", flush=True)

# 4. Verification: recovered H vs truth and probe fit.
eta_true_field = build_eta_field(theta_true, mesh, lengths, det_center, ETA0, sigma_min)
eta_opt_field = build_eta_field(theta, mesh, lengths, det_center, ETA0, sigma_min)
eta_uniform_field = build_eta_field(theta_uniform, mesh, lengths, det_center, ETA0, sigma_min)
probes_opt = jnp.asarray(jax.block_until_ready(sim_jit(eta_opt_field)))
probes_uniform = jnp.asarray(jax.block_until_ready(sim_jit(eta_uniform_field)))

# --- Results ---------------------------------------------------------------
bump_true = np.asarray(eta_true_field) - ETA0
bump_opt = np.asarray(eta_opt_field) - ETA0
correlation = float(np.corrcoef(bump_true.ravel(), bump_opt.ravel())[0, 1])
bump_rel_l2 = float(np.linalg.norm(bump_opt - bump_true) / max(np.linalg.norm(bump_true), 1.0e-12))

p_true = decode_params(theta_true, det_center, lengths)
p_opt = decode_params(theta, det_center, lengths)
print("\n--- verification ---")
print(f"H correlation (eta - eta0): {correlation:.4f}, relative L2 error: {bump_rel_l2:.3e}")
for key, value in p_true.items():
    print(f"  {key:>4s}: true={value:+.4e}  recovered={p_opt[key]:+.4e}")
for name, probes in (("target", probes_obs), ("uniform", probes_uniform),
                     ("optimized", probes_opt)):
    center_j = float(np.asarray(probes)[-1, center_probe, 1])
    print(f"j_z at center probe, final time: {name} = {center_j:.4e}")

fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
ax_loss.semilogy(loss_history)
ax_loss.set_xlabel("Adam step")
ax_loss.set_ylabel("loss")
ax_loss.set_title("Effective resistivity inversion: loss history")
ax_loss.grid(True, which="both", ls="--", alpha=0.5)
fig_loss.savefig(OUTDIR / "loss_history.png", dpi=150, bbox_inches="tight")
plt.close(fig_loss)

# The 4 probes where optimized eta differs most from uniform eta in j_z (channel 1).
probe_diff_rms = jnp.sqrt(jnp.mean((probes_opt[..., 1] - probes_uniform[..., 1]) ** 2, axis=0))
probe_indices = list(np.asarray(jnp.argsort(-probe_diff_rms))[:4])
print(f"top-4 probes by |j_z(optimized) - j_z(uniform)|: {probe_indices}")
fig_probes, axes_probes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for axis, probe in zip(axes_probes.flat, probe_indices, strict=False):
    axis.plot(times, np.asarray(probes_obs)[:, probe, 1], "ko", ms=4, label="target (obs)")
    axis.plot(times, np.asarray(probes_uniform)[:, probe, 1], "b--", label="uniform eta")
    axis.plot(times, np.asarray(probes_opt)[:, probe, 1], "r-", label="optimized eta")
    axis.set_title(f"j_z at probe {probe}")
    axis.set_xlabel("t")
    axis.legend(frameon=False, fontsize="small")
fig_probes.suptitle("Current-density probe time series")
fig_probes.savefig(OUTDIR / "probe_fits.png", dpi=150, bbox_inches="tight")
plt.close(fig_probes)

vmax_eta = max(float(np.max(np.asarray(eta_true_field))), float(np.max(np.asarray(eta_opt_field))))
fig_eta, axes_eta = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
extent = (0.0, LX, 0.0, LY)
axes_eta[0].imshow(np.asarray(eta_true_field).T, origin="lower", cmap="viridis",
                   vmin=ETA0, vmax=vmax_eta, extent=extent)
axes_eta[0].set_title("truth eta(x,y): H-shaped anomaly")
im = axes_eta[1].imshow(np.asarray(eta_opt_field).T, origin="lower", cmap="viridis",
                        vmin=ETA0, vmax=vmax_eta, extent=extent)
axes_eta[1].set_title(f"recovered eta(x,y)  (corr {correlation:.3f})")
fig_eta.colorbar(im, ax=axes_eta, shrink=0.85, label=r"$\eta$")
fig_eta.savefig(OUTDIR / "eta_fields.png", dpi=150, bbox_inches="tight")
plt.close(fig_eta)

# Evolution GIF: j_z over time, target vs uniform vs optimized.
jz_stacks = []
for eta_field in (eta_true_field, eta_uniform_field, eta_opt_field):
    psi_stack = np.asarray(
        simulate_fields(eta_field, NU, lengths, DT, steps_per_obs, num_obs, psi0, omega0)
    )
    jz_stacks.append(
        np.asarray([np.asarray(current_density(frame, lengths=lengths)) for frame in psi_stack])
    )
jz_target, jz_uniform, jz_optimized = jz_stacks
gif_path = render_evolution_gif(
    jz_target, jz_uniform, jz_optimized, times, lengths, OUTDIR / "evolution.gif"
)

npz_path = OUTDIR / "effective_resistivity_matching.npz"
np.savez_compressed(
    npz_path,
    times=times,
    theta_true=np.asarray(theta_true),
    theta_recovered=np.asarray(theta),
    loss_history=np.asarray(loss_history),
    probes_obs=np.asarray(probes_obs),
    probes_uniform=np.asarray(probes_uniform),
    probes_optimized=np.asarray(probes_opt),
    eta_true=np.asarray(eta_true_field),
    eta_optimized=np.asarray(eta_opt_field),
    jz_target=jz_target,
    jz_uniform=jz_uniform,
    jz_optimized=jz_optimized,
)
print(f"\nSaved {OUTDIR / 'loss_history.png'}, {OUTDIR / 'probe_fits.png'}, "
      f"{OUTDIR / 'eta_fields.png'}")
print(f"Saved {gif_path}")
print(f"Saved {npz_path}")
