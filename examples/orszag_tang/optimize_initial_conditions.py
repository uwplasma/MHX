"""Recover Orszag--Tang initial-condition parameters by reverse-mode autodiff.

Recovers the four parameters (p0, p1, p2, p3) of a modified Orszag--Tang initial
condition by differentiating the reduced-MHD evolution end to end (checkpointed
RK4) and minimizing the mean-squared distance to a "true" synthetic target state
with L-BFGS (optax). It writes the loss curve, a GIF comparing the target with a
baseline [1,1,1,1] run, and a GIF of the optimization. Optional one-dimensional
scans (SCAN_ETA, SCAN_P1 ... SCAN_P4) sweep each parameter around its true value
and save loss versus parameter. The default 64 x 64 run takes a few minutes on a
laptop CPU; set RESOLUTION = 32 for a quick look, and raise OPT_STEPS, the
resolution and T_END to scale up (reverse mode stores the tape, so memory grows
with the number of steps). Needs optax. Run from the repository root:

    python examples/orszag_tang/optimize_initial_conditions.py
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
from jaxtyping import Array

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import advance_rk4

# Enable double precision for stability
jax.config.update("jax_enable_x64", True)

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/examples/orszag_tang_ic_opt")
P0 = 1.5  # true value of parameter p0
P1 = 0.8  # true value of parameter p1
P2 = 1.2  # true value of parameter p2
P3 = 0.5  # true value of parameter p3
NX = 64  # grid points in x
NY = 64  # grid points in y
RESOLUTION = None  # if set, overrides both NX and NY
DOMAIN_SIZE = 10.0  # box side
ETA = 1.0e-3  # magnetic resistivity
NU = 1.0e-3  # kinematic viscosity
DT = 0.01  # time step
T_END = 1.0  # end time of the simulation
OPT_STEPS = 20  # number of L-BFGS iterations
LBFGS_MEMORY_SIZE = 10  # L-BFGS history size
# Optional parameter scans (loss vs parameter around the true values)
SCAN_ETA = False  # sweep eta
SCAN_P1 = False  # sweep p0 (theta[0], labelled p1)
SCAN_P2 = False  # sweep p1 (theta[1], labelled p2)
SCAN_P3 = False  # sweep p2 (theta[2], labelled p3)
SCAN_P4 = False  # sweep p3 (theta[3], labelled p4)
SCAN_POINTS = 21  # points per parameter scan
SCAN_RELATIVE_RANGE = 0.5  # scan range as a fraction around the true value

# --- Helpers ---------------------------------------------------------------


def modified_orszag_tang_initial_state(
    theta: Array, grid: CartesianGrid
) -> ReducedMHDState:
    x, y = grid.mesh()
    Lx, Ly = grid.lengths

    kx = 2.0 * jnp.pi / Lx
    ky = 2.0 * jnp.pi / Ly

    p0, p1, p2, p3 = theta

    psi = 0.2 * (p0 * (1.0 / ky) * jnp.cos(ky * y) + p1 * (0.5 / kx) * jnp.cos(2.0 * kx * x))
    omega = 0.2 * (p2 * kx * jnp.cos(kx * x) + p3 * ky * jnp.cos(ky * y))

    return ReducedMHDState(psi=psi, omega=omega)


@partial(jax.jit, static_argnums=(2, 3, 4))
def run_block(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    grid_lengths: tuple[float, float],
    dt: float,
    steps: int,
) -> ReducedMHDState:
    """Run ``steps`` RK4 steps, checkpointed to bound reverse-mode memory."""

    def rhs(active_state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(active_state, params, lengths=grid_lengths)

    return advance_rk4(state, rhs, dt=dt, steps=steps, checkpoint=True)


def evolve_system(
    theta: Array,
    grid: CartesianGrid,
    params: ReducedMHDParams,
    dt: float,
    steps_per_block: int,
    num_blocks: int,
) -> ReducedMHDState:
    state = modified_orszag_tang_initial_state(theta, grid)
    for _ in range(num_blocks):
        state = run_block(state, params, grid.lengths, dt, steps_per_block)
    return state


def simulate_and_track_history(
    theta: Array,
    grid: CartesianGrid,
    params: ReducedMHDParams,
    dt: float,
    steps_per_block: int,
    num_blocks: int,
) -> tuple[ReducedMHDState, list[np.ndarray]]:
    """Evolve the system and capture the state at each checkpoint block."""
    state = modified_orszag_tang_initial_state(theta, grid)
    history = [np.array(state.omega)]

    for _ in range(num_blocks):
        state = run_block(state, params, grid.lengths, dt, steps_per_block)
        history.append(np.array(state.omega))

    return state, history


def loss_fn(
    theta: Array,
    target_state: ReducedMHDState,
    grid: CartesianGrid,
    params: ReducedMHDParams,
    dt: float,
    steps_per_block: int,
    num_blocks: int,
) -> tuple[float, ReducedMHDState]:
    """Return the loss and the current evolved state as auxiliary data."""
    current_state = evolve_system(theta, grid, params, dt, steps_per_block, num_blocks)

    loss_psi = jnp.mean((current_state.psi - target_state.psi) ** 2)
    loss_omega = jnp.mean((current_state.omega - target_state.omega) ** 2)

    return loss_psi + loss_omega, current_state


def run_parameter_scan(
    name: str,
    theta_index: int | None,
    true_value: float,
    grid: CartesianGrid,
    params: ReducedMHDParams,
    target_state: ReducedMHDState,
    true_theta: Array,
    dt: float,
    steps_per_block: int,
    num_blocks: int,
    num_points: int,
    relative_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep one parameter around its true value and return (values, losses).

    For a theta parameter pass its index into ``true_theta`` (0..3); pass
    ``theta_index=None`` to sweep eta instead. All other parameters are held
    at their true values, and the loss is measured against ``target_state``.
    """
    lo = true_value * (1.0 - relative_range)
    hi = true_value * (1.0 + relative_range)
    values = np.linspace(lo, hi, num_points)

    losses = []
    for v in values:
        if theta_index is None:
            scan_params = ReducedMHDParams(resistivity=float(v), viscosity=params.viscosity)
            theta = true_theta
        else:
            scan_params = params
            theta = true_theta.at[theta_index].set(jnp.asarray(v))

        state = evolve_system(theta, grid, scan_params, dt, steps_per_block, num_blocks)
        loss_psi = jnp.mean((state.psi - target_state.psi) ** 2)
        loss_omega = jnp.mean((state.omega - target_state.omega) ** 2)
        losses.append(float(loss_psi + loss_omega))

    return values, np.asarray(losses)


def print_scan_results(name: str, values: np.ndarray, losses: np.ndarray, true_value: float):
    """Print a scan table and the parameter value attaining the minimum loss."""
    i_min = int(np.argmin(losses))
    print(f"\n--- Scan: {name} vs Loss ---")
    print(f"{'value':>12s} {'loss':>14s}")
    for v, loss_val in zip(values, losses, strict=False):
        print(f"{v:12.6g} {loss_val:14.6e}")
    print(f"Minimum loss at {name} = {values[i_min]:.6g} (true value {true_value:.6g})")


def plot_scan(name: str, values: np.ndarray, losses: np.ndarray, outdir: Path):
    """Plot loss vs parameter value and save to ``scan_<name>_vs_loss.png``."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, losses, "o-", linewidth=2)
    ax.set_xlabel(name)
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Parameter Scan: {name} vs Loss")
    ax.grid(True, ls="--", alpha=0.7)
    if float(np.min(losses)) > 0.0:
        ax.set_yscale("log")
    path = outdir / f"scan_{name}_vs_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scan plot to {path}")


def render_optimization_gif(
    target_omega: np.ndarray,
    history_omegas: list[np.ndarray],
    domain_size: float,
    path: Path,
):
    """Render a side-by-side GIF comparing the true state vs the optimizing state."""
    print("\nRendering optimization process GIF... (this might take a minute)")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    vmax = max(float(np.max(np.abs(target_omega))), np.finfo(float).eps)

    im_true = axes[0].imshow(
        target_omega.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(0.0, domain_size, 0.0, domain_size),
    )
    axes[0].set_title("True Evolved State (Target)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im_opt = axes[1].imshow(
        history_omegas[0].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(0.0, domain_size, 0.0, domain_size),
    )
    axes[1].set_title("Optimization Step 1")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.colorbar(im_true, ax=axes, shrink=0.75, label=r"Vorticity ($\omega$)")

    frames = []
    for i, omega_frame in enumerate(history_omegas):
        im_opt.set_data(omega_frame.T)
        axes[1].set_title(f"Optimization Step {i+1}")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(buf.copy())

    plt.close(fig)
    imageio.mimsave(path, frames, fps=10, loop=0)
    print(f"Saved optimization visualization GIF to {path}")


def render_time_evolution_gif(
    target_history: list[np.ndarray],
    default_history: list[np.ndarray],
    domain_size: float,
    path: Path,
):
    """Render a side-by-side GIF comparing the true vs default [1,1,1,1] evolution."""
    print("Rendering time evolution comparison GIF... (this might take a minute)")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    vmax = max(float(np.max(np.abs(target_history[-1]))), np.finfo(float).eps)

    im_target = axes[0].imshow(
        target_history[0].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(0.0, domain_size, 0.0, domain_size),
    )
    axes[0].set_title("True Simulation (Target)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im_default = axes[1].imshow(
        default_history[0].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(0.0, domain_size, 0.0, domain_size),
    )
    axes[1].set_title("Baseline Simulation [1,1,1,1]")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.colorbar(im_target, ax=axes, shrink=0.75, label=r"Vorticity ($\omega$)")

    frames = []
    num_frames = len(target_history)
    for i in range(num_frames):
        im_target.set_data(target_history[i].T)
        im_default.set_data(default_history[i].T)

        progress = (i / max(1, num_frames - 1)) * 100
        axes[0].set_title(f"True Simulation Target (Time: {progress:.0f}%)")
        axes[1].set_title(f"Baseline Simulation (Time: {progress:.0f}%)")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(buf.copy())

    plt.close(fig)
    imageio.mimsave(path, frames, fps=15, loop=0)
    print(f"Saved time evolution comparison GIF to {path}")


# --- Run -------------------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
nx, ny = (NX, NY) if RESOLUTION is None else (RESOLUTION, RESOLUTION)

grid = CartesianGrid.from_mesh_config(
    MeshConfig(
        shape=(nx, ny),
        lower=(0.0, 0.0),
        upper=(DOMAIN_SIZE, DOMAIN_SIZE),
    )
)

steps_per_block = max(1, int(0.1 / DT))
num_blocks = int(round(T_END / (DT * steps_per_block)))

params = ReducedMHDParams(resistivity=ETA, viscosity=NU)
true_theta = jnp.array([P0, P1, P2, P3])

print("=== Initial-condition optimization in MHX ===")
print(f"grid {nx}x{ny}, domain {DOMAIN_SIZE:g}, eta={ETA:g}, nu={NU:g}, dt={DT:g}, "
      f"t_end={T_END:g} ({num_blocks} blocks of {steps_per_block} steps)")
print(f"true theta = {[P0, P1, P2, P3]}, {OPT_STEPS} L-BFGS steps; devices: {jax.devices()}",
      flush=True)

initial_state = modified_orszag_tang_initial_state(true_theta, grid)
compile_start = time.perf_counter()
run_block.lower(
    initial_state, params, grid.lengths, DT, steps_per_block
).compile()
integration_compile_seconds = time.perf_counter() - compile_start
integration_run_seconds = 0.0

# 1. Generate the "true" target state and the baseline state history.
print("\nSimulating target state and baseline [1,1,1,1] state (this may take a "
      "moment to compile)...")
integration_start = time.perf_counter()
target_state, target_history = simulate_and_track_history(
    true_theta, grid, params, DT, steps_per_block, num_blocks
)

init_theta = jnp.array([1.0, 1.0, 1.0, 1.0])
_, default_history = simulate_and_track_history(
    init_theta, grid, params, DT, steps_per_block, num_blocks
)
integration_run_seconds += time.perf_counter() - integration_start

# 2. Render the time-evolution comparison GIF.
render_time_evolution_gif(
    target_history, default_history, DOMAIN_SIZE,
    OUTDIR / "time_evolution_comparison.gif",
)

target_omega_np = np.array(target_state.omega)

# 2.5 Toggleable parameter scans: loss vs each parameter (eta, p1..p4).
scan_specs = [
    ("eta", None, ETA),
    ("p1", 0, P0),
    ("p2", 1, P1),
    ("p3", 2, P2),
    ("p4", 3, P3),
]
enabled_scans = {
    "eta": SCAN_ETA,
    "p1": SCAN_P1,
    "p2": SCAN_P2,
    "p3": SCAN_P3,
    "p4": SCAN_P4,
}
scan_results: dict[str, dict[str, np.ndarray]] = {}
if any(enabled_scans.values()):
    print("\n=== Running parameter scans (loss vs parameter) ===")
    for name, theta_index, true_value in scan_specs:
        if not enabled_scans[name]:
            continue
        print(f"\nSweeping {name} around true value {true_value:.6g} ...")
        integration_start = time.perf_counter()
        values, losses = run_parameter_scan(
            name, theta_index, true_value, grid, params, target_state, true_theta,
            DT, steps_per_block, num_blocks, SCAN_POINTS, SCAN_RELATIVE_RANGE,
        )
        integration_run_seconds += time.perf_counter() - integration_start
        print_scan_results(name, values, losses, true_value)
        plot_scan(name, values, losses, OUTDIR)
        scan_results[name] = {"values": values, "losses": losses}

    np.savez(
        OUTDIR / "scan_results.npz",
        **{f"{k}_values": v["values"] for k, v in scan_results.items()},
        **{f"{k}_losses": v["losses"] for k, v in scan_results.items()},
    )
    print(f"Saved all scan results to {OUTDIR / 'scan_results.npz'}")

# 3. Setup optimizer (L-BFGS via optax).
optimizer = optax.lbfgs(memory_size=LBFGS_MEMORY_SIZE)
opt_state = optimizer.init(init_theta)

# 4. Define the JIT-compiled optimization step.
@jax.jit
def step(theta, opt_state, target):
    (loss_val, current_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        theta, target, grid, params, DT, steps_per_block, num_blocks
    )

    def value_fn(t):
        return loss_fn(t, target, grid, params, DT, steps_per_block, num_blocks)[0]

    updates, opt_state = optimizer.update(
        grads, opt_state, theta, value=loss_val, grad=grads, value_fn=value_fn
    )
    theta = optax.apply_updates(theta, updates)
    return theta, opt_state, loss_val, current_state


compile_start = time.perf_counter()
step_executable = step.lower(init_theta, opt_state, initial_state).compile()
optimizer_compile_seconds = time.perf_counter() - compile_start
compile_seconds = integration_compile_seconds + optimizer_compile_seconds
print(f"Compile time: {compile_seconds:.3f} s", flush=True)

# 5. Run the optimization loop.
theta = init_theta
loss_history = []
omega_history = []
optimizer_run_start = time.perf_counter()

print("\nStarting L-BFGS optimization loop...")
for i in range(1, OPT_STEPS + 1):
    t0 = time.perf_counter()

    theta, opt_state, loss_val, current_state = step_executable(theta, opt_state, target_state)
    theta.block_until_ready()

    t1 = time.perf_counter()
    loss_history.append(float(loss_val))
    omega_history.append(np.array(current_state.omega))

    if i % max(1, OPT_STEPS // 10) == 0 or i == 1:
        params_str = [f"{p:.4f}" for p in theta]
        print(f"Step {i:03d} | Loss: {loss_val:.6e} | Params: {params_str} "
              f"| Time: {t1 - t0:.3f}s", flush=True)

optimizer_run_seconds = time.perf_counter() - optimizer_run_start
run_seconds = integration_run_seconds + optimizer_run_seconds
print(f"Run time: {run_seconds:.3f} s")

# --- Results ---------------------------------------------------------------
print("\n=== Optimization completed ===")
print(f"recovered theta = {[round(float(p), 4) for p in theta]} "
      f"(true {[P0, P1, P2, P3]}), final loss = {loss_history[-1]:.3e}")

# 6. Plot the loss curve.
plt.figure(figsize=(8, 5))
plt.plot(loss_history, linewidth=2)
plt.yscale("log")
plt.xlabel("Optimization Step")
plt.ylabel("MSE Loss")
plt.title("Optimization of Modified Orszag-Tang Initial Conditions")
plt.grid(True, which="both", ls="--", alpha=0.7)

plot_path = OUTDIR / "initial_condition_opt_loss.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved loss plot to {plot_path}")

# 7. Render the optimization process GIF.
render_optimization_gif(
    target_omega_np, omega_history, DOMAIN_SIZE, OUTDIR / "optimization_process.gif"
)
