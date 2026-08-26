"""Optimize Orszag--Tang initial-condition parameters via reverse-mode autodiff.

Recovers the modified Orszag--Tang initial-condition parameters ``(p0, p1,
p2, p3)`` by differentiating the reduced-MHD evolution end to end (reverse
mode) and minimizing an L-BFGS loss against a "true" synthetic target state.

Also supports toggleable one-dimensional parameter scans (``--scan-eta``,
``--scan-p1`` ... ``--scan-p4``) that sweep each parameter around its true
value and record ``loss vs parameter``.

Usage
-----
    python examples/orszag_tang/optimize_initial_conditions.py --resolution 32
    python examples/orszag_tang/optimize_initial_conditions.py \
        --scan-eta --scan-p1 --scan-p2 --scan-p3 --scan-p4
"""

import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

# Enable double precision for stability
jax.config.update("jax_enable_x64", True)


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


@partial(jax.jit, static_argnums=(2, 4))
@partial(jax.checkpoint, static_argnums=(2, 4))
def run_block(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    grid_lengths: tuple[float, float],
    dt: float,
    steps: int,
) -> ReducedMHDState:
    def step_fn(carry, _):
        s = carry
        k1 = reduced_mhd_rhs(s, params, lengths=grid_lengths)

        s_k2 = ReducedMHDState(
            psi=s.psi + 0.5 * dt * k1.psi, omega=s.omega + 0.5 * dt * k1.omega
        )
        k2 = reduced_mhd_rhs(s_k2, params, lengths=grid_lengths)

        s_k3 = ReducedMHDState(
            psi=s.psi + 0.5 * dt * k2.psi, omega=s.omega + 0.5 * dt * k2.omega
        )
        k3 = reduced_mhd_rhs(s_k3, params, lengths=grid_lengths)

        s_k4 = ReducedMHDState(psi=s.psi + dt * k3.psi, omega=s.omega + dt * k3.omega)
        k4 = reduced_mhd_rhs(s_k4, params, lengths=grid_lengths)

        psi_next = s.psi + (dt / 6.0) * (k1.psi + 2.0 * k2.psi + 2.0 * k3.psi + k4.psi)
        omega_next = s.omega + (dt / 6.0) * (k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega)

        return ReducedMHDState(psi=psi_next, omega=omega_next), None

    final_state, _ = jax.lax.scan(step_fn, state, None, length=steps)
    return final_state


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize modified Orszag-Tang initial condition parameters in MHX."
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/examples/orszag_tang_ic_opt"))
    parser.add_argument("--p0", type=float, default=1.5, help="True value for parameter p0")
    parser.add_argument("--p1", type=float, default=0.8, help="True value for parameter p1")
    parser.add_argument("--p2", type=float, default=1.2, help="True value for parameter p2")
    parser.add_argument("--p3", type=float, default=0.5, help="True value for parameter p3")
    parser.add_argument("--nx", type=int, default=64, help="Grid resolution X")
    parser.add_argument("--ny", type=int, default=64, help="Grid resolution Y")
    parser.add_argument("--resolution", type=int, default=None, help="Set both nx and ny")
    parser.add_argument("--domain-size", type=float, default=10.0, help="Size of the domain")
    parser.add_argument("--eta", type=float, default=1e-3, help="Magnetic resistivity (eta)")
    parser.add_argument("--nu", type=float, default=1e-3, help="Kinematic viscosity (nu)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (dt)")
    parser.add_argument("--t-end", "--t_end", dest="t_end", type=float, default=1.0,
                        help="End time for the simulation")
    parser.add_argument("--opt-steps", type=int, default=20,
                        help="Number of optimization steps (L-BFGS iterations)")
    parser.add_argument("--lbfgs-memory-size", type=int, default=10,
                        help="L-BFGS history memory size")
    # --- Toggleable parameter scans (loss vs parameter) ---
    parser.add_argument("--scan-eta", action="store_true", help="Sweep eta and record loss vs eta")
    parser.add_argument("--scan-p1", action="store_true",
                        help="Sweep p1 (theta[0]) and record loss vs p1")
    parser.add_argument("--scan-p2", action="store_true",
                        help="Sweep p2 (theta[1]) and record loss vs p2")
    parser.add_argument("--scan-p3", action="store_true",
                        help="Sweep p3 (theta[2]) and record loss vs p3")
    parser.add_argument("--scan-p4", action="store_true",
                        help="Sweep p4 (theta[3]) and record loss vs p4")
    parser.add_argument("--scan-points", type=int, default=21,
                        help="Number of points per parameter scan")
    parser.add_argument("--scan-relative-range", type=float, default=0.5,
                        help="Scan range as a fraction around the true value")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.resolution is not None:
        args.nx = args.resolution
        args.ny = args.resolution

    grid = CartesianGrid.from_mesh_config(
        MeshConfig(
            shape=(args.nx, args.ny),
            lower=(0.0, 0.0),
            upper=(args.domain_size, args.domain_size),
        )
    )

    steps_per_block = max(1, int(0.1 / args.dt))
    num_blocks = int(round(args.t_end / (args.dt * steps_per_block)))

    params = ReducedMHDParams(resistivity=args.eta, viscosity=args.nu)
    true_theta = jnp.array([args.p0, args.p1, args.p2, args.p3])

    print("=== Starting Initial Condition Optimization in MHX ===")

    initial_state = modified_orszag_tang_initial_state(true_theta, grid)
    compile_start = time.perf_counter()
    run_block.lower(
        initial_state, params, grid.lengths, args.dt, steps_per_block
    ).compile()
    integration_compile_seconds = time.perf_counter() - compile_start
    integration_run_seconds = 0.0

    # 1. Generate the "true" target state and the baseline state history.
    print("\nSimulating target state and baseline [1,1,1,1] state (this may take a "
          "moment to compile)...")
    integration_start = time.perf_counter()
    target_state, target_history = simulate_and_track_history(
        true_theta, grid, params, args.dt, steps_per_block, num_blocks
    )

    init_theta = jnp.array([1.0, 1.0, 1.0, 1.0])
    _, default_history = simulate_and_track_history(
        init_theta, grid, params, args.dt, steps_per_block, num_blocks
    )
    integration_run_seconds += time.perf_counter() - integration_start

    # 2. Render the time-evolution comparison GIF.
    render_time_evolution_gif(
        target_history, default_history, args.domain_size,
        outdir / "time_evolution_comparison.gif",
    )

    target_omega_np = np.array(target_state.omega)

    # 2.5 Toggleable parameter scans: loss vs each parameter (eta, p1..p4).
    scan_specs = [
        ("eta", None, args.eta),
        ("p1", 0, args.p0),
        ("p2", 1, args.p1),
        ("p3", 2, args.p2),
        ("p4", 3, args.p3),
    ]
    enabled_scans = {
        "eta": args.scan_eta,
        "p1": args.scan_p1,
        "p2": args.scan_p2,
        "p3": args.scan_p3,
        "p4": args.scan_p4,
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
                args.dt, steps_per_block, num_blocks, args.scan_points, args.scan_relative_range,
            )
            integration_run_seconds += time.perf_counter() - integration_start
            print_scan_results(name, values, losses, true_value)
            plot_scan(name, values, losses, outdir)
            scan_results[name] = {"values": values, "losses": losses}

        np.savez(
            outdir / "scan_results.npz",
            **{f"{k}_values": v["values"] for k, v in scan_results.items()},
            **{f"{k}_losses": v["losses"] for k, v in scan_results.items()},
        )
        print(f"Saved all scan results to {outdir / 'scan_results.npz'}")

    # 3. Setup optimizer (L-BFGS via optax).
    optimizer = optax.lbfgs(memory_size=args.lbfgs_memory_size)
    opt_state = optimizer.init(init_theta)

    # 4. Define the JIT-compiled optimization step.
    @jax.jit
    def step(theta, opt_state, target):
        (loss_val, current_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            theta, target, grid, params, args.dt, steps_per_block, num_blocks
        )

        def value_fn(t):
            return loss_fn(t, target, grid, params, args.dt, steps_per_block, num_blocks)[0]

        updates, opt_state = optimizer.update(
            grads, opt_state, theta, value=loss_val, grad=grads, value_fn=value_fn
        )
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss_val, current_state

    compile_start = time.perf_counter()
    step_executable = step.lower(init_theta, opt_state, initial_state).compile()
    optimizer_compile_seconds = time.perf_counter() - compile_start
    compile_seconds = integration_compile_seconds + optimizer_compile_seconds
    print(f"Compile time: {compile_seconds:.3f} s")

    # 5. Run the optimization loop.
    theta = init_theta
    loss_history = []
    omega_history = []
    optimizer_run_start = time.perf_counter()

    print("\nStarting L-BFGS optimization loop...")
    for i in range(1, args.opt_steps + 1):
        t0 = time.perf_counter()

        theta, opt_state, loss_val, current_state = step_executable(theta, opt_state, target_state)
        theta.block_until_ready()

        t1 = time.perf_counter()
        loss_history.append(float(loss_val))
        omega_history.append(np.array(current_state.omega))

        if i % max(1, args.opt_steps // 10) == 0 or i == 1:
            params_str = [f"{p:.4f}" for p in theta]
            print(f"Step {i:03d} | Loss: {loss_val:.6e} | Params: {params_str} "
                  f"| Time: {t1 - t0:.3f}s")

    print("\n=== Optimization Completed! ===")
    optimizer_run_seconds = time.perf_counter() - optimizer_run_start
    run_seconds = integration_run_seconds + optimizer_run_seconds
    print(f"Run time: {run_seconds:.3f} s")

    # 6. Plot the loss curve.
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Optimization Step")
    plt.ylabel("MSE Loss")
    plt.title("Optimization of Modified Orszag-Tang Initial Conditions")
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plot_path = outdir / "initial_condition_opt_loss.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss plot to {plot_path}")

    # 7. Render the optimization process GIF.
    render_optimization_gif(
        target_omega_np, omega_history, args.domain_size, outdir / "optimization_process.gif"
    )


if __name__ == "__main__":
    main()
