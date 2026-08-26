"""Optimize viscosity and resistivity of incompressible Orszag-Tang using JAX
autodiff against Vencels energy curves."""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib
from scipy.interpolate import interp1d

matplotlib.use("Agg")  # Headless Matplotlib backend to prevent WSL/NERSC hangs
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import reduced_mhd_rhs, stream_function
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import gradient
from mhx.state import ReducedMHDParams, ReducedMHDState

# The Vencels CSV ships alongside this example.
VENCELS_CSV = Path(__file__).resolve().parent.parent / "vencelsdata.csv"

# Enable double precision for stability
jax.config.update("jax_enable_x64", True)


def orszag_tang_initial_state(grid: CartesianGrid) -> ReducedMHDState:
    """Return the incompressible reduced-MHD Orszag--Tang initial condition.

    Scaled by 0.2 to match Vencels' ``delta_B / B_0 = 0.2`` and ``U_s`` bulk
    flow magnitude.
    """
    x, y = grid.mesh()
    Lx, Ly = grid.lengths
    kx = 2.0 * jnp.pi / Lx
    ky = 2.0 * jnp.pi / Ly

    psi = 0.2 * ((1.0 / ky) * jnp.cos(ky * y) + (0.5 / kx) * jnp.cos(2.0 * kx * x))
    omega = 0.2 * (kx * jnp.cos(kx * x) + ky * jnp.cos(ky * y))

    return ReducedMHDState(psi=psi, omega=omega)


def load_vencels_reference(csv_path: Path):
    """Parse the Vencels CSV format and return energy interpolation functions."""
    t_B, y_B = [], []
    t_e, y_e = [], []
    t_i, y_i = [], []

    if not csv_path.exists():
        return None, None, None

    with open(csv_path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            cols = line.strip().split("\t")
            if len(cols) >= 1 and cols[0]:
                b = cols[0].replace('"', "").split(",")
                if len(b) == 2:
                    t_B.append(float(b[0]))
                    y_B.append(float(b[1]))
            if len(cols) >= 2 and cols[1]:
                r = cols[1].replace('"', "").split(",")
                if len(r) == 2:
                    t_e.append(float(r[0]))
                    y_e.append(float(r[1]))
            if len(cols) >= 3 and cols[2]:
                g = cols[2].replace('"', "").split(",")
                if len(g) == 2:
                    t_i.append(float(g[0]))
                    y_i.append(float(g[1]))

    t_B, y_B = zip(*sorted(zip(t_B, y_B, strict=False)), strict=False)
    t_e, y_e = zip(*sorted(zip(t_e, y_e, strict=False)), strict=False)
    t_i, y_i = zip(*sorted(zip(t_i, y_i, strict=False)), strict=False)

    interp_B = interp1d(t_B, y_B, bounds_error=False, fill_value="extrapolate")
    interp_e = interp1d(t_e, y_e, bounds_error=False, fill_value="extrapolate")
    interp_i = interp1d(t_i, y_i, bounds_error=False, fill_value="extrapolate")

    return interp_B, interp_e, interp_i


def magnetic_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    """Return mean magnetic energy density."""
    grad_psi = gradient(state.psi, lengths=lengths)
    return 0.5 * sum(jnp.mean(component**2) for component in grad_psi)


def kinetic_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    """Return mean kinetic energy density."""
    phi = stream_function(state.omega, lengths=lengths)
    grad_phi = gradient(phi, lengths=lengths)
    return 0.5 * sum(jnp.mean(component**2) for component in grad_phi)


@partial(jax.checkpoint, static_argnums=(2, 4))
def run_block(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    grid_lengths: tuple[float, float],
    dt: float,
    steps: int,
) -> ReducedMHDState:
    """Run ``steps`` steps of RK4 time integration, checkpointed to prevent OOM."""

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


def run_differentiable_simulation(
    theta: Array,
    grid: CartesianGrid,
    dt: float,
    steps_per_block: int,
    num_blocks: int,
) -> tuple[Array, Array, float]:
    """Simulate Orszag-Tang and return the magnetic and kinetic energy histories."""
    eta = jnp.maximum(theta[0], 5.0e-4)  # Clamp to prevent instability / division by zero
    nu = jnp.maximum(theta[1], 5.0e-4)
    params = ReducedMHDParams(resistivity=eta, viscosity=nu)

    state0 = orszag_tang_initial_state(grid)
    EB0 = magnetic_energy(state0, lengths=grid.lengths)
    EK0 = kinetic_energy(state0, lengths=grid.lengths)
    Etot0 = EB0 + EK0

    EB_list = [EB0]
    EK_list = [EK0]

    state = state0
    for _ in range(num_blocks):
        state = run_block(state, params, grid.lengths, dt, steps_per_block)
        EB_list.append(magnetic_energy(state, lengths=grid.lengths))
        EK_list.append(kinetic_energy(state, lengths=grid.lengths))

    return jnp.stack(EB_list), jnp.stack(EK_list), Etot0


def loss_fn(
    theta: Array,
    grid: CartesianGrid,
    dt: float,
    steps_per_block: int,
    num_blocks: int,
    ref_B_vals: Array,
    ref_particles_vals: Array,
) -> float:
    """L2 loss between simulation energy-change histories and reference curves."""
    EB, EK, Etot0 = run_differentiable_simulation(theta, grid, dt, steps_per_block, num_blocks)

    # Energy changes normalized by initial total energy.
    delta_EB = (EB - EB[0]) / Etot0
    delta_particles = -delta_EB  # In incompressible MHD, Delta(E_K + E_diss) = -Delta(E_B)

    loss_B = jnp.sum((delta_EB - ref_B_vals) ** 2)
    loss_particles = jnp.sum((delta_particles - ref_particles_vals) ** 2)

    return loss_B + loss_particles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/examples/orszag_tang_opt"))
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--domain-size", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--t-end", dest="t_end", type=float, default=30.0)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--opt-steps", type=int, default=15)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Grid & time history bounds.
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(
            shape=(args.nx, args.ny),
            lower=(0.0, 0.0),
            upper=(args.domain_size, args.domain_size),
        )
    )
    steps_per_block = args.save_every
    block_duration = args.dt * steps_per_block
    exact_num_blocks = args.t_end / block_duration
    num_blocks = int(round(exact_num_blocks))
    if not np.isclose(exact_num_blocks, num_blocks, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "t_end must be an integer multiple of dt * save_every so the "
            "differentiable trajectory reaches the requested end time exactly"
        )
    executed_t_end = num_blocks * block_duration

    times = np.arange(num_blocks + 1, dtype=np.float64) * block_duration
    vencels_time = times / args.domain_size  # Normalized time t/t_A

    # 2. Load Vencels target curves.
    interp_B, interp_e, interp_i = load_vencels_reference(VENCELS_CSV)
    if interp_B is None or interp_e is None or interp_i is None:
        raise FileNotFoundError(f"Could not load vencelsdata.csv at: {VENCELS_CSV}")

    ref_B_vals = jnp.array(interp_B(vencels_time))
    ref_e_vals = jnp.array(interp_e(vencels_time))
    ref_i_vals = jnp.array(interp_i(vencels_time))
    ref_particles_vals = ref_e_vals + ref_i_vals

    # 3. JAX gradient function.
    loss_val_grad = jax.jit(
        jax.value_and_grad(
            partial(
                loss_fn,
                grid=grid,
                dt=args.dt,
                steps_per_block=steps_per_block,
                num_blocks=num_blocks,
                ref_B_vals=ref_B_vals,
                ref_particles_vals=ref_particles_vals,
            )
        )
    )

    # 4. Optimization loop (initial guess eta = nu = 0.010).
    theta = jnp.array([0.010, 0.010])
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    lr = jnp.array([1.0e-3, 1.0e-3])  # per-parameter update rate

    compile_start = time.perf_counter()
    loss_val_grad_executable = loss_val_grad.lower(theta).compile()
    compile_seconds = time.perf_counter() - compile_start

    print(f"=== Starting Incompressible Orszag-Tang Viscoresistive Optimization "
          f"({args.nx}x{args.ny}) ===")
    print(
        f"Requested t_end={args.t_end:g}; executing {num_blocks} blocks "
        f"to physical time {executed_t_end:g} (t/t_A={executed_t_end / args.domain_size:g})."
    )
    print(f"Optimizing resistivity (eta) and viscosity (nu) to match Vencels "
          f"energy curves for {args.opt_steps} steps...")

    history = []
    print(f"Compile time: {compile_seconds:.3f} s")
    run_start = time.perf_counter()

    # Store initial curve for final plotting comparison.
    init_EB, init_EK, init_Etot0 = run_differentiable_simulation(
        theta, grid, args.dt, steps_per_block, num_blocks
    )
    init_delta_EB = np.array((init_EB - init_EB[0]) / init_Etot0)
    init_delta_particles = -init_delta_EB

    for step in range(1, args.opt_steps + 1):
        t0 = time.perf_counter()

        loss_val, grad = loss_val_grad_executable(theta)

        eta = float(jnp.maximum(theta[0], 5e-4))
        nu = float(jnp.maximum(theta[1], 5e-4))

        # Custom Adam step.
        beta1, beta2 = 0.9, 0.999
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        theta.block_until_ready()
        t1 = time.perf_counter()

        print(f"Step {step:02d} | Loss: {loss_val:.6e} | eta: {eta:.6f} | nu: {nu:.6f} "
              f"| Time: {t1 - t0:.4f}s")
        history.append((step, float(loss_val), eta, nu))

    # 5. Output final results & comparison plot.
    final_eta = float(jnp.maximum(theta[0], 5e-4))
    final_nu = float(jnp.maximum(theta[1], 5e-4))
    print("\n=== Optimization Completed! ===")
    print(f"Optimized Resistivity (eta): {final_eta:.6f}")
    print(f"Optimized Viscosity (nu):   {final_nu:.6f}")

    opt_EB, opt_EK, opt_Etot0 = run_differentiable_simulation(
        theta, grid, args.dt, steps_per_block, num_blocks
    )
    opt_delta_EB = np.array((opt_EB - opt_EB[0]) / opt_Etot0)
    opt_delta_particles = -opt_delta_EB
    run_seconds = time.perf_counter() - run_start
    print(f"Run time: {run_seconds:.3f} s")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(vencels_time, ref_B_vals, label="B (Vencels Target)", color="navy",
            linestyle=":", alpha=0.8)
    ax.plot(vencels_time, ref_particles_vals, label="e+i (Vencels Target)", color="purple",
            linestyle=":", alpha=0.8)

    ax.plot(vencels_time, init_delta_EB, label="B (Initial Guess)", color="navy",
            linestyle="--", alpha=0.5)
    ax.plot(vencels_time, init_delta_particles, label="e+i (Initial Guess)", color="purple",
            linestyle="--", alpha=0.5)

    ax.plot(vencels_time, opt_delta_EB, label="B (Optimized)", color="navy", linewidth=2.0)
    ax.plot(vencels_time, opt_delta_particles, label="e+i (Optimized)", color="purple",
            linewidth=2.0)

    ax.set_title(
        "Orszag-Tang Parameter Optimization (Autodiff)\n"
        f"physical t_end={executed_t_end:g}; t/t_A={executed_t_end / args.domain_size:g}"
    )
    ax.set_xlabel(r"$t/t_A$")
    ax.set_ylabel(r"$\Delta E$")
    ax.set_xlim(0.0, 3.0)
    ax.legend(frameon=False)

    plot_path = outdir / "orszag_tang_reconstruction.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved reconstruction history plot to: {plot_path}")


if __name__ == "__main__":
    main()
