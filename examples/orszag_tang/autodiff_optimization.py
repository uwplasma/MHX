"""Fit viscosity and resistivity of the Orszag--Tang vortex to Vencels energy curves.

Differentiates the incompressible reduced-MHD Orszag--Tang run end to end (reverse
mode, with checkpointed RK4 blocks) and minimizes the squared difference between
the simulated magnetic and kinetic-plus-dissipated energy changes and the Vencels
kinetic reference curves with a hand-written Adam update of (eta, nu). It then
re-runs the optimized parameters and writes the fluid-versus-kinetic figure. The
default 256 x 256 run takes several minutes on a laptop CPU; lower NX and NY for
a quick look, raise T_END and OPT_STEPS for a better fit (T_END must be a multiple
of DT * SAVE_EVERY). Run from the repository root:

    python examples/orszag_tang/autodiff_optimization.py
"""

import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from mhx.benchmarks.vencels import (
    VENCELS_CSV,
    load_vencels_reference,
    run_orszag_tang,
    vencels_orszag_tang_initial_state,
)
from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_energy
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.plotting import plot_kinetic_comparison
from mhx.state import ReducedMHDParams
from mhx.time_integrators import advance_rk4

# Enable double precision for stability
jax.config.update("jax_enable_x64", True)

# --- Input parameters ------------------------------------------------------
OUTDIR = Path("outputs/examples/orszag_tang_opt")
NX = 256  # grid points in x
NY = 256  # grid points in y
DOMAIN_SIZE = 10.0  # box side (also the Alfven time t_A)
DT = 0.001  # time step
T_END = 2.0  # end time; must be an integer multiple of DT * SAVE_EVERY
SAVE_EVERY = 200  # RK4 steps per checkpointed block (one energy sample per block)
OPT_STEPS = 15  # Adam iterations
ETA0 = 0.010  # initial guess of the resistivity
NU0 = 0.010  # initial guess of the viscosity
LEARNING_RATE = 1.0e-3  # Adam update rate for both parameters

# --- Helpers ---------------------------------------------------------------


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

    def rhs(active_state):
        return reduced_mhd_rhs(active_state, params, lengths=grid.lengths)

    state0 = vencels_orszag_tang_initial_state(grid)
    EB0 = magnetic_energy(state0, lengths=grid.lengths)
    EK0 = kinetic_energy(state0, lengths=grid.lengths)
    Etot0 = EB0 + EK0

    EB_list = [EB0]
    EK_list = [EK0]

    state = state0
    for _ in range(num_blocks):
        state = advance_rk4(state, rhs, dt=dt, steps=steps_per_block, checkpoint=True)
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


# --- Run -------------------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
grid = CartesianGrid.from_mesh_config(
    MeshConfig(shape=(NX, NY), lower=(0.0, 0.0), upper=(DOMAIN_SIZE, DOMAIN_SIZE))
)
steps_per_block = SAVE_EVERY
block_duration = DT * steps_per_block
exact_num_blocks = T_END / block_duration
num_blocks = int(round(exact_num_blocks))
if not np.isclose(exact_num_blocks, num_blocks, rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "T_END must be an integer multiple of DT * SAVE_EVERY so the "
        "differentiable trajectory reaches the requested end time exactly"
    )
executed_t_end = num_blocks * block_duration
times = np.arange(num_blocks + 1, dtype=np.float64) * block_duration
vencels_time = times / DOMAIN_SIZE  # normalized time t / t_A

# Vencels target curves.
interp_B, interp_e, interp_i = load_vencels_reference(VENCELS_CSV)
if interp_B is None or interp_e is None or interp_i is None:
    raise FileNotFoundError(f"Could not load vencelsdata.csv at: {VENCELS_CSV}")
ref_B_vals = jnp.array(interp_B(vencels_time))
ref_particles_vals = jnp.array(interp_e(vencels_time)) + jnp.array(interp_i(vencels_time))

loss_val_grad = jax.jit(
    jax.value_and_grad(
        partial(
            loss_fn,
            grid=grid,
            dt=DT,
            steps_per_block=steps_per_block,
            num_blocks=num_blocks,
            ref_B_vals=ref_B_vals,
            ref_particles_vals=ref_particles_vals,
        )
    )
)

theta = jnp.array([ETA0, NU0])
m = jnp.zeros_like(theta)
v = jnp.zeros_like(theta)
lr = jnp.array([LEARNING_RATE, LEARNING_RATE])  # per-parameter update rate

print(f"=== Incompressible Orszag-Tang viscoresistive optimization ({NX}x{NY}) ===")
print(f"Requested t_end={T_END:g}; executing {num_blocks} blocks "
      f"to physical time {executed_t_end:g} (t/t_A={executed_t_end / DOMAIN_SIZE:g}).")
print(f"Optimizing resistivity (eta) and viscosity (nu) to match Vencels energy curves "
      f"for {OPT_STEPS} steps; devices: {jax.devices()}", flush=True)

compile_start = time.perf_counter()
loss_val_grad_executable = loss_val_grad.lower(theta).compile()
compile_seconds = time.perf_counter() - compile_start
print(f"Compile time: {compile_seconds:.3f} s", flush=True)

history = []
run_start = time.perf_counter()
for step in range(1, OPT_STEPS + 1):
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
          f"| Time: {t1 - t0:.4f}s", flush=True)
    history.append((step, float(loss_val), eta, nu))

final_eta = float(jnp.maximum(theta[0], 5e-4))
final_nu = float(jnp.maximum(theta[1], 5e-4))

# Re-run with the optimized parameters.
print("\nRe-running with the optimized parameters ...", flush=True)
result = run_orszag_tang(
    shape=(NX, NY),
    resistivity=final_eta,
    viscosity=final_nu,
    dt=DT,
    t_end=executed_t_end,
    save_every=steps_per_block,
)
run_seconds = time.perf_counter() - run_start
print(f"Run time: {run_seconds:.3f} s")

# --- Results ---------------------------------------------------------------
print("\n=== Optimization completed ===")
print(f"Optimized resistivity (eta): {final_eta:.6f}")
print(f"Optimized viscosity (nu):   {final_nu:.6f}")

figure_path = plot_kinetic_comparison(
    result, OUTDIR / "kinetic_comparison.png", VENCELS_CSV, DOMAIN_SIZE
)
print(f"Saved kinetic comparison figure to: {figure_path}")
