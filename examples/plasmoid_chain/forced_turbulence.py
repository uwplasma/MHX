"""Hybrid forced-turbulent-reconnection runner on the plasmoid-chain domain.

Builds the forced-turbulent-reconnection setup (from
``src/mhx/benchmarks/turbulence.py``) on the long thin two-sheet plasmoid domain
used by ``plasmoids.py``:

- **Domain / sheet** from the plasmoid chain: ``2*sheet_sep × 8π`` with two thin
  Harris sheets (``--sheet-sep 12.57``, ``--nx 1024`` etc.).
- **Seed / forcing / goal / rendered field** from forced-turbulent reconnection:
  a seeded tearing mode plus broadband turbulence, a persistent large-scale
  vorticity source applied every step, the reconnection-rate proxy goal, and an
  out-of-plane current density ``j_z`` rendered field with flux contours.

Usage (requires numpy/jax):
    python examples/plasmoid_chain/forced_turbulence.py \\
        --nx 256 --ny 1024 --sheet-sep 12.57 --t-end 60 --outdir outputs/forced_turbulence

Writes ``forced_turbulence_trajectory.npz`` (``time``, ``psi``, ``current_density``)
plus a ``forced_turbulent_reconnection.gif`` in ``--outdir``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from mhx.benchmarks.turbulence import (
    _broadband_scalar_field,
    turbulent_initial_state,
)
from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import laplacian
from mhx.physics.equilibria import PeriodicDoubleHarrisEquilibrium
from mhx.runtime import configure_jax
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import rk4_step


def _dealias_filter(grid: CartesianGrid):
    from mhx.numerics.spectral import spectral_wavenumbers

    kx = spectral_wavenumbers(grid.shape[0], grid.lengths[0])
    ky = spectral_wavenumbers(grid.shape[1], grid.lengths[1])
    cutoff_x = (2.0 / 3.0) * np.pi * grid.shape[0] / grid.lengths[0]
    cutoff_y = (2.0 / 3.0) * np.pi * grid.shape[1] / grid.lengths[1]
    mask = (jnp.abs(kx)[:, None] < cutoff_x) & (jnp.abs(ky)[None, :] < cutoff_y)

    def dealias(state: ReducedMHDState) -> ReducedMHDState:
        return ReducedMHDState(
            psi=jnp.real(jnp.fft.ifftn(mask * jnp.fft.fftn(state.psi))),
            omega=jnp.real(jnp.fft.ifftn(mask * jnp.fft.fftn(state.omega))),
        )

    return dealias


def _advance_block(state, step, steps: int):
    def body(carry, _):
        return step(carry), None

    return jax.lax.scan(body, state, None, length=steps)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=256, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=1024, help="Grid points in y")
    parser.add_argument("--sheet-sep", type=float, default=12.57)
    parser.add_argument("--width", type=float, default=0.32)
    parser.add_argument("--t-end", dest="t_end", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eta", type=float, default=1.5e-3)
    parser.add_argument("--nu", type=float, default=1.5e-3)
    parser.add_argument("--perturbation-amplitude", type=float, default=1.0e-2)
    parser.add_argument("--turbulent-flux-amplitude", type=float, default=1.5e-2)
    parser.add_argument("--turbulent-flow-amplitude", type=float, default=1.5e-2)
    parser.add_argument("--forcing-amplitude", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/forced_turbulence"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_jax(enable_x64=True)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    Lx, Ly = 2.0 * args.sheet_sep, 8.0 * np.pi
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=(args.nx, args.ny), lower=(0.0, 0.0), upper=(Lx, Ly))
    )

    # Sheet + turbulence composite initial state (forced-TR recipe).
    sheet = PeriodicDoubleHarrisEquilibrium(
        width=args.width,
        perturbation_amplitude=args.perturbation_amplitude,
        perturbation_mode=(0, 1),
    ).initial_state(grid)
    turb = turbulent_initial_state(
        grid,
        seed=args.seed,
        flux_amplitude=args.turbulent_flux_amplitude,
        flow_amplitude=args.turbulent_flow_amplitude,
        kmin=1,
        kmax=4,
    )
    state = ReducedMHDState(psi=sheet.psi + turb.psi, omega=sheet.omega + turb.omega)

    # Persistent large-scale vorticity source (forced-TR recipe).
    forcing_stream = _broadband_scalar_field(grid, seed=args.seed + 2027, kmin=1, kmax=3)
    forcing_omega = args.forcing_amplitude * laplacian(
        jnp.asarray(forcing_stream), lengths=grid.lengths
    )
    params = ReducedMHDParams(resistivity=args.eta, viscosity=args.nu)

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        base = reduced_mhd_rhs(state, params, lengths=grid.lengths)
        return ReducedMHDState(psi=base.psi, omega=base.omega + forcing_omega)

    dealias = _dealias_filter(grid)

    def step(state: ReducedMHDState) -> ReducedMHDState:
        return dealias(rk4_step(state, rhs, args.dt))
    advance = jax.jit(_advance_block, static_argnames=("step", "steps"))

    num_steps = int(args.t_end / args.dt)
    num_blocks = num_steps // args.save_every
    if num_blocks < 1:
        raise ValueError("--t-end too short for --save-every")

    compile_start = time.perf_counter()
    advance_executable = advance.lower(
        state, step=step, steps=args.save_every
    ).compile()
    compile_seconds = time.perf_counter() - compile_start
    print(f"Compile time: {compile_seconds:.3f} s")

    psi_frames = [np.asarray(state.psi)]
    t_history = [0.0]
    run_seconds = 0.0
    for block in range(1, num_blocks + 1):
        t0 = time.perf_counter()
        state = advance_executable(state)
        state.psi.block_until_ready()
        elapsed = time.perf_counter() - t0
        run_seconds += elapsed
        psi_frames.append(np.asarray(state.psi))
        t_history.append(block * args.save_every * args.dt)
        print(f"  block {block}/{num_blocks} done in {elapsed:.1f}s", flush=True)

    print(f"Run time: {run_seconds:.3f} s")

    psi_series = np.asarray(psi_frames, dtype=np.float64)
    current_series = np.asarray(
        [
            -np.asarray(laplacian(frame, lengths=grid.lengths), dtype=np.float64)
            for frame in psi_series
        ]
    )
    trajectory = outdir / "forced_turbulence_trajectory.npz"
    np.savez_compressed(
        trajectory,
        time=np.array(t_history),
        psi=psi_series,
        current_density=current_series,
        nx=args.nx,
        ny=args.ny,
        lx=Lx,
        ly=Ly,
        eta=args.eta,
        forcing_amplitude=args.forcing_amplitude,
        seed=args.seed,
    )
    print(f"Saved {trajectory}")

    # Render a compact forced-TR GIF (j_z + flux contours).
    frames = []
    indices = np.unique(np.linspace(0, len(t_history) - 1, 18, dtype=int))
    for idx in indices:
        jz = current_series[idx]
        psi = psi_series[idx]
        vmax = max(float(np.percentile(np.abs(jz), 99.0)), np.finfo(float).eps)
        fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=80)
        ax.imshow(jz.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  extent=(0.0, Lx, 0.0, Ly), aspect="auto")
        levels = np.linspace(float(np.percentile(psi, 5.0)),
                             float(np.percentile(psi, 95.0)), 18)
        ax.contour(psi.T, levels=levels, colors="black", linewidths=0.35)
        ax.set_title(f"Forced turbulent reconnection, t={t_history[idx]:.0f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    gif = outdir / "forced_turbulent_reconnection.gif"
    imageio.mimsave(gif, frames, duration=90, loop=0, palettesize=32)
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
