#!/usr/bin/env python3
"""Many-plasmoid reduced-MHD tearing example.

How to get MANY plasmoids
-------------------------
1. **Long, thin sheets** — use the periodic double-Harris equilibrium with a
   small ``width`` (here 0.05). The tearing wavelength is ``lambda ~ 10*width``,
   so ``width=0.05`` fits ~50 islands along each ``Ly = 8*pi`` sheet.
2. **Broadband seeding** — white noise excites every tearing mode at once
   (``m_y = 1, 2, 3, ...``); all grow simultaneously into a chain.
3. **High global Lundquist number** — ``S_L = Ly*vA/eta``. With ``eta=5e-5``,
   ``S_L ~ 5e5``, far above the Sweet-Parker plasmoid-instability onset
   (``S ~ 1e4``, Loureiro-Schekochihin-Cowley 2007).
4. **Hyper-resistivity** — ``-eta4*laplacian^2`` damps grid-scale noise
   (``k^4`` diffusion) without touching the low-k tearing modes. ``eta4`` is
   auto-scaled with resolution so the grid-scale damping rate ``eta4*k_max^4``
   stays ~100/s (override with ``--eta4``).
5. **Two sheets** (double Harris) — two chains at once, plus sheet-sheet
   interaction.

Stability
---------
The nonlinear phase thins the current sheets until they approach the grid
scale, which is exactly where pseudo-spectral codes blow up without
dealiasing. This script applies a 2/3-rule spectral filter after every RK4
step (the legacy solver dealiased; the current ``reduced_mhd_rhs`` does not)
plus the hyper-resistivity above.

Outputs (in ``--outdir``)
-------------------------
- ``plasmoids.gif``                 ψ-coloured map + flux contours (with colourbar)
- ``plasmoids_final.png``           last frame
- ``plasmoids_diagnostics.png``     energies and |J_z|_max time histories
- ``plasmoids_diagnostics.npz``     time series (Ek, Eb, jz_max, mode amplitudes)
- ``plasmoids_final_state.npz``     final psi/omega arrays + metadata

Colouring
---------
Frames are coloured by the magnetic flux ``ψ`` so the whole domain is
coloured (J_z is concentrated in the thin sheets and leaves most of the
graph near-white). Plasmoid islands appear as coloured blobs ringed by the
black flux contours.

Usage
-----
Quick check (fast iteration, ~2x cheaper than the production preset)::

    python examples/plasmoid_chain/plasmoids.py

Production-scale chain (higher resolution, ``eta4`` auto-scales)::

    python examples/plasmoid_chain/plasmoids.py --nx 512 --ny 2048 --t-end 60

Spread the two Harris sheets further apart (default separation 2*pi; scale
``--nx`` with it so the sheets stay resolved, roughly ``nx ~ 80*sheet_sep``)::

    python examples/plasmoid_chain/plasmoids.py --sheet-sep 12.57 --nx 1024

Zoom into one sheet::

    python examples/plasmoid_chain/plasmoids.py --crop "0.1,0.4"

Note: this is an exploratory demonstration, not a production Sweet-Parker
plasmoid-chain claim (see ``docs/audit.md`` and ``docs/media.md`` for the
repo's claim boundaries).
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

from mhx.config import MeshConfig
from mhx.diagnostics import (
    kinetic_energy,
    magnetic_energy,
    mode_amplitude,
)
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import spectral_wavenumbers
from mhx.physics.equilibria import PeriodicDoubleHarrisEquilibrium
from mhx.physics.terms import HyperResistivityTerm
from mhx.runtime import configure_jax
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import rk4_step

TRACKED_Y_MODES = (1, 2, 4, 8, 16, 32)  # y-harmonics monitored for multi-mode growth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Many-plasmoid double-Harris tearing (current MHX API)."
    )
    parser.add_argument("--nx", type=int, default=512, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=1024, help="Grid points in y")
    parser.add_argument(
        "--sheet-sep",
        type=float,
        default=2.0 * np.pi,
        help="Separation between the two Harris sheets (default 2*pi; Lx = 2*sep)",
    )
    parser.add_argument("--t-end", dest="t_end", type=float, default=60.0, help="End time")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step")
    parser.add_argument("--save-every", type=int, default=250, help="Steps between frames")
    parser.add_argument("--eta", type=float, default=5e-5, help="Resistivity")
    parser.add_argument("--nu", type=float, default=5e-5, help="Viscosity")
    parser.add_argument(
        "--eta4",
        type=float,
        default=None,
        help="Hyper-resistivity / hyper-viscosity (auto-scales with resolution if omitted)",
    )
    parser.add_argument("--width", type=float, default=0.05, help="Half-width of each Harris sheet")
    parser.add_argument("--noise-amp", type=float, default=2e-3, help="White-noise seed amplitude")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible noise")
    parser.add_argument(
        "--crop",
        default=None,
        help='x-fraction crop of the domain, e.g. "0.10,0.40" to zoom on the left sheet',
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output directory (gitignored by default)",
    )
    return parser.parse_args()


def make_rhs(
    params: ReducedMHDParams,
    lengths: tuple[float, float],
    eta4: float,
    nu4: float,
):
    """Return the reduced-MHD RHS with hyper-resistivity attached."""
    term = HyperResistivityTerm(eta4=eta4, nu4=nu4)
    return lambda state: reduced_mhd_rhs(state, params, lengths=lengths, terms=(term,))


def make_dealias_filter(grid: CartesianGrid):
    """Return a 2/3-rule spectral filter (dealiasing).

    The current ``reduced_mhd_rhs`` does not dealias its nonlinear products
    (the legacy solver did). Without a filter, aliased energy piles up at the
    Nyquist scale as the current sheets thin and the run eventually blows up
    (NaN). Zeroing modes above 2/3 of Nyquist each step restores stability.
    """
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


def make_step(rhs, dealias, dt: float):
    """One full RK4 step followed by the dealiasing filter."""
    return lambda state: dealias(rk4_step(state, rhs, dt))


def advance_block(state: ReducedMHDState, step, steps: int) -> ReducedMHDState:
    """Advance ``steps`` full steps with ``jax.lax.scan`` (memory-safe blocks)."""

    def body(carry, _):
        return step(carry), None

    return jax.lax.scan(body, state, None, length=steps)[0]


def render_frame(
    state: ReducedMHDState,
    grid: CartesianGrid,
    time_value: float,
    crop: tuple[float, float] | None,
) -> np.ndarray:
    """Draw the ψ-coloured map + flux contours; return the RGBA frame."""
    psi = np.asarray(state.psi)
    nx = psi.shape[0]

    i0, i1 = 0, nx
    if crop is not None:
        i0 = int(crop[0] * nx)
        i1 = max(int(crop[1] * nx), i0 + 1)
        psi = psi[i0:i1, :]

    # Physical extent: pixel centres sit at cell centres (offset 0.5*dx),
    # which matches imshow's extent mapping for cell-centred grids.
    dx = grid.spacing[0]
    extent = (i0 * dx, i1 * dx, 0.0, grid.lengths[1])

    # Symmetric percentile limits so the flux fills the colour range.
    if np.isfinite(psi).all():
        lo, hi = np.percentile(psi, [1.0, 99.0])
        limit = max(abs(float(lo)), abs(float(hi)))
    else:
        limit = 1.0
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0

    aspect = grid.lengths[1] / grid.lengths[0]
    fig, ax = plt.subplots(figsize=(4.2, 4.2 * aspect), constrained_layout=True)
    image = ax.imshow(
        psi.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
        extent=extent, aspect="auto",
    )
    fig.colorbar(image, ax=ax, shrink=0.9, label=r"$\psi$")

    # Overlay flux contours on the coloured flux map (islands = closed loops).
    # Must share the imshow extent/origin, otherwise contour uses raw array
    # indices and autoscaling blows the axes out, squeezing the image away.
    ax.contour(
        psi.T, levels=48, colors="black", linewidths=0.4, alpha=0.6,
        extent=extent, origin="lower",
    )

    ax.set_title(f"t = {time_value:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    args = parse_args()
    configure_jax(enable_x64=True)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    crop = None
    if args.crop:
        lo, hi = (float(v) for v in args.crop.split(","))
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("--crop must be two increasing fractions in [0, 1]")
        crop = (lo, hi)

    sheet_sep = args.sheet_sep
    if sheet_sep <= 0.0:
        raise ValueError("--sheet-sep must be positive")
    Lx, Ly = 2.0 * sheet_sep, 8.0 * np.pi
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=(args.nx, args.ny), lower=(0.0, 0.0), upper=(Lx, Ly))
    )

    # Hyper-resistivity: default auto-scales so the grid-scale damping rate
    # eta4*k_max^4 is ~100/s regardless of resolution.
    k_max = max(np.pi * args.nx / Lx, np.pi * args.ny / Ly)
    if args.eta4 is None:
        args.eta4 = 100.0 / k_max**4

    # Physics summary: estimate the expected island count per sheet.
    lambda_island = 10.0 * args.width  # lambda ~ 2*pi/(k_max), k_max*a ~ 0.6
    expected_per_sheet = max(1, int(round(Ly / lambda_island)))
    s_lundquist = Ly / args.eta  # v_A ~ 1 in these units
    print("=" * 70)
    print(f"Double-Harris many-plasmoid run: {args.nx}x{args.ny} on [{Lx:.3f}, {Ly:.3f}]")
    print(f"  sheet width a      = {args.width:.4g}  (a/Lx = {args.width / Lx:.4g})")
    print(f"  sheet separation   = {sheet_sep:.3g}  (sheets at x = Lx/4 and 3Lx/4)")
    print(f"  resistivity eta    = {args.eta:.3g}  ->  S_L = Ly/eta ~ {s_lundquist:.2g}")
    print(f"  hyper-resistivity  = {args.eta4:.3g}  (grid damping ~100/s at k_max={k_max:.3g})")
    print(
        f"  tearing wavelength ~ {lambda_island:.3g}  ->  "
        f"~{expected_per_sheet} islands per sheet (2 sheets)"
    )
    print("=" * 70)

    # 1. Clean double-Harris equilibrium (two thin sheets).
    eq = PeriodicDoubleHarrisEquilibrium(
        width=args.width, amplitude=1.0, perturbation_amplitude=0.0
    )
    base = eq.initial_state(grid)

    # 2. Broadband white-noise seed -> every tearing mode grows at once.
    key = jax.random.PRNGKey(args.seed)
    noise = args.noise_amp * jax.random.normal(key, base.psi.shape)
    state = ReducedMHDState(psi=base.psi + noise, omega=base.omega)

    params = ReducedMHDParams(resistivity=args.eta, viscosity=args.nu)
    rhs = make_rhs(params, grid.lengths, args.eta4, args.eta4)
    dealias = make_dealias_filter(grid)
    step = make_step(rhs, dealias, args.dt)

    advance = jax.jit(advance_block, static_argnames=("step", "steps"))

    frames: list[np.ndarray] = []
    t_history: list[float] = []
    ek_history: list[float] = []
    eb_history: list[float] = []
    jzmax_history: list[float] = []
    mode_history: list[np.ndarray] = []

    num_steps = int(args.t_end / args.dt)
    num_blocks = num_steps // args.save_every
    if num_blocks < 1:
        raise ValueError("--t-end too short for --save-every")

    # Diagnostics at t=0.
    frames.append(render_frame(state, grid, 0.0, crop))
    jz0 = np.abs(np.asarray(current_density(state.psi, lengths=grid.lengths)))
    t_history.append(0.0)
    ek_history.append(float(kinetic_energy(state, lengths=grid.lengths)))
    eb_history.append(float(magnetic_energy(state, lengths=grid.lengths)))
    jzmax_history.append(float(np.max(jz0)))
    mode_history.append(
        np.array([float(mode_amplitude(state, mode=(0, m))) for m in TRACKED_Y_MODES])
    )

    for block in range(1, num_blocks + 1):
        start = time.time()
        state = advance(state, step=step, steps=args.save_every)
        state.psi.block_until_ready()
        elapsed = time.time() - start

        time_value = block * args.save_every * args.dt
        psi_np = np.asarray(state.psi)
        if not np.isfinite(psi_np).all():
            print(f"NaN detected at block {block}; simulation crashed — stop.")
            break

        frame = render_frame(state, grid, time_value, crop)
        frames.append(frame)

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
            f"{elapsed:6.2f}s | max|Jz| = {float(np.max(jz)):.3e}"
        )

    # ---- Save outputs ------------------------------------------------------
    gif_path = outdir / "plasmoids.gif"
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"Saved {gif_path}")

    plt.imsave(outdir / "plasmoids_final.png", frames[-1])
    print(f"Saved {outdir / 'plasmoids_final.png'}")

    diag_path = outdir / "plasmoids_diagnostics.npz"
    np.savez(
        diag_path,
        time=np.array(t_history),
        kinetic_energy=np.array(ek_history),
        magnetic_energy=np.array(eb_history),
        jz_max=np.array(jzmax_history),
        y_mode_amplitudes=np.array(mode_history),
        y_modes=np.array(TRACKED_Y_MODES),
        nx=args.nx,
        ny=args.ny,
        lx=Lx,
        ly=Ly,
        width=args.width,
        eta=args.eta,
        eta4=args.eta4,
        seed=args.seed,
    )
    print(f"Saved {diag_path}")

    np.savez(
        outdir / "plasmoids_final_state.npz",
        psi=np.asarray(state.psi),
        omega=np.asarray(state.omega),
        nx=args.nx,
        ny=args.ny,
        lx=Lx,
        ly=Ly,
    )
    print(f"Saved {outdir / 'plasmoids_final_state.npz'}")

    # Summary plot.
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t_history, eb_history, "r-", label=r"$E_B$")
    axs[0].plot(t_history, ek_history, "b-", label=r"$E_K$")
    axs[0].set_ylabel("energy")
    axs[0].legend(frameon=False)
    axs[1].semilogy(t_history, jzmax_history, "k-")
    axs[1].set_ylabel(r"max $|J_z|$")
    axs[1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(outdir / "plasmoids_diagnostics.png", dpi=160)
    plt.close(fig)
    print(f"Saved {outdir / 'plasmoids_diagnostics.png'}")

    print("=" * 70)
    print(f"Done. Y-mode amplitudes grew to: "
          f"{dict(zip(TRACKED_Y_MODES, np.round(mode_history[-1], 6), strict=False))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
