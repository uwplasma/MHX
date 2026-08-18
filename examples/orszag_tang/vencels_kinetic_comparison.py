"""Vencels kinetic comparison for the nonlinear Orszag--Tang vortex.

Runs the incompressible reduced-MHD Orszag--Tang vortex on a ``10 x 10`` domain
(matching the Vencels ``L_x = 10 d_i`` reference box), then produces the
three-panel kinetic comparison figure:

1. **Energy partitioning** — magnetic / kinetic / ``kinetic + dissipated``
   energy changes against the Vencels magnetic, electron, and ion reference
   curves read from ``vencelsdata.csv``.
2. **Magnetic power spectrum** — the 1D isotropic magnetic spectrum at
   ``t = 1.0 * t_A`` against the Vencels reference spectrum, with ``k d_e = 1``
   and ``k lambda_D = 1`` markers.
3. **Current density** — the 2D ``j_z`` sheet at ``t = 1.0 * t_A`` on the
   Vencels ``(e n_0 v_te)`` normalization.

The three-panel comparison code is intentionally self-contained in this
example: the Orszag--Tang initial state, the CSV loaders, and the figure
builder all live here, so the example runs standalone without importing
private helpers from other examples.

Usage
-----
    python examples/orszag_tang/vencels_kinetic_comparison.py --nx 24 --ny 24

The Vencels reference data must be present at ``examples/vencelsdata.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.interpolate import interp1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_divergence_linf, magnetic_energy
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4

# The Vencels box is ``L_x = 10 d_i`` so the Alfvén time is ``t_A = 10``.
DOMAIN_SIZE = 10.0
VENCELS_CSV = Path(__file__).resolve().parent.parent / "vencelsdata.csv"


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


def run_orszag_tang(
    *,
    shape: tuple[int, int],
    resistivity: float,
    viscosity: float,
    dt: float,
    t_end: float,
    save_every: int,
) -> dict[str, Any]:
    """Evolve the Orszag--Tang vortex and return time histories plus states."""
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=(DOMAIN_SIZE, DOMAIN_SIZE))
    )
    initial_state = orszag_tang_initial_state(grid)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    steps = int(round(t_end / dt))
    if steps % save_every != 0:
        raise ValueError("t_end / dt must be divisible by save_every")

    trajectory = evolve_rk4(
        initial_state, rhs, dt=dt, steps=steps, save_every=save_every
    )

    time = np.concatenate(([0.0], np.asarray(trajectory.times, dtype=np.float64)))
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
        for i in range(time.size)
    )
    magnetic = np.asarray(
        [float(magnetic_energy(s, lengths=grid.lengths)) for s in states],
        dtype=np.float64,
    )
    kinetic = np.asarray(
        [float(kinetic_energy(s, lengths=grid.lengths)) for s in states],
        dtype=np.float64,
    )
    total = magnetic + kinetic

    return {
        "time": time,
        "psi": psi,
        "omega": omega,
        "current_density": current,
        "magnetic_energy": magnetic,
        "kinetic_energy": kinetic,
        "total_energy": total,
        "final_divergence": float(magnetic_divergence_linf(states[-1], lengths=grid.lengths)),
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Vencels CSV loaders
# ---------------------------------------------------------------------------


def load_vencels_reference(csv_path: Path):
    """Parse the Vencels CSV and return interpolators for magnetic, electron, and ion energies."""
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


def load_vencels_spectrum(csv_path: Path):
    """Load the Vencels magnetic power spectrum reference points."""
    k_ref, power_ref = [], []
    if not csv_path.exists():
        return None, None
    with open(csv_path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            if not line.strip():
                continue
            cols = line.strip().split("\t")
            if len(cols) >= 5 and cols[3] and cols[4]:
                k_ref.append(float(cols[3]))
                power_ref.append(float(cols[4]))
    return np.array(k_ref), np.array(power_ref)


def compute_magnetic_power_spectrum(
    psi_field: np.ndarray, domain_size: float = DOMAIN_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1D isotropic magnetic power spectrum."""
    psi_hat = np.fft.fftn(psi_field)
    N = psi_field.shape[0]

    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    k_squared = kx_grid**2 + ky_grid**2
    physical_derivative_factor = (2.0 * np.pi / domain_size) ** 2
    power_B_2d = k_squared * physical_derivative_factor * np.abs(psi_hat) ** 2

    dx_dy = (domain_size / N) ** 2
    scaling_factor = dx_dy / (N**2)
    power_B_2d *= scaling_factor

    k_radius = np.round(np.sqrt(k_squared)).astype(int)
    k_max = N // 2
    k_1d = np.arange(1, k_max)
    power_1d = np.zeros(k_max - 1)

    dk = 2.0 * np.pi / domain_size
    for r in k_1d:
        power_1d[r - 1] = np.sum(power_B_2d[k_radius == r]) / dk

    return k_1d, power_1d


# ---------------------------------------------------------------------------
# Three-panel kinetic comparison figure
# ---------------------------------------------------------------------------


def write_kinetic_comparison_figure(
    result: dict[str, Any], path: Path, csv_path: Path, domain_size: float
) -> Path:
    """Build and save the three-panel fluid-vs-kinetic comparison figure."""
    import matplotlib.transforms as mtransforms

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)

    time = result["time"]
    magnetic = result["magnetic_energy"]
    kinetic = result["kinetic_energy"]
    total = result["total_energy"]
    psi = result["psi"]
    current = result["current_density"]

    # --- Panel 1: relative energy partitioning ---
    initial_total = max(float(total[0]), np.finfo(np.float64).tiny)
    delta_E_B = (magnetic - magnetic[0]) / initial_total
    delta_E_K = (kinetic - kinetic[0]) / initial_total
    delta_E_total = (total - total[0]) / initial_total

    vencels_time = time / domain_size  # t / t_A

    interp_B, interp_e, interp_i = load_vencels_reference(csv_path)
    if interp_B is not None:
        ref_B = interp_B(vencels_time)
        ref_e = interp_e(vencels_time)
        ref_i = interp_i(vencels_time)
        ref_total_particles = ref_e + ref_i

        axes[0].plot(
            vencels_time, ref_B, label="B (Vencels)",
            color="navy", linestyle=":", alpha=0.7,
        )
        axes[0].plot(
            vencels_time, ref_e, label="e (Vencels)",
            color="red", linestyle=":", alpha=0.7,
        )
        axes[0].plot(
            vencels_time, ref_i, label="i (Vencels)",
            color="green", linestyle=":", alpha=0.7,
        )
        axes[0].plot(vencels_time, ref_total_particles, label="e+i (Vencels)", color="purple",
                     linestyle="--", alpha=0.8)

    delta_E_particles = delta_E_K - delta_E_total  # kinetic + dissipated heat
    axes[0].plot(
        vencels_time, delta_E_B, label=r"$\delta E_B / W_0$ (Fluid)",
        color="navy", linewidth=2,
    )
    axes[0].plot(
        vencels_time, delta_E_K, label=r"$\delta E_K / W_0$ (Fluid)",
        color="crimson", linewidth=2,
    )
    axes[0].plot(vencels_time, delta_E_particles, label=r"$\delta E_{K+diss} / W_0$ (Fluid)",
                 color="purple", linewidth=2)

    axes[0].set_title("Energy Partitioning (Fluid vs Kinetic)")
    axes[0].set_xlabel(r"$t / t_A$")
    axes[0].set_ylabel(r"$\delta W / W_0$")
    axes[0].legend(frameon=False, fontsize="small")
    axes[0].set_xlim(0, 3)

    # --- Panel 2: 1D magnetic power spectrum ---
    target_time = 1.0 * domain_size
    time_idx = int(np.argmin(np.abs(time - target_time)))

    k_1d, power_1d = compute_magnetic_power_spectrum(psi[time_idx], domain_size=domain_size)
    k_1d_phys = k_1d * (2.0 * np.pi / domain_size)

    k_ref, power_ref = load_vencels_spectrum(csv_path)
    if k_ref is not None:
        axes[1].loglog(
            k_ref, power_ref, label="SPS Reference",
            color="red", linestyle=":", alpha=0.8,
        )

    axes[1].loglog(k_1d_phys, power_1d, label=f"MHX Fluid (t={time[time_idx]:.1f})",
                   color="teal", linewidth=2)

    axes[1].axvline(x=5.0, color="blue", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].axvline(x=30.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    trans = mtransforms.blended_transform_factory(axes[1].transData, axes[1].transAxes)
    axes[1].text(4.3, 0.05, r"$kd_e = 1$", rotation=90, color="black", transform=trans,
                 verticalalignment="bottom")
    axes[1].text(26.0, 0.05, r"$k\lambda_D = 1$", rotation=90, color="black", transform=trans,
                 verticalalignment="bottom")

    axes[1].set_title("Magnetic Power Spectrum")
    axes[1].set_xlabel(r"$k d_i$")
    axes[1].set_ylabel(r"$S_B$")
    axes[1].set_xlim(0.5, 200.0)
    axes[1].legend(frameon=False, fontsize="small")

    # --- Panel 3: 2D current density ---
    normalization_factor = np.sqrt(2.0 / (0.25 * 25.0))
    plot_current = current[time_idx] * normalization_factor

    im = axes[2].imshow(
        plot_current,
        cmap="jet",
        origin="lower",
        vmin=-0.3,
        vmax=0.3,
        extent=(0.0, 50.0, 0.0, 50.0),
    )
    axes[2].set_title(f"Current Density ($j_z$) at t={time[time_idx]:.2f}")
    axes[2].set_xlabel(r"$\mathrm{y/d_e}$")
    axes[2].set_ylabel(r"$\mathrm{x/d_e}$")
    axes[2].set_xticks([0, 50])
    axes[2].set_yticks([0, 50])
    fig.colorbar(im, ax=axes[2], shrink=0.75, ticks=[-0.3, 0.0, 0.3])

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/examples/orszag_tang_kinetic"))
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--eta", type=float, default=1.0e-2)
    parser.add_argument("--nu", type=float, default=1.0e-2)
    parser.add_argument("--dt", type=float, default=5.0e-3)
    parser.add_argument("--t-end", type=float, default=30.0)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not VENCELS_CSV.exists():
        raise FileNotFoundError(f"Could not find Vencels data at {VENCELS_CSV}")

    result = run_orszag_tang(
        shape=(args.nx, args.ny),
        resistivity=args.eta,
        viscosity=args.nu,
        dt=args.dt,
        t_end=args.t_end,
        save_every=args.save_every,
    )

    figure_path = write_kinetic_comparison_figure(
        result, outdir / "kinetic_comparison.png", VENCELS_CSV, DOMAIN_SIZE
    )

    diagnostics = {
        "schema": "mhx.example.orszag_tang_vencels_comparison.v1",
        "shape": [args.nx, args.ny],
        "domain": [0.0, DOMAIN_SIZE, 0.0, DOMAIN_SIZE],
        "resistivity": args.eta,
        "viscosity": args.nu,
        "dt": args.dt,
        "t_end": args.t_end,
        "final_magnetic_divergence_linf": result["final_divergence"],
        "vencels_csv": str(VENCELS_CSV),
    }
    manifest_path = write_manifest(
        outdir / "manifest.json",
        config=diagnostics,
        outputs={"kinetic_comparison": figure_path.name},
        claim_level="validation",
        claim_scope="Fluid-vs-kinetic Orszag--Tang comparison against the Vencels reference.",
    )
    print(f"wrote {manifest_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
