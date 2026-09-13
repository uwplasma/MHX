"""Kelvin--Helmholtz with smooth compressible MHD.

A small, smooth, periodic compressible-MHD variant of the Kelvin--Helmholtz
example.  Designed for pedagogy and local execution -- **not** a
shock-capturing production MHD solver.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.kelvin_helmholtz import (
    CompressibleKelvinHelmholtzConfig,
    CompressibleMHDParams,
    compressible_kelvin_helmholtz_grid,
    compressible_kelvin_helmholtz_initial_state,
    primitive_from_conservative,
    run_compressible_kelvin_helmholtz,
)
from mhx.runtime import configure_jax


def main() -> None:
    configure_jax(enable_x64=True)

    output_root = Path(os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples"))
    outdir = output_root / "kelvin_helmholtz_compressible_mhd"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {outdir.resolve()}")

    # Inputs: low-Mach and intentionally short.  Increase ``shape``, reduce
    # ``dt``, and document convergence before interpreting nonlinear
    # compressible-MHD dynamics.
    config = CompressibleKelvinHelmholtzConfig(
        shape=(24, 48),
        dt=5.0e-4,
        t_end=1.0e-2,
        save_every=10,
        density=1.0,
        pressure=10.0,
        magnetic_field=(0.1, 0.0),
        flow_speed=0.2,
        perturbation_amplitude=1.0e-2,
    )
    print(config)

    # Inspect the initial condition.
    grid = compressible_kelvin_helmholtz_grid(config)
    params = CompressibleMHDParams(gamma=config.gamma)
    state0 = compressible_kelvin_helmholtz_initial_state(grid, config)
    primitive0 = primitive_from_conservative(state0, params)

    print("grid shape:", grid.shape)
    density0 = np.asarray(primitive0.density)
    print("density range:", float(np.min(density0)), float(np.max(density0)))
    pressure0 = np.asarray(primitive0.pressure)
    print("pressure range:", float(np.min(pressure0)), float(np.max(pressure0)))
    velocity_x0 = np.asarray(primitive0.velocity_x)
    print("velocity_x range:", float(np.min(velocity_x0)), float(np.max(velocity_x0)))
    print("magnetic field:", config.magnetic_field)

    # Run the smooth compressible-MHD KH example.
    result = run_compressible_kelvin_helmholtz(config)
    result.trajectory.times.block_until_ready()
    final_primitive = primitive_from_conservative(result.final_state, result.params)

    print("saved times:", np.asarray(result.trajectory.times))
    print("dye entropy:", np.asarray(result.dye_entropy))
    print("density min history:", np.asarray(result.density_min))
    print("pressure min history:", np.asarray(result.pressure_min))
    final_dye = np.asarray(final_primitive.dye)
    print("final dye range:", float(np.min(final_dye)), float(np.max(final_dye)))

    # Plot outputs: initial/final dye, final density and pressure, and dye
    # entropy history.
    extent = (config.lower[0], config.upper[0], config.lower[1], config.upper[1])
    initial_dye = np.asarray(primitive0.dye)
    final_density = np.asarray(final_primitive.density)
    final_pressure = np.asarray(final_primitive.pressure)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    t_initial = float(result.trajectory.times[0])
    t_final = float(result.trajectory.times[-1])

    im0 = axes[0].imshow(
        initial_dye.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=0.0, vmax=1.0
    )
    axes[0].set_title(f"Initial Dye Concentration, t={t_initial:.3f}")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Concentration (c)")

    im1 = axes[1].imshow(
        final_dye.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=0.0, vmax=1.0
    )
    axes[1].set_title(f"Final Dye Concentration, t={t_final:.3f}")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Concentration (c)")

    im2 = axes[2].imshow(final_density.T, origin="lower", extent=extent, cmap="viridis")
    axes[2].set_title(f"Final Density, t={t_final:.3f}")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label=r"Density ($\rho$)")

    im3 = axes[3].imshow(final_pressure.T, origin="lower", extent=extent, cmap="magma")
    axes[3].set_title(f"Final Pressure, t={t_final:.3f}")
    fig.colorbar(im3, ax=axes[3], shrink=0.8, label="Pressure (P)")

    for ax in axes:
        ax.set_xlabel("x/Lx")
        ax.set_ylabel("y/Ly")

    fig.savefig(outdir / "kh_compressible_mhd_snapshots.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(np.asarray(result.trajectory.times), np.asarray(result.dye_entropy), marker="o")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Dye Entropy")
    ax.set_title("Smooth Compressible-MHD KH Dye Entropy")
    ax.grid(alpha=0.3)
    fig.savefig(outdir / "kh_compressible_mhd_entropy.png", dpi=180)
    plt.close(fig)

    print("wrote", outdir / "kh_compressible_mhd_snapshots.png")
    print("wrote", outdir / "kh_compressible_mhd_entropy.png")


if __name__ == "__main__":
    main()
