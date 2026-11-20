"""Fluid-versus-kinetic comparison figure for the Vencels Orszag--Tang benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from mhx.benchmarks.vencels import load_vencels_reference, load_vencels_spectrum
from mhx.diagnostics import magnetic_power_spectrum_2d


def plot_kinetic_comparison(
    result: dict[str, Any], path: Path, csv_path: Path, domain_size: float
) -> Path:
    """Build and save the three-panel fluid-vs-kinetic comparison figure.

    Args:
        result: Output of :func:`mhx.benchmarks.vencels.run_orszag_tang`.
        path: PNG file to write.
        csv_path: Vencels reference CSV; reference curves are skipped if it is missing.
        domain_size: Box size, which is also the Alfven time ``t_A``.
    """
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

    k_1d, power_1d = magnetic_power_spectrum_2d(psi[time_idx], domain_size=domain_size)
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
