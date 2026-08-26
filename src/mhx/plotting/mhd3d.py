"""Paper-comparison figures for periodic incompressible 3D MHD."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from mhx.diagnostics.mhd3d import (
    alfven_collision_reference,
    collision_mode_histories,
    peak_window_spectra,
    trajectory_bulk_diagnostics,
)


def _power_law_slope(
    k: np.ndarray, spectrum: np.ndarray, start: float, stop: float
) -> float:
    mask = (
        (k >= start)
        & (k <= stop)
        & np.isfinite(spectrum)
        & (spectrum > 0.0)
    )
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(k[mask]), np.log(spectrum[mask]), 1)[0])


def plot_taylor_green_paper_comparison(
    results: Mapping[str, object],
    path: str | Path,
    *,
    peak_half_width: float = 0.25,
) -> tuple[Path, dict[str, dict[str, float]]]:
    """Reproduce the quantities in Lee et al. (2010), Figures 1, 2, and 6."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {"insulating": "#3266a8", "alternative": "#d97928", "conducting": "#26866d"}
    exponents = {"insulating": 2.0, "alternative": 5.0 / 3.0, "conducting": 3.0 / 2.0}
    labels = {"insulating": "I", "alternative": "A", "conducting": "C"}

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    metrics: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        bulk = trajectory_bulk_diagnostics(
            result.trajectory,
            shape=result.shape,
            viscosity=float(result.parameters.viscosity),
            resistivity=float(result.parameters.resistivity),
        )
        peak = peak_window_spectra(
            result.trajectory,
            shape=result.shape,
            dissipation=bulk["dissipation"],
            half_width=peak_half_width,
        )
        color = colors[name]
        label = labels[name]
        axes[0, 0].plot(bulk["time"], bulk["dissipation"], color=color, label=label)
        axes[0, 1].plot(
            bulk["time"],
            bulk["magnetic"] / bulk["kinetic"],
            color=color,
            label=label,
        )

        k = np.asarray(peak["k"])
        kinetic = np.asarray(peak["kinetic"])
        magnetic = np.asarray(peak["magnetic"])
        total = np.asarray(peak["total"])
        valid_ratio = (k > 0) & (kinetic > 0.0)
        axes[1, 0].semilogx(
            k[valid_ratio],
            magnetic[valid_ratio] / kinetic[valid_ratio],
            color=color,
            label=label,
        )

        exponent = exponents[name]
        valid = (k > 0) & (total > 0.0)
        fit_max = max(4, min(result.shape[0] // 6, int(k.max())))
        slope = _power_law_slope(k, total, 3, fit_max)
        compensated = k[valid] ** exponent * total[valid]
        axes[1, 1].loglog(
            k[valid], compensated, color=color,
            label=f"{label}: slope {slope:.2f} (target {-exponent:.2f})",
        )
        fit_mask = valid & (k >= 3) & (k <= fit_max)
        plateau = float(
            np.exp(np.mean(np.log(k[fit_mask] ** exponent * total[fit_mask])))
        )
        axes[1, 1].hlines(
            plateau, 3, fit_max, color=color, linestyle=":", linewidth=1.4
        )
        metrics[name] = {
            "peak_time": float(peak["peak_time"]),
            "peak_dissipation": float(np.max(bulk["dissipation"])),
            "spectral_slope": slope,
            "target_slope": -exponent,
            "fit_k_min": 3.0,
            "fit_k_max": float(fit_max),
            "peak_samples": float(peak["sample_count"]),
        }

    axes[0, 0].set_title("Paper Fig. 1(a): total dissipation")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(r"$\epsilon=\nu\langle\omega^2\rangle+\eta\langle j^2\rangle$")
    axes[0, 1].set_title("Paper Fig. 1(b): magnetic / kinetic energy")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(r"$E_M/E_V$")
    axes[1, 0].set_title("Paper Fig. 2: peak-window spectral ratio")
    axes[1, 0].set_xlabel(r"$k$")
    axes[1, 0].set_ylabel(r"$E_M(k)/E_V(k)$")
    axes[1, 1].set_title("Paper Fig. 6: class-compensated spectra")
    axes[1, 1].set_xlabel(r"$k$")
    axes[1, 1].set_ylabel("compensated total energy")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(
        "Taylor--Green MHD paper comparison\n"
        "I: target $k^{-2}$, A: $k^{-5/3}$, C: $k^{-3/2}$"
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output, metrics


def plot_alfven_collision_paper_comparison(
    result,
    path: str | Path,
    *,
    amplitude_plus: float,
    amplitude_minus: float,
    alfven_speed: float = 1.0,
) -> tuple[Path, dict[str, float]]:
    """Compare MHX modes with Howes--Nielson equations (36), (40), and (41)."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    modes = collision_mode_histories(result.trajectory, shape=result.shape)
    theory = alfven_collision_reference(
        modes["time"],
        amplitude_plus=amplitude_plus,
        amplitude_minus=amplitude_minus,
        alfven_speed=alfven_speed,
    )
    time = modes["time"]

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    axes[0, 0].plot(time, modes["secondary_magnetic"], label="MHX magnetic")
    axes[0, 0].plot(time, theory["secondary_magnetic"], "--", label="Eq. (36)")
    axes[0, 0].plot(time, modes["secondary_velocity"], ":", label="MHX velocity")
    axes[0, 0].set_title(r"Secondary $(1,1,0)$: purely magnetic, $2\omega_0$")

    marker_stride = max(1, len(time) // 24)
    axes[0, 1].plot(
        time, theory["tertiary_plus_magnetic"], "--", color="#3266a8",
        label=r"Eq. (40), $b_{(2,1,-1)}$",
    )
    axes[0, 1].plot(
        time, theory["tertiary_minus_magnetic"], "--", color="#d97928",
        label=r"Eq. (40), $b_{(1,2,1)}$",
    )
    axes[0, 1].plot(
        time, modes["tertiary_plus_magnetic"], color="#3266a8",
        marker="o", markevery=marker_stride, markersize=3, zorder=3,
        label=r"MHX $b_{(2,1,-1)}$",
    )
    axes[0, 1].plot(
        time, modes["tertiary_minus_magnetic"], color="#d97928",
        marker="s", markevery=marker_stride, markersize=3, zorder=3,
        label=r"MHX $b_{(1,2,1)}$",
    )
    axes[0, 1].set_title("Secular tertiary Alfvén waves")

    denominator = np.maximum(modes["secondary_magnetic"], np.finfo(float).tiny)
    axes[1, 0].semilogy(time, modes["secondary_velocity"] / denominator)
    axes[1, 0].set_title("Secondary velocity / magnetic amplitude")
    axes[1, 0].set_ylabel("purity ratio")

    tertiary_energy = (
        modes["tertiary_plus_magnetic"] ** 2
        + modes["tertiary_minus_magnetic"] ** 2
    )
    theory_energy = (
        theory["tertiary_plus_magnetic"] ** 2
        + theory["tertiary_minus_magnetic"] ** 2
    )
    secular_energy = (
        theory["tertiary_plus_secular_magnetic"] ** 2
        + theory["tertiary_minus_secular_magnetic"] ** 2
    )
    positive = (time > 0.0) & (tertiary_energy > 0.0)
    axes[1, 1].loglog(
        time[positive], tertiary_energy[positive], color="black", label="MHX"
    )
    axes[1, 1].loglog(
        time[positive], theory_energy[positive], "--", color="#7a4fa3",
        label="full Eq. (40)",
    )
    axes[1, 1].loglog(
        time[positive], secular_energy[positive], ":", color="#bf3030",
        label=r"Eq. (41) secular $t^2$",
    )
    axes[1, 1].set_title("Tertiary modal magnetic power")
    for axis in axes.flat:
        axis.set_xlabel("time")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(
        "Howes--Nielson Alfvén-wave collision\n"
        "primary → magnetic secondary → secular tertiary modes"
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)

    slope = _power_law_slope(time, tertiary_energy, 2.0 * np.pi, float(time[-1]))
    nonzero_secondary = modes["secondary_magnetic"] > 1.0e-14
    purity = (
        float(
            np.max(
                modes["secondary_velocity"][nonzero_secondary]
                / denominator[nonzero_secondary]
            )
        )
        if np.any(nonzero_secondary)
        else 0.0
    )
    metrics = {
        "tertiary_energy_slope_after_2pi": slope,
        "target_tertiary_energy_slope": 2.0,
        "tertiary_energy_relative_l2_vs_equation_40": float(
            np.linalg.norm(tertiary_energy - theory_energy)
            / np.linalg.norm(theory_energy)
        ),
        "secondary_relative_l2_vs_equation_36": float(
            np.linalg.norm(modes["secondary_magnetic"] - theory["secondary_magnetic"])
            / np.linalg.norm(theory["secondary_magnetic"])
        ),
        "max_secondary_velocity_to_magnetic": purity,
    }
    return output, metrics


__all__ = [
    "plot_alfven_collision_paper_comparison",
    "plot_taylor_green_paper_comparison",
]
