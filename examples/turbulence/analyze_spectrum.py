"""Analyse the turbulent magnetic power spectrum from a saved ``turbulent_spectrum.npz``.

Averages the spectrum over a configurable time window, fits a power-law slope
over a configurable ``k`` range, and writes ``custom_spectrum_analysis.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mhx.benchmarks.turbulence import TURBULENCE_DOMAIN


def analyze_spectrum() -> None:
    # This script lives at examples/turbulence/ and is also run standalone.
    # Ensure the repo root is importable so the sibling `examples` package
    # resolves regardless of the current working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from examples.turbulence.benchmark_turbulent_spectrum import (
        _compute_magnetic_power_spectrum,
    )

    root_dir = Path(__file__).resolve().parent.parent.parent

    npz_path = root_dir / "outputs" / "turbulent_spectrum_output" / "turbulent_spectrum.npz"
    if not npz_path.exists():
        print(f"Error: Could not find {npz_path}")
        print("Make sure the turbulent_spectrum.npz file exists first.")
        return

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    time = data["time"]
    psi_history = data["psi"]

    # --- Configure averaging & fit range here ---
    t_min = 180.0
    t_max = 200.0
    k_min = 5.0
    k_max = 85.0  # N/3 for 256x256 is 85

    domain_size = TURBULENCE_DOMAIN[0]

    window_mask = (time >= t_min) & (time <= t_max)
    valid_frames = psi_history[window_mask]
    valid_times = time[window_mask]

    if len(valid_frames) == 0:
        print(f"Warning: No frames found between t={t_min} and t={t_max}.")
        print(f"Simulation only ran from t={time[0]:.1f} to t={time[-1]:.1f}.")
        print("Falling back to the final frame...")
        valid_frames = psi_history[-1:]
        valid_times = time[-1:]
        t_min = valid_times[0]
        t_max = valid_times[0]

    print(f"Averaging spectrum over {len(valid_frames)} frames from t={valid_times[0]:.1f} "
          f"to t={valid_times[-1]:.1f}...")

    power_1d_sum = None
    k_1d = None
    for psi_frame in valid_frames:
        k_out, p_out = _compute_magnetic_power_spectrum(psi_frame, domain_size)
        if power_1d_sum is None:
            power_1d_sum = p_out
            k_1d = k_out
        else:
            power_1d_sum += p_out

    power_1d = power_1d_sum / len(valid_frames)

    mask = (k_1d >= k_min) & (k_1d <= k_max)
    k_fit = k_1d[mask]
    power_fit = power_1d[mask]

    slope, intercept = np.polyfit(np.log10(k_fit), np.log10(power_fit), 1)

    print("-" * 50)
    print(f"Fit Range: k = [{k_min}, {k_max}]")
    print(f"Calculated Slope: {slope:.3f}")
    print("-" * 50)

    fig, ax = plt.subplots(figsize=(8, 6))

    label = (
        f"Time-Averaged Spectrum\n(t = {t_min:.1f} to {t_max:.1f})"
        if t_min != t_max
        else "Magnetic Power Spectrum"
    )
    ax.loglog(k_1d, power_1d, label=label, color="teal", linewidth=2)

    fit_line = (10**intercept) * (k_fit**slope)
    ax.loglog(k_fit, fit_line, color="red", linewidth=3, linestyle="--",
              label=f"Fit (Slope: {slope:.3f})")

    ax.axvspan(k_min, k_max, color="red", alpha=0.1, label="Fit Region")

    ax.set_title("Turbulent Magnetic Power Spectrum Analysis", fontsize=14)
    ax.set_xlabel(r"Wavenumber $k$", fontsize=12)
    ax.set_ylabel(r"Spectral Energy Density $S_B(k)$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    out_path = root_dir / "outputs" / "turbulent_spectrum_output" / "custom_spectrum_analysis.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved custom plot to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    analyze_spectrum()
