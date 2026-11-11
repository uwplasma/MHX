"""Analyse the turbulent magnetic power spectrum from a saved ``turbulent_spectrum.npz``.

Averages the spectrum over the time window ``T_MIN`` to ``T_MAX``, fits a power-law
slope over ``K_MIN`` to ``K_MAX``, and writes ``custom_spectrum_analysis.png`` next
to the input file. Takes seconds. Run ``benchmark_turbulent_spectrum.py`` first, then
from the repository root:

    python examples/turbulence/analyze_spectrum.py

To analyse a bigger run, point ``NPZ`` at its file and set ``K_MAX`` to about N/3.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mhx.benchmarks.turbulence import TURBULENCE_DOMAIN
from mhx.diagnostics import fit_power_law, magnetic_power_spectrum_2d

# --- Input parameters ------------------------------------------------------
NPZ = Path("outputs/examples/turbulence_2d/turbulent_spectrum.npz")  # saved run
T_MIN = 180.0  # start of the averaging window
T_MAX = 200.0  # end of the averaging window
K_MIN = 5.0  # lower edge of the fit range
K_MAX = 85.0  # upper edge of the fit range (N/3 for 256x256 is 85)

# --- Run -------------------------------------------------------------------
if not NPZ.exists():
    raise SystemExit(f"Could not find {NPZ}; run benchmark_turbulent_spectrum.py first.")

print(f"Loading {NPZ}...", flush=True)
with np.load(NPZ) as data:
    time_axis = data["time"]
    psi_history = data["psi"]

t_min, t_max = T_MIN, T_MAX
window_mask = (time_axis >= t_min) & (time_axis <= t_max)
valid_frames = psi_history[window_mask]
valid_times = time_axis[window_mask]

if len(valid_frames) == 0:
    print(f"Warning: No frames found between t={t_min} and t={t_max}.")
    print(f"Simulation only ran from t={time_axis[0]:.1f} to t={time_axis[-1]:.1f}.")
    print("Falling back to the final frame...")
    valid_frames = psi_history[-1:]
    valid_times = time_axis[-1:]
    t_min = t_max = valid_times[0]

print(
    f"Averaging spectrum over {len(valid_frames)} frames from t={valid_times[0]:.1f} "
    f"to t={valid_times[-1]:.1f}...",
    flush=True,
)

power_1d_sum = None
k_1d = None
for psi_frame in valid_frames:
    k_out, p_out = magnetic_power_spectrum_2d(psi_frame, TURBULENCE_DOMAIN[0])
    if power_1d_sum is None:
        power_1d_sum, k_1d = p_out, k_out
    else:
        power_1d_sum += p_out
power_1d = power_1d_sum / len(valid_frames)

mask = (k_1d >= K_MIN) & (k_1d <= K_MAX)
k_fit = k_1d[mask]
slope, intercept = fit_power_law(k_1d, power_1d, kmin=K_MIN, kmax=K_MAX)

# --- Results ---------------------------------------------------------------
print("-" * 50)
print(f"Fit Range: k = [{K_MIN}, {K_MAX}]")
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
ax.axvspan(K_MIN, K_MAX, color="red", alpha=0.1, label="Fit Region")
ax.set_title("Turbulent Magnetic Power Spectrum Analysis", fontsize=14)
ax.set_xlabel(r"Wavenumber $k$", fontsize=12)
ax.set_ylabel(r"Spectral Energy Density $S_B(k)$", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

out_path = NPZ.parent / "custom_spectrum_analysis.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved custom plot to: {out_path}")
