"""Publication-style FAST Orszag--Tang and turbulence media example.

Edit the parameters below, then run:

    python examples/publication_orszag_tang_turbulence.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np

from mhx.benchmarks import (
    write_forced_turbulent_reconnection_validation,
    write_orszag_tang_vortex_validation,
)
from mhx.io import write_manifest
from mhx.runtime import configure_jax

CI_SMOKE = False
FAST_MODE = os.environ.get("MHX_EXAMPLE_FAST", "0") == "1"
OUTDIR_ROOT = Path(
    os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples/publication")
).expanduser()
RUN_DIR = OUTDIR_ROOT / "orszag_tang_turbulence"
ORSZAG_TANG_DIR = RUN_DIR / "orszag_tang"
TURBULENCE_DIR = RUN_DIR / "forced_turbulent_reconnection"

ORSZAG_TANG_SHAPE = (16, 16) if FAST_MODE else (64, 64)
ORSZAG_TANG_DT = 1.0e-2
ORSZAG_TANG_T_END = 1.0 if FAST_MODE else 8.0
ORSZAG_TANG_SAVE_EVERY = 10 if FAST_MODE else 50

TURBULENCE_SHAPE = (16, 16) if FAST_MODE else (64, 64)
TURBULENCE_DT = 2.0e-2
TURBULENCE_T_END = 2.0 if FAST_MODE else 20.0
TURBULENCE_SAVE_EVERY = 10 if FAST_MODE else 50
WRITE_MOVIES = True
FIGURE_DPI = 220

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

configure_jax(enable_x64=True)
orszag_manifest_path, orszag_validation = write_orszag_tang_vortex_validation(
    ORSZAG_TANG_DIR,
    movies=WRITE_MOVIES,
    shape=ORSZAG_TANG_SHAPE,
    dt=ORSZAG_TANG_DT,
    t_end=ORSZAG_TANG_T_END,
    save_every=ORSZAG_TANG_SAVE_EVERY,
    min_relative_energy_drop=1.0e-4,
    min_current_high_k_growth=0.0,
    min_vorticity_high_k_growth=0.0,
)
turbulence_manifest_path, turbulence_validation = write_forced_turbulent_reconnection_validation(
    TURBULENCE_DIR,
    movies=WRITE_MOVIES,
    shape=TURBULENCE_SHAPE,
    dt=TURBULENCE_DT,
    t_end=TURBULENCE_T_END,
    save_every=TURBULENCE_SAVE_EVERY,
    min_reconnection_proxy_change=1.0e-8,
    min_current_linf_growth=0.0,
)

with np.load(ORSZAG_TANG_DIR / "orszag_tang_vortex.npz", allow_pickle=False) as history:
    orszag_time = np.asarray(history["time"], dtype=float)
    orszag_total_energy = np.asarray(history["total_energy"], dtype=float)
    orszag_current_fraction = np.asarray(history["current_high_k_fraction"], dtype=float)
    orszag_vorticity_fraction = np.asarray(history["vorticity_high_k_fraction"], dtype=float)
    orszag_final_current = np.asarray(history["current_density"][-1], dtype=float)

with np.load(
    TURBULENCE_DIR / "forced_turbulent_reconnection.npz",
    allow_pickle=False,
) as history:
    turbulence_time = np.asarray(history["time"], dtype=float)
    turbulence_total_energy = np.asarray(history["total_energy"], dtype=float)
    turbulence_current_fraction = np.asarray(history["current_high_k_fraction"], dtype=float)
    turbulence_reconnection = np.asarray(history["reconnection_proxy"], dtype=float)
    turbulence_rate = np.asarray(history["reconnection_rate_proxy"], dtype=float)
    turbulence_final_current = np.asarray(history["current_density"][-1], dtype=float)

summary_path = RUN_DIR / "figures" / "publication_orszag_tang_turbulence_summary.png"
summary_path.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)

axes[0, 0].plot(
    orszag_time,
    orszag_total_energy / orszag_total_energy[0],
    label="Orszag--Tang",
)
axes[0, 0].plot(
    turbulence_time,
    turbulence_total_energy / turbulence_total_energy[0],
    label="forced turbulence",
)
axes[0, 0].set_title("Normalized total energy")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel(r"$E(t)/E(0)$")
axes[0, 0].grid(True, alpha=0.25)
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(orszag_time, orszag_current_fraction, label="OT current")
axes[0, 1].plot(orszag_time, orszag_vorticity_fraction, label="OT vorticity")
axes[0, 1].plot(turbulence_time, turbulence_current_fraction, label="turbulent current")
axes[0, 1].set_title("High-wavenumber content")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("spectral fraction")
axes[0, 1].grid(True, alpha=0.25)
axes[0, 1].legend(frameon=False, fontsize="small")

axes[0, 2].plot(turbulence_time, turbulence_reconnection, label=r"$\psi_{rec}$ proxy")
rate_axis = axes[0, 2].twinx()
rate_axis.plot(turbulence_time, turbulence_rate, ":", color="tab:red", label=r"$d\psi_{rec}/dt$")
axes[0, 2].set_title("Forced turbulent reconnection proxy")
axes[0, 2].set_xlabel("time")
axes[0, 2].set_ylabel("flux separation")
rate_axis.set_ylabel("proxy rate")
axes[0, 2].grid(True, alpha=0.25)
handles, labels = axes[0, 2].get_legend_handles_labels()
extra_handles, extra_labels = rate_axis.get_legend_handles_labels()
axes[0, 2].legend(handles + extra_handles, labels + extra_labels, frameon=False)

current_limit = max(
    float(np.max(np.abs(orszag_final_current))),
    float(np.max(np.abs(turbulence_final_current))),
    np.finfo(float).eps,
)
orszag_image = axes[1, 0].imshow(
    orszag_final_current.T,
    origin="lower",
    cmap="RdBu_r",
    vmin=-current_limit,
    vmax=current_limit,
)
axes[1, 0].set_title("Orszag--Tang final current")
axes[1, 0].set_xlabel("grid x")
axes[1, 0].set_ylabel("grid y")
figure.colorbar(orszag_image, ax=axes[1, 0], shrink=0.75)

turbulence_image = axes[1, 1].imshow(
    turbulence_final_current.T,
    origin="lower",
    cmap="RdBu_r",
    vmin=-current_limit,
    vmax=current_limit,
)
axes[1, 1].set_title("Forced turbulence final current")
axes[1, 1].set_xlabel("grid x")
axes[1, 1].set_ylabel("grid y")
figure.colorbar(turbulence_image, ax=axes[1, 1], shrink=0.75)

axes[1, 2].axis("off")
summary_lines = [
    f"Orszag--Tang validation passed: {orszag_validation['passed']}",
    f"Turbulence validation passed: {turbulence_validation['passed']}",
    f"OT shape: {ORSZAG_TANG_SHAPE}, t_end={ORSZAG_TANG_T_END:g}",
    f"turbulence shape: {TURBULENCE_SHAPE}, t_end={TURBULENCE_T_END:g}",
    "Generated movies: OT flux/current/vorticity and turbulent flux/current",
]
axes[1, 2].text(
    0.02,
    0.98,
    "\n".join(summary_lines),
    va="top",
    ha="left",
    transform=axes[1, 2].transAxes,
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
)

figure.suptitle("MHX publication example: Orszag--Tang cascade and turbulence", fontsize=14)
figure.savefig(summary_path, dpi=FIGURE_DPI)
plt.close(figure)

write_manifest(
    RUN_DIR / "manifest.json",
    config={
        "orszag_tang_shape": list(ORSZAG_TANG_SHAPE),
        "orszag_tang_t_end": ORSZAG_TANG_T_END,
        "turbulence_shape": list(TURBULENCE_SHAPE),
        "turbulence_t_end": TURBULENCE_T_END,
        "fast_mode": FAST_MODE,
    },
    outputs={
        "orszag_tang_manifest": "orszag_tang/manifest.json",
        "forced_turbulent_reconnection_manifest": ("forced_turbulent_reconnection/manifest.json"),
        "publication_summary": "figures/publication_orszag_tang_turbulence_summary.png",
    },
    claim_level="validation",
    claim_scope=(
        "Standalone publication-style nonlinear reduced-MHD examples: "
        "Orszag--Tang cascade plus forced turbulent-reconnection proxy media."
    ),
)

print(f"orszag_tang_manifest: {orszag_manifest_path}")
print(f"orszag_tang_validation_passed: {orszag_validation['passed']}")
print(f"turbulence_manifest: {turbulence_manifest_path}")
print(f"turbulence_validation_passed: {turbulence_validation['passed']}")
print(f"summary_figure: {summary_path}")
print(f"orszag_tang_current_movie: {ORSZAG_TANG_DIR / 'figures' / 'orszag_tang_current.gif'}")
print(
    "turbulence_current_movie: "
    f"{TURBULENCE_DIR / 'figures' / 'forced_turbulent_reconnection_current.gif'}"
)
