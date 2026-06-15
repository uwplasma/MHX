"""Publication-style FAST Harris tearing example.

Edit the parameters below, then run:

    python examples/publication_linear_harris_tearing.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from mhx.benchmarks import write_linear_tearing_timedomain_validation
from mhx.io import write_manifest
from mhx.runtime import configure_jax

CI_SMOKE = True
OUTDIR_ROOT = Path(
    os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples/publication")
).expanduser()
RUN_DIR = OUTDIR_ROOT / "linear_harris_tearing"

GRID_POINTS = 96
HALF_WIDTH = 10.0
LUNDQUIST = 1000.0
WAVENUMBER = 0.5
DT = 0.25
T_END = 24.0
FIT_START_FRACTION = 0.25
FIGURE_DPI = 220

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

configure_jax(enable_x64=True)
manifest_path, validation = write_linear_tearing_timedomain_validation(
    RUN_DIR,
    grid_points=GRID_POINTS,
    half_width=HALF_WIDTH,
    lundquist=LUNDQUIST,
    wavenumber=WAVENUMBER,
    dt=DT,
    t_end=T_END,
    fit_start_fraction=FIT_START_FRACTION,
)

diagnostics = json.loads((RUN_DIR / "diagnostics.json").read_text(encoding="utf-8"))
with np.load(RUN_DIR / "linear_tearing_timedomain.npz", allow_pickle=False) as history:
    time = np.asarray(history["time"], dtype=float)
    amplitude = np.asarray(history["amplitude"], dtype=float)
    exact_amplitude = np.asarray(history["exact_amplitude"], dtype=float)
    relative_error = np.asarray(history["relative_amplitude_error"], dtype=float)

summary_path = RUN_DIR / "figures" / "publication_linear_harris_tearing_summary.png"
summary_path.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

axes[0, 0].semilogy(time, amplitude, "o", ms=3.5, label="RK4 replay")
axes[0, 0].semilogy(time, exact_amplitude, "-", lw=2.0, label="eigenvalue prediction")
axes[0, 0].set_title("Linear Harris tearing growth")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel("mode amplitude")
axes[0, 0].grid(True, alpha=0.25)
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(time, relative_error, color="tab:red", lw=2.0)
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Time-domain replay error")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("relative amplitude error")
axes[0, 1].grid(True, alpha=0.25)

check_items = tuple(validation["checks"].items())
check_labels = [name.replace("_", "\n") for name, _ in check_items]
check_values = [1.0 if passed else 0.0 for _, passed in check_items]
bar_colors = ["#2ca02c" if passed else "#d62728" for _, passed in check_items]
axes[1, 0].barh(check_labels, check_values, color=bar_colors)
axes[1, 0].set_xlim(0.0, 1.05)
axes[1, 0].set_title("Validation gates")
axes[1, 0].set_xlabel("pass = 1")
axes[1, 0].grid(True, axis="x", alpha=0.25)

axes[1, 1].axis("off")
summary_lines = [
    "Direct finite-difference Harris eigenmode",
    rf"$S={LUNDQUIST:g}$, $ka={WAVENUMBER:g}$, $N_x={GRID_POINTS}$",
    rf"$\gamma_\mathrm{{eig}}={diagnostics['selected_eigenvalue']['real']:.6f}$",
    rf"$\gamma_\mathrm{{fit}}={diagnostics['fitted_growth_rate']:.6f}$",
    rf"relative growth error = {diagnostics['relative_growth_error']:.2e}",
    rf"final mode alignment = {diagnostics['final_mode_alignment']:.8f}",
]
axes[1, 1].text(
    0.02,
    0.98,
    "\n".join(summary_lines),
    va="top",
    ha="left",
    transform=axes[1, 1].transAxes,
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
)

figure.suptitle("MHX publication example: linear Harris tearing", fontsize=14)
figure.savefig(summary_path, dpi=FIGURE_DPI)
plt.close(figure)

write_manifest(
    manifest_path,
    config=diagnostics,
    outputs={
        "diagnostics": "diagnostics.json",
        "validation": "validation.json",
        "history": "linear_tearing_timedomain.npz",
        "linear_tearing_timedomain": "figures/linear_tearing_timedomain.png",
        "publication_summary": "figures/publication_linear_harris_tearing_summary.png",
    },
    claim_level="validation",
    claim_scope=(
        "Standalone publication-style Harris tearing replay with eigenvalue, "
        "time-domain growth, and gate summary panels."
    ),
)

print(f"manifest: {manifest_path}")
print(f"validation_passed: {validation['passed']}")
print(f"summary_figure: {summary_path}")
print(f"benchmark_figure: {RUN_DIR / 'figures' / 'linear_tearing_timedomain.png'}")
print(f"history: {RUN_DIR / 'linear_tearing_timedomain.npz'}")
