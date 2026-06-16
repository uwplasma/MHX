"""Publication-style Kelvin--Helmholtz validation and differentiation-media example.

Edit the parameters below, then run:

    MHX_EXAMPLE_FAST=1 python examples/publication_kelvin_helmholtz_validation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from mhx.benchmarks import (
    CompressibleKelvinHelmholtzConfig,
    KelvinHelmholtzConfig,
    write_kelvin_helmholtz_validation,
)
from mhx.io import write_manifest
from mhx.runtime import configure_jax

CI_SMOKE = False
FAST_MODE = os.environ.get("MHX_EXAMPLE_FAST", "0") == "1"
PRESET = os.environ.get("MHX_KH_PRESET", "fast" if FAST_MODE else "validation")
OUTDIR_ROOT = Path(
    os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples/publication")
).expanduser()
RUN_DIR = OUTDIR_ROOT / "kelvin_helmholtz_validation"

PRIMARY_SHAPE = (16, 32) if PRESET == "fast" else (64, 128)
COMPARISON_SHAPE = (12, 24) if PRESET == "fast" else (32, 64)
COMPRESSIBLE_SHAPE = (12, 24) if PRESET == "fast" else (24, 48)
DT = 2.0e-3
T_END = 0.12 if PRESET == "fast" else 2.0
SAVE_EVERY = 10 if PRESET == "fast" else 50
COMPRESSIBLE_DT = 5.0e-4
COMPRESSIBLE_T_END = 0.01 if PRESET == "fast" else 0.04
COMPRESSIBLE_SAVE_EVERY = 10 if PRESET == "fast" else 20
MIN_ENTROPY_GAIN = 1.0e-4 if PRESET == "fast" else 1.0e-2
MAX_RESOLUTION_ENTROPY_RDIFF = 5.0e-2 if PRESET == "fast" else 1.0e-2
MAX_DYE_OVERSHOOT = 2.0e-2
WRITE_MOVIES = True
FIGURE_DPI = 220

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if PRESET not in {"fast", "validation"}:
    raise ValueError("MHX_KH_PRESET must be 'fast' or 'validation'")

configure_jax(enable_x64=True)
primary_config = KelvinHelmholtzConfig(
    shape=PRIMARY_SHAPE,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
)
comparison_config = KelvinHelmholtzConfig(
    shape=COMPARISON_SHAPE,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
)
compressible_config = CompressibleKelvinHelmholtzConfig(
    shape=COMPRESSIBLE_SHAPE,
    dt=COMPRESSIBLE_DT,
    t_end=COMPRESSIBLE_T_END,
    save_every=COMPRESSIBLE_SAVE_EVERY,
)
manifest_path, validation = write_kelvin_helmholtz_validation(
    RUN_DIR,
    movies=WRITE_MOVIES,
    primary_config=primary_config,
    comparison_config=comparison_config,
    compressible_config=compressible_config,
    min_entropy_gain=MIN_ENTROPY_GAIN,
    max_resolution_entropy_rdiff=MAX_RESOLUTION_ENTROPY_RDIFF,
    max_dye_overshoot=MAX_DYE_OVERSHOOT,
    min_saved_samples=3 if PRESET == "fast" else 10,
)

diagnostics = json.loads((RUN_DIR / "diagnostics.json").read_text(encoding="utf-8"))
with np.load(RUN_DIR / "kelvin_helmholtz_incompressible.npz", allow_pickle=False) as history:
    time = np.asarray(history["time"], dtype=float)
    entropy = np.asarray(history["entropy"], dtype=float)
    dye = np.asarray(history["dye"], dtype=float)
    omega = np.asarray(history["omega"], dtype=float)
with np.load(
    RUN_DIR / "kelvin_helmholtz_resolution_comparison.npz",
    allow_pickle=False,
) as comparison:
    comparison_time = np.asarray(comparison["time"], dtype=float)
    comparison_entropy = np.asarray(comparison["entropy"], dtype=float)
with np.load(
    RUN_DIR / "kelvin_helmholtz_compressible_mhd.npz",
    allow_pickle=False,
) as compressible:
    compressible_time = np.asarray(compressible["time"], dtype=float)
    density_min = np.asarray(compressible["density_min"], dtype=float)
    pressure_min = np.asarray(compressible["pressure_min"], dtype=float)

summary_path = RUN_DIR / "figures" / "publication_kelvin_helmholtz_summary.png"
summary_path.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)

axes[0, 0].plot(time, entropy, "o-", label=f"primary {PRIMARY_SHAPE}")
axes[0, 0].plot(comparison_time, comparison_entropy, "s--", label=f"comparison {COMPARISON_SHAPE}")
axes[0, 0].set_title("Passive-dye entropy gate")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel(r"$\int -c\log c\,dA$")
axes[0, 0].grid(True, alpha=0.25)
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(compressible_time, density_min, "o-", label=r"$\min\rho$")
axes[0, 1].plot(compressible_time, pressure_min, "s--", label=r"$\min p$")
axes[0, 1].set_title("Smooth compressible-MHD positivity")
axes[0, 1].set_xlabel("time")
axes[0, 1].grid(True, alpha=0.25)
axes[0, 1].legend(frameon=False)

axes[0, 2].axis("off")
summary_lines = [
    f"validation passed: {validation['passed']}",
    f"preset: {PRESET}",
    f"primary shape: {PRIMARY_SHAPE}, t_end={T_END:g}",
    f"entropy gain: {diagnostics['primary_entropy_gain']:.3e}",
    (
        "resolution entropy relative difference: "
        f"{diagnostics['resolution_entropy_relative_difference']:.3e}"
    ),
    f"dye overshoot: {diagnostics['primary_dye_overshoot']:.3e}",
]
axes[0, 2].text(
    0.02,
    0.98,
    "\n".join(summary_lines),
    va="top",
    ha="left",
    transform=axes[0, 2].transAxes,
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
)

frame_indices = [0, dye.shape[0] // 2, dye.shape[0] - 1]
dye_min = float(np.min(dye))
dye_max = float(np.max(dye))
omega_limit = max(float(np.percentile(np.abs(omega), 99.5)), np.finfo(float).eps)
for column, frame_index in enumerate(frame_indices):
    image = axes[1, column].imshow(
        dye[frame_index].T,
        origin="lower",
        cmap="viridis",
        vmin=dye_min,
        vmax=dye_max,
    )
    axes[1, column].contour(
        omega[frame_index].T,
        levels=np.linspace(-omega_limit, omega_limit, 13),
        colors="white",
        linewidths=0.45,
        alpha=0.75,
    )
    axes[1, column].set_title(f"dye + vorticity contours, t={time[frame_index]:.2f}")
    axes[1, column].set_xlabel("grid x")
    axes[1, column].set_ylabel("grid y")
    figure.colorbar(image, ax=axes[1, column], shrink=0.72)

figure.suptitle("MHX publication example: Kelvin--Helmholtz validation bundle", fontsize=14)
figure.savefig(summary_path, dpi=FIGURE_DPI)
plt.close(figure)

write_manifest(
    manifest_path,
    config={
        **diagnostics,
        "fast_mode": FAST_MODE,
        "preset": PRESET,
    },
    outputs={
        "diagnostics": "diagnostics.json",
        "validation": "validation.json",
        "incompressible_history": "kelvin_helmholtz_incompressible.npz",
        "resolution_comparison": "kelvin_helmholtz_resolution_comparison.npz",
        "compressible_history": "kelvin_helmholtz_compressible_mhd.npz",
        "entropy_figure": "figures/kelvin_helmholtz_entropy.png",
        "snapshot_figure": "figures/kelvin_helmholtz_snapshots.png",
        "compressible_minima_figure": "figures/kelvin_helmholtz_compressible_minima.png",
        "dye_movie": "figures/kelvin_helmholtz_dye.gif",
        "publication_summary": "figures/publication_kelvin_helmholtz_summary.png",
    },
    claim_level="validation",
    claim_scope=(
        "Standalone Kelvin--Helmholtz validation example. The gates cover "
        "finite histories, entropy response, resolution consistency, dye bounds, "
        "and smooth compressible-MHD positivity; they do not claim shock-capturing "
        "production compressible-MHD accuracy."
    ),
)

print(f"manifest: {manifest_path}")
print(f"validation_passed: {validation['passed']}")
print(f"summary_figure: {summary_path}")
print(f"entropy_figure: {RUN_DIR / 'figures' / 'kelvin_helmholtz_entropy.png'}")
print(f"snapshot_figure: {RUN_DIR / 'figures' / 'kelvin_helmholtz_snapshots.png'}")
print(f"dye_movie: {RUN_DIR / 'figures' / 'kelvin_helmholtz_dye.gif'}")
