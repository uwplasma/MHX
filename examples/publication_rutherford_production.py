"""Publication-style FAST Rutherford production-path example.

Edit the parameters below, then run:

    python examples/publication_rutherford_production.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from mhx.campaigns import (
    WalltimePolicy,
    write_rutherford_production_execution,
    write_rutherford_production_plan,
    write_rutherford_resume_plan,
)
from mhx.io import write_manifest
from mhx.runtime import configure_jax

CI_SMOKE = False
FAST_MODE = os.environ.get("MHX_EXAMPLE_FAST", "0") == "1"
OUTDIR_ROOT = Path(
    os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples/publication")
).expanduser()
RUN_DIR = OUTDIR_ROOT / "rutherford_production_path"

SHAPE = (8, 8) if FAST_MODE else (32, 32)
DT = 1.0e-2 if FAST_MODE else 5.0e-2
TARGET_SAVED_FRAMES = 120 if FAST_MODE else 240
MIN_PRODUCTION_RESOLUTION = 8 if FAST_MODE else 32
MAX_STEPS_THIS_CHUNK = 6 if FAST_MODE else 120
WRITE_MOVIES = True
FIGURE_DPI = 220

WALLTIME_POLICY = WalltimePolicy(
    max_walltime_hours=1.0,
    seconds_per_step_estimate=0.1,
    checkpoint_interval_minutes=1.0,
    preemption_margin_minutes=1.0,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

configure_jax(enable_x64=True)
plan_manifest_path, plan_validation = write_rutherford_production_plan(
    RUN_DIR,
    shape=SHAPE,
    dt=DT,
    target_saved_frames=TARGET_SAVED_FRAMES,
    min_production_resolution=MIN_PRODUCTION_RESOLUTION,
    walltime_policy=WALLTIME_POLICY,
)
initial_resume_path, initial_resume_validation = write_rutherford_resume_plan(RUN_DIR)
execution_manifest_path, execution_validation = write_rutherford_production_execution(
    RUN_DIR,
    max_steps=MAX_STEPS_THIS_CHUNK,
    write_movies=WRITE_MOVIES,
)
resume_path, resume_validation = write_rutherford_resume_plan(RUN_DIR)

campaign_plan = json.loads((RUN_DIR / "campaign_plan.json").read_text(encoding="utf-8"))
resume_plan = json.loads(resume_path.read_text(encoding="utf-8"))
execution_diagnostics = json.loads((RUN_DIR / "diagnostics.json").read_text(encoding="utf-8"))
with np.load(RUN_DIR / "production_history.npz", allow_pickle=False) as history:
    time = np.asarray(history["time"], dtype=float)
    reconnected_flux = np.asarray(history["reconnected_flux"], dtype=float)
    island_width = np.asarray(history["rutherford_island_width"], dtype=float)
    reconnection_rate = np.asarray(history["reconnection_rate_proxy"], dtype=float)
    magnetic_energy = np.asarray(history["magnetic_energy"], dtype=float)
    kinetic_energy = np.asarray(history["kinetic_energy"], dtype=float)
    total_energy = np.asarray(history["total_energy"], dtype=float)
    current_aspect_ratio = np.asarray(history["current_sheet_aspect_ratio"], dtype=float)
    current_length = np.asarray(history["current_sheet_length"], dtype=float)
    current_thickness = np.asarray(history["current_sheet_thickness"], dtype=float)

summary_path = RUN_DIR / "figures" / "publication_rutherford_production_summary.png"
summary_path.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

axes[0, 0].plot(time, reconnected_flux, "o-", label="reconnected flux")
island_axis = axes[0, 0].twinx()
island_axis.plot(time, island_width, "s--", color="tab:orange", label="island width")
axes[0, 0].set_title("Rutherford observables")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel("flux amplitude")
island_axis.set_ylabel("island width proxy")
axes[0, 0].grid(True, alpha=0.25)
handles, labels = axes[0, 0].get_legend_handles_labels()
extra_handles, extra_labels = island_axis.get_legend_handles_labels()
axes[0, 0].legend(handles + extra_handles, labels + extra_labels, frameon=False)

axes[0, 1].plot(time, reconnection_rate, "o-", color="tab:red")
axes[0, 1].axhline(0.0, color="0.35", lw=1.0)
axes[0, 1].set_title("Reconnection-rate proxy")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel(r"$d\psi_{rec}/dt$")
axes[0, 1].grid(True, alpha=0.25)

axes[1, 0].plot(time, magnetic_energy, label=r"$E_B$")
axes[1, 0].plot(time, kinetic_energy, label=r"$E_K$")
axes[1, 0].plot(time, total_energy, label=r"$E$")
axes[1, 0].set_title("Energy budget")
axes[1, 0].set_xlabel("time")
axes[1, 0].set_ylabel("mean energy")
axes[1, 0].grid(True, alpha=0.25)
axes[1, 0].legend(frameon=False)

axes[1, 1].plot(time, current_aspect_ratio, label="aspect ratio")
geometry_axis = axes[1, 1].twinx()
geometry_axis.plot(time, current_length, "--", label="length")
geometry_axis.plot(time, current_thickness, ":", label="thickness")
axes[1, 1].set_title("Current-sheet geometry proxy")
axes[1, 1].set_xlabel("time")
axes[1, 1].set_ylabel("aspect ratio")
geometry_axis.set_ylabel("length scale")
axes[1, 1].grid(True, alpha=0.25)
handles, labels = axes[1, 1].get_legend_handles_labels()
extra_handles, extra_labels = geometry_axis.get_legend_handles_labels()
axes[1, 1].legend(handles + extra_handles, labels + extra_labels, frameon=False)

figure.suptitle(
    "MHX publication example: restartable Rutherford production path",
    fontsize=14,
)
figure.savefig(summary_path, dpi=FIGURE_DPI)
plt.close(figure)

write_manifest(
    RUN_DIR / "manifest.json",
    config={
        "shape": list(SHAPE),
        "dt": DT,
        "target_saved_frames": TARGET_SAVED_FRAMES,
        "max_steps_this_chunk": MAX_STEPS_THIS_CHUNK,
        "fast_mode": FAST_MODE,
    },
    outputs={
        "campaign_plan": "campaign_plan.json",
        "resume_plan": "resume_plan.json",
        "execution_history": "production_history.npz",
        "execution_diagnostics": "diagnostics.json",
        "execution_validation": "validation.json",
        "publication_summary": "figures/publication_rutherford_production_summary.png",
    },
    claim_level="validation",
    claim_scope=(
        "Standalone publication-style Rutherford production-path example. "
        "This demonstrates restartable execution and observables; production "
        "Rutherford claims require the promotion gates documented in docs."
    ),
)

print(f"plan_manifest: {plan_manifest_path}")
print(f"plan_validation_passed: {plan_validation['passed']}")
print(f"initial_resume_plan: {initial_resume_path}")
print(f"initial_resume_validation_passed: {initial_resume_validation['passed']}")
print(f"resume_plan: {resume_path}")
print(f"resume_validation_passed: {resume_validation['passed']}")
print(f"resume_start_step: {resume_plan['start_step']}")
print(f"execution_manifest: {execution_manifest_path}")
print(f"execution_validation_passed: {execution_validation['passed']}")
print(f"summary_figure: {summary_path}")
print(f"history: {RUN_DIR / 'production_history.npz'}")
print(f"flux_movie: {RUN_DIR / 'figures' / 'fixed_scale_flux_movie.gif'}")
print(f"current_movie: {RUN_DIR / 'figures' / 'fixed_scale_current_density_movie.gif'}")
print(f"target_step: {campaign_plan['estimated_steps']}")
print(f"chunk_end_step: {execution_diagnostics['end_step']}")
