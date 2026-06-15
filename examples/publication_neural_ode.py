"""Publication-style FAST neural-ODE training example.

Edit the parameters below, then run:

    python examples/publication_neural_ode.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from mhx.io import write_manifest
from mhx.neural_ode import write_neural_ode_training_bundle
from mhx.runtime import configure_jax

CI_SMOKE = False
FAST_MODE = os.environ.get("MHX_EXAMPLE_FAST", "0") == "1"
OUTDIR_ROOT = Path(
    os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples/publication")
).expanduser()
RUN_DIR = OUTDIR_ROOT / "neural_ode"

SHAPE = (8, 8) if FAST_MODE else (16, 16)
SEEDS = (0, 1, 2, 3) if FAST_MODE else (0, 1, 2, 3, 4, 5)
STEPS = 5 if FAST_MODE else 24
DT = 1.0e-2
OBSERVATION_COUNT = 2
HIDDEN_SIZE = 4 if FAST_MODE else 8
RIDGE = 1.0e-8
MODEL_SEED = 0
FIGURE_DPI = 220

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

configure_jax(enable_x64=True)
manifest_path, validation = write_neural_ode_training_bundle(
    RUN_DIR,
    shape=SHAPE,
    seeds=SEEDS,
    steps=STEPS,
    dt=DT,
    observation_count=OBSERVATION_COUNT,
    hidden_size=HIDDEN_SIZE,
    ridge=RIDGE,
    model_seed=MODEL_SEED,
    write_figures=True,
)

splits = json.loads((RUN_DIR / "splits.json").read_text(encoding="utf-8"))
baseline_metrics = json.loads((RUN_DIR / "baseline_metrics.json").read_text(encoding="utf-8"))
latent_metrics = json.loads((RUN_DIR / "latent_ode_metrics.json").read_text(encoding="utf-8"))
failure_modes = json.loads((RUN_DIR / "failure_modes.json").read_text(encoding="utf-8"))

with np.load(RUN_DIR / "latent_ode_predictions.npz", allow_pickle=False) as predictions_file:
    predictions = np.asarray(predictions_file["predictions"], dtype=float)
    targets = np.asarray(predictions_file["targets"], dtype=float)
    times = np.asarray(predictions_file["times"], dtype=float)
    target_names = [str(name) for name in predictions_file["target_names"]]
    seeds = np.asarray(predictions_file["seeds"], dtype=int)

test_seed = int(splits["test"][0])
test_seed_index = int(np.where(seeds == test_seed)[0][0])
mode_amplitude_index = target_names.index("mode_amplitude")
total_energy_index = target_names.index("total_energy")

baseline_test_rmse = {
    name: values["test"]["rmse"] for name, values in baseline_metrics["baselines"].items()
}
rmse_names = [*baseline_test_rmse, "latent_ode"]
rmse_values = [
    *baseline_test_rmse.values(),
    latent_metrics["latent_ode_test_rmse"],
]

summary_path = RUN_DIR / "figures" / "publication_neural_ode_summary.png"
summary_path.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

axes[0, 0].plot(
    times,
    targets[test_seed_index, :, mode_amplitude_index],
    "o-",
    label="solver target",
)
axes[0, 0].plot(
    times,
    predictions[test_seed_index, :, mode_amplitude_index],
    "s--",
    label="latent ODE",
)
axes[0, 0].axvspan(
    times[0],
    times[OBSERVATION_COUNT - 1],
    color="0.9",
    label="observed prefix",
)
axes[0, 0].set_title(f"Mode-amplitude forecast, seed {test_seed}")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel("mode amplitude")
axes[0, 0].grid(True, alpha=0.25)
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(
    times,
    targets[test_seed_index, :, total_energy_index],
    "o-",
    label="solver target",
)
axes[0, 1].plot(
    times,
    predictions[test_seed_index, :, total_energy_index],
    "s--",
    label="latent ODE",
)
axes[0, 1].axvspan(times[0], times[OBSERVATION_COUNT - 1], color="0.9")
axes[0, 1].set_title(f"Total-energy forecast, seed {test_seed}")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("total energy")
axes[0, 1].grid(True, alpha=0.25)
axes[0, 1].legend(frameon=False)

bar_positions = np.arange(len(rmse_names))
axes[1, 0].bar(bar_positions, rmse_values, color=["#9ecae1", "#9ecae1", "#31a354"])
axes[1, 0].set_xticks(bar_positions)
axes[1, 0].set_xticklabels([name.replace("_", "\n") for name in rmse_names])
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("Test RMSE versus deterministic baselines")
axes[1, 0].set_ylabel("RMSE")
axes[1, 0].grid(True, axis="y", alpha=0.25)

axes[1, 1].axis("off")
summary_lines = [
    f"training validation passed: {validation['passed']}",
    f"train/validation/test seeds: {splits['train']} / {splits['validation']} / {splits['test']}",
    f"latent test RMSE: {latent_metrics['latent_ode_test_rmse']:.3e}",
    f"best baseline test RMSE: {latent_metrics['best_baseline_test_rmse']:.3e}",
    (f"latent/best-baseline RMSE ratio: {latent_metrics['test_rmse_ratio_to_best_baseline']:.2f}"),
    "failure-mode warnings: " + ", ".join(failure_modes["warnings"] or ["none"]),
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

figure.suptitle("MHX publication example: deterministic latent neural ODE", fontsize=14)
figure.savefig(summary_path, dpi=FIGURE_DPI)
plt.close(figure)

write_manifest(
    manifest_path,
    config={
        "shape": list(SHAPE),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "dt": DT,
        "observation_count": OBSERVATION_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "fast_mode": FAST_MODE,
    },
    outputs={
        "dataset": "dataset.npz",
        "splits": "splits.json",
        "baseline_metrics": "baseline_metrics.json",
        "latent_ode_model": "latent_ode_model.json",
        "latent_ode_metrics": "latent_ode_metrics.json",
        "latent_ode_predictions": "latent_ode_predictions.npz",
        "failure_modes": "failure_modes.json",
        "validation": "validation.json",
        "publication_summary": "figures/publication_neural_ode_summary.png",
        "latent_ode_rmse_comparison": "figures/latent_ode_rmse_comparison.png",
    },
    claim_level="validation",
    claim_scope=(
        "Standalone deterministic latent neural-ODE example with fixed splits, "
        "baselines, forecasts, and failure-mode diagnostics."
    ),
)

print(f"manifest: {manifest_path}")
print(f"validation_passed: {validation['passed']}")
print(f"summary_figure: {summary_path}")
print(f"latent_predictions: {RUN_DIR / 'latent_ode_predictions.npz'}")
print(f"latent_metrics: {RUN_DIR / 'latent_ode_metrics.json'}")
print(f"training_figure: {RUN_DIR / 'figures' / 'latent_ode_predictions.png'}")
