"""Plot the checked-in reconnection-ensemble measurements."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "docs" / "_static" / "performance"
OUTPUT = ROOT / "docs" / "_static" / "readme" / "strong_scaling.png"

datasets = [
    json.loads(
        (DATA_DIR / "cpu_ensemble_strong_scaling.json").read_text(encoding="utf-8")
    ),
    json.loads(
        (DATA_DIR / "gpu_ensemble_strong_scaling.json").read_text(encoding="utf-8")
    ),
]

figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
colors = {"cpu": "#3266a8", "gpu": "#b94b45"}

for axis, data in zip(axes, datasets, strict=True):
    counts = data["device_counts"]
    speedup = data["speedup"]
    platform = data["platform"].upper()
    shape = data["shape"][0]
    axis.plot(counts, speedup, "o-", color=colors[data["platform"]], label="measured")
    axis.plot(counts, counts, "--", color="0.45", label="ideal")
    axis.set_xticks(counts)
    axis.set_xlabel("devices")
    axis.set_ylabel("speedup")
    axis.set_title(f"{platform}, four fixed {shape} x {shape} cases")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

figure.suptitle("MHX reconnection-ensemble strong scaling")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT, dpi=180)
plt.close(figure)
print(OUTPUT)
