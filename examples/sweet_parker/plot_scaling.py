"""Plot the Sweet-Parker scaling scan from finished collapse_check.py runs.

Reads each run's ``summary.json`` (steady-window medians) and draws three panels:

* ``delta/L`` against the sheet Lundquist number ``S_L``,
* the normalized reconnection rate ``E* = eta|j_X|/B_up^2`` against ``S_L``,
* the local log-log slope between neighbouring runs, which should approach
  ``-1/2`` for both quantities.

The dashed grey lines in the first two panels are the Pm-corrected
Sweet-Parker predictions (Park, Monticello & White 1984):
``delta/L = S^-1/2 (1+Pm)^1/4`` and ``E* = S^-1/2 (1+Pm)^-1/4``, which assume
the viscous outflow ``v_out = v_A (1+Pm)^-1/2``. With the measured outflow
``u = v_out/v_A`` instead, Sweet-Parker mass and Ohm balance give
``delta/L = (u S)^-1/2`` and ``E* = u^1/2 S^-1/2``; the hollow markers are the
outflow-corrected data ``delta/L * u^1/2`` and ``E* / u^1/2``, which should lie
on the dotted ``S^-1/2`` line.

Takes seconds. Edit ``RUN_DIRS`` (folders written by ``collapse_check.py``, each
with a ``summary.json`` that has a steady window; at least two) and ``OUTPUT`` in
"Input parameters", then run from the repository root::

    python examples/sweet_parker/plot_scaling.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- Input parameters ------------------------------------------------------
RUN_DIRS = [  # finished collapse_check.py run folders, one per resistivity
    Path("outputs/sweet_parker/eta2e-3"),
    Path("outputs/sweet_parker/eta1e-3"),
]
OUTPUT = Path("outputs/examples/sweet_parker/sweet_parker_scaling.png")  # figure to write

# Reference palette (categorical slots 1-2) and text/surface roles.
DELTA_COLOR = "#2a78d6"
RATE_COLOR = "#eb6834"
THEORY_COLOR = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e6e5e1"
SURFACE = "#fcfcfb"


def load_points(run_dirs: list[Path]) -> list[dict]:
    points = []
    for run_dir in map(Path, run_dirs):
        summary = json.loads((run_dir / "summary.json").read_text())
        metrics = summary["metrics"]
        if "lundquist_window" not in metrics:
            print(f"[plot] skipping {run_dir.name}: no steady window")
            continue
        points.append(
            {
                "name": run_dir.name,
                "eta": summary["resistivity"],
                "pm": summary["viscosity"] / summary["resistivity"],
                "lundquist": metrics["lundquist_window"],
                "rate": metrics["normalized_rate_window"],
                "delta_over_length": 1.0 / metrics["aspect_ratio_window"],
                "outflow": metrics["outflow_over_alfven"],
            }
        )
    return sorted(points, key=lambda point: point["lundquist"])


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
    axis.grid(True, which="minor", color=GRID_COLOR, linewidth=0.4)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    axis.xaxis.label.set_color(TEXT_PRIMARY)
    axis.yaxis.label.set_color(TEXT_PRIMARY)
    axis.title.set_color(TEXT_PRIMARY)

# --- Run -------------------------------------------------------------------

print(f"[plot] reading {len(RUN_DIRS)} runs", flush=True)
points = load_points(RUN_DIRS)
if len(points) < 2:
    raise SystemExit("need at least two runs with a steady window")
pm = points[0]["pm"]
lundquist = np.array([point["lundquist"] for point in points])
rate = np.array([point["rate"] for point in points])
delta_over_length = np.array([point["delta_over_length"] for point in points])
root_outflow = np.sqrt(np.array([point["outflow"] for point in points]))
s_line = np.geomspace(lundquist.min() / 1.3, lundquist.max() * 1.3, 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
fig.patch.set_facecolor(SURFACE)
panels = [
    (
        axes[0],
        delta_over_length,
        DELTA_COLOR,
        "o",
        r"$\delta/L$",
        s_line**-0.5 * (1.0 + pm) ** 0.25,
        "Sheet aspect ratio",
        delta_over_length * root_outflow,
        r"MHX $\times\sqrt{v_{out}/v_A}$ (outflow-corrected)",
    ),
    (
        axes[1],
        rate,
        RATE_COLOR,
        "s",
        r"$E^* = \eta|j_X|/B_{up}^2$",
        s_line**-0.5 * (1.0 + pm) ** -0.25,
        "Normalized reconnection rate",
        rate / root_outflow,
        r"MHX $\div\sqrt{v_{out}/v_A}$ (outflow-corrected)",
    ),
]
for axis, values, color, marker, label, theory, title, corrected, corrected_label in panels:
    style_axis(axis)
    axis.loglog(
        s_line,
        theory,
        color=THEORY_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=rf"Sweet–Parker, Pm = {pm:g}",
    )
    slope = loglog_slope(lundquist, values)
    axis.loglog(
        lundquist,
        values,
        color=color,
        marker=marker,
        markersize=8,
        linewidth=2,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=f"MHX (fitted slope {slope:+.2f})",
    )
    axis.loglog(
        s_line,
        s_line**-0.5,
        color=THEORY_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=r"$S_L^{-1/2}$ (Sweet–Parker, measured outflow)",
    )
    axis.loglog(
        lundquist,
        corrected,
        color=color,
        marker=marker,
        markersize=8,
        linewidth=1.5,
        linestyle="-",
        alpha=0.9,
        markerfacecolor=SURFACE,
        markeredgecolor=color,
        markeredgewidth=1.8,
        label=corrected_label,
    )
    for point, value in zip(points, values, strict=True):
        axis.annotate(
            rf"$\eta$={point['eta']:g}",
            (point["lundquist"], value),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    axis.set_xlabel(r"sheet Lundquist number $S_L = L B_{up}/\eta$")
    axis.set_ylabel(label)
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

axis = axes[2]
style_axis(axis)
midpoints = np.sqrt(lundquist[1:] * lundquist[:-1])
log_steps = np.diff(np.log(lundquist))
local_delta = np.diff(np.log(delta_over_length)) / log_steps
local_rate = np.diff(np.log(rate)) / log_steps
axis.axhline(
    -0.5, color=THEORY_COLOR, linestyle="--", linewidth=1.5, label="Sweet–Parker −1/2"
)
axis.semilogx(
    midpoints,
    local_delta,
    color=DELTA_COLOR,
    marker="o",
    markersize=8,
    linewidth=2,
    markeredgecolor=SURFACE,
    markeredgewidth=1.5,
    label=r"$\delta/L$",
)
axis.semilogx(
    midpoints,
    local_rate,
    color=RATE_COLOR,
    marker="s",
    markersize=8,
    linewidth=2,
    markeredgecolor=SURFACE,
    markeredgewidth=1.5,
    label=r"$E^*$",
)
axis.annotate(
    f"{local_delta[-1]:+.2f}",
    (midpoints[-1], local_delta[-1]),
    textcoords="offset points",
    xytext=(8, 4),
    fontsize=9,
    color=TEXT_PRIMARY,
)
axis.annotate(
    f"{local_rate[-1]:+.2f}",
    (midpoints[-1], local_rate[-1]),
    textcoords="offset points",
    xytext=(8, -12),
    fontsize=9,
    color=TEXT_PRIMARY,
)
axis.set_xlabel(r"$S_L$ (geometric mean of neighbouring runs)")
axis.set_ylabel(r"local slope $d\ln(\cdot)/d\ln S_L$")
axis.set_title("Local scaling exponent")
axis.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

fig.suptitle(
    "Sweet–Parker scaling from MHX X-point collapse (reduced MHD, Pm = 1)",
    color=TEXT_PRIMARY,
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)
print(f"[plot] wrote {OUTPUT}")
for point in points:
    print(
        f"  eta={point['eta']:<8g} S_L={point['lundquist']:8.0f} "
        f"delta/L={point['delta_over_length']:.4f} E*={point['rate']:.5f}"
    )
print(
    f"  all-point slopes: delta/L {loglog_slope(lundquist, delta_over_length):+.3f}, "
    f"E* {loglog_slope(lundquist, rate):+.3f}"
)
