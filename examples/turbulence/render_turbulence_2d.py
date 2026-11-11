r"""Animate forced 2-D MHD turbulence: current density beside its magnetic spectrum.

Each frame shows the current density j_z on the left and the magnetic spectrum
S_B(k) on the right, with a power-law fit over k = KMIN to KMAX (the inertial band
that avoids the 2/3 dealiasing edge). The spectrum is plotted only up to KPLOT and
uses the benchmark's own normalization.

Writes ``turbulence_2d_dynamic.gif``, ``turbulence_2d_final.png`` and
``slope_history.csv`` to ``<npz folder>/dynamic/``. Rendering 60 frames takes about
a minute. Run it from the repository root, after benchmark_turbulent_spectrum.py:

    python examples/turbulence/render_turbulence_2d.py

To render a different run, point ``NPZ`` at its file; lower ``FRAMES`` for a
quicker, smaller GIF.
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from mhx.diagnostics import fit_power_law, magnetic_power_spectrum_2d

# --- Input parameters ------------------------------------------------------
NPZ = Path("outputs/examples/turbulence_2d/turbulent_spectrum.npz")  # saved run
OUTDIR = None  # output folder; None means <npz folder>/dynamic
KMIN = 4  # lower edge of the fit band
KMAX = 40  # upper edge of the fit band
KPLOT = 100  # largest wavenumber shown
FRAMES = 60  # frames in the GIF
DOMAIN = 6.283185307179586  # box side 2*pi

# --- Run -------------------------------------------------------------------
npz = NPZ.resolve()
outdir = Path(OUTDIR) if OUTDIR is not None else npz.parent / "dynamic"
outdir.mkdir(parents=True, exist_ok=True)

print(f"Loading {npz}...", flush=True)
with np.load(npz) as data:
    time_axis = np.asarray(data["time"], dtype=float)
    psi = np.asarray(data["psi"], dtype=float)
    current = np.asarray(data["current_density"], dtype=float)

indices = np.unique(np.linspace(0, len(time_axis) - 1, min(len(time_axis), FRAMES), dtype=int))
print(f"Computing spectra and fits for {len(indices)} of {len(time_axis)} frames...", flush=True)
spectra = []
slopes = []
for n, index in enumerate(indices):
    k, power = magnetic_power_spectrum_2d(psi[index], domain_size=DOMAIN)
    try:
        slope, intercept = fit_power_law(k, power, kmin=KMIN, kmax=KMAX)
    except ValueError:
        slope, intercept = np.nan, np.nan
    spectra.append((k, power))
    slopes.append((slope, intercept))
    print(f"  spectrum {n + 1}/{len(indices)} (t = {time_axis[index]:.2f})", flush=True)
slopes = np.asarray(slopes)

limit = float(np.percentile(np.abs(current[indices]), 99.5))
limit = limit if limit > 0.0 else 1.0
shown_values = np.concatenate([p[(k <= KPLOT) & (p > 0.0)] for k, p in spectra])
y_low, y_high = 0.5 * shown_values.min(), 2.0 * shown_values.max()

cmap = plt.get_cmap("RdBu_r")
current_norm = Normalize(vmin=-limit, vmax=limit)


def draw(axis_j, axis_s, frame_number: int) -> None:
    """Draw one frame: j_z on ``axis_j`` and the spectrum with its fit on ``axis_s``."""
    index = indices[frame_number]
    k, power = spectra[frame_number]
    slope, intercept = slopes[frame_number]
    axis_j.clear()
    axis_j.imshow(
        current[index].T, origin="lower", extent=(0.0, DOMAIN, 0.0, DOMAIN),
        cmap=cmap, vmin=-limit, vmax=limit, aspect="equal",
    )
    axis_j.set_xlabel(r"$x$")
    axis_j.set_ylabel(r"$y$")
    axis_j.set_title(f"$j_z$, t = {time_axis[index]:.2f}")

    axis_s.clear()
    shown = (k <= KPLOT) & (power > 0.0)
    axis_s.loglog(k[shown], power[shown], color="#00a6a6", lw=2, label=r"$S_B(k)$")
    if np.isfinite(slope):
        fit_k = np.geomspace(KMIN, KMAX, 100)
        axis_s.loglog(
            fit_k, 10.0**intercept * fit_k**slope, "k--", lw=1.5,
            label=rf"fit $k^{{{slope:.2f}}}$, $k\in[{KMIN},{KMAX}]$",
        )
    axis_s.axvline(KMIN, color="0.6", ls=":", lw=1.0)
    axis_s.axvline(KMAX, color="0.6", ls=":", lw=1.0)
    axis_s.set_xlim(1, KPLOT)
    axis_s.set_ylim(y_low, y_high)
    axis_s.set_xlabel(r"wavenumber $k$")
    axis_s.set_ylabel(r"magnetic spectrum $S_B(k)$")
    axis_s.set_title(f"Magnetic spectrum, t = {time_axis[index]:.2f}")
    axis_s.grid(True, which="both", alpha=0.2)
    axis_s.legend(frameon=False, loc="lower left", fontsize="small")


figure, (axis_j, axis_s) = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
figure.colorbar(ScalarMappable(norm=current_norm, cmap=cmap), ax=axis_j, shrink=0.8,
                label=r"$j_z$ (code units)")
draw(axis_j, axis_s, len(indices) - 1)
figure.savefig(outdir / "turbulence_2d_final.png", dpi=160)

print(f"Rendering {len(indices)} GIF frames...", flush=True)
render_start = time.perf_counter()


def render_frame(i: int) -> None:
    draw(axis_j, axis_s, i)
    print(f"  frame {i + 1}/{len(indices)} ({time.perf_counter() - render_start:.1f} s)",
          flush=True)


animation = FuncAnimation(figure, render_frame, frames=len(indices), interval=150)
animation.save(outdir / "turbulence_2d_dynamic.gif", writer=PillowWriter(fps=6), dpi=90)
plt.close(figure)

with (outdir / "slope_history.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["t", "slope_S_B", "intercept_log10"])
    for index, (slope, intercept) in zip(indices, slopes, strict=True):
        writer.writerow([time_axis[index], slope, intercept])

# --- Results ---------------------------------------------------------------
print(f"fit k in [{KMIN}, {KMAX}] over {len(indices)} frames")
print(f"final-frame slope = {slopes[-1][0]:.3f}")
print(f"wrote {outdir}")
