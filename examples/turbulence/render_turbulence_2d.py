r"""Animate forced 2-D MHD turbulence: current density beside its magnetic spectrum.

Each frame shows the current density j_z on the left and the magnetic spectrum
S_B(k) on the right, with a power-law fit over k = 4 to 40 (the inertial band
that avoids the 2/3 dealiasing edge). The spectrum is plotted only up to k = 100.
The spectrum uses the benchmark's own normalization, so the numbers match the
benchmark's spectrum.

Outputs go to <npz folder>/dynamic/: turbulence_2d_dynamic.gif,
turbulence_2d_final.png, and slope_history.csv.

Run it from the repository root, after benchmark_turbulent_spectrum.py:
    python examples/turbulence/render_turbulence_2d.py \
        --npz outputs/turbulence_2d/turbulent_spectrum.npz
    python examples/turbulence/render_turbulence_2d.py --kmin 4 --kmax 40 --kplot 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples.turbulence.benchmark_turbulent_spectrum import (  # noqa: E402
    _compute_magnetic_power_spectrum,
)

DEFAULT_NPZ = Path("outputs/turbulence_2d/turbulent_spectrum.npz")
DOMAIN = 2.0 * np.pi


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--outdir", type=Path, default=None, help="default: <npz folder>/dynamic")
    parser.add_argument("--kmin", type=int, default=4, help="lower edge of the fit band")
    parser.add_argument("--kmax", type=int, default=40, help="upper edge of the fit band")
    parser.add_argument("--kplot", type=int, default=100, help="largest wavenumber shown")
    parser.add_argument("--frames", type=int, default=60, help="frames in the GIF")
    args = parser.parse_args()

    npz = args.npz.resolve()
    outdir = args.outdir or npz.parent / "dynamic"
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(npz) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)

    indices = np.unique(np.linspace(0, len(time) - 1, min(len(time), args.frames), dtype=int))
    spectra = []
    slopes = []
    for index in indices:
        k, power = _compute_magnetic_power_spectrum(psi[index], domain_size=DOMAIN)
        fit = (k >= args.kmin) & (k <= args.kmax) & (power > 0.0)
        if fit.sum() >= 2:
            slope, intercept = np.polyfit(np.log10(k[fit]), np.log10(power[fit]), 1)
        else:
            slope, intercept = np.nan, np.nan
        spectra.append((k, power))
        slopes.append((slope, intercept))
    slopes = np.asarray(slopes)

    limit = float(np.percentile(np.abs(current[indices]), 99.5))
    limit = limit if limit > 0.0 else 1.0
    shown_values = np.concatenate(
        [p[(k <= args.kplot) & (p > 0.0)] for k, p in spectra]
    )
    y_low, y_high = 0.5 * shown_values.min(), 2.0 * shown_values.max()

    cmap = plt.get_cmap("RdBu_r")
    current_norm = Normalize(vmin=-limit, vmax=limit)

    def draw(axis_j, axis_s, frame_number: int) -> None:
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
        axis_j.set_title(f"$j_z$, t = {time[index]:.2f}")

        axis_s.clear()
        shown = (k <= args.kplot) & (power > 0.0)
        axis_s.loglog(k[shown], power[shown], color="#00a6a6", lw=2, label=r"$S_B(k)$")
        if np.isfinite(slope):
            fit_k = np.geomspace(args.kmin, args.kmax, 100)
            axis_s.loglog(
                fit_k, 10.0**intercept * fit_k**slope, "k--", lw=1.5,
                label=rf"fit $k^{{{slope:.2f}}}$, $k\in[{args.kmin},{args.kmax}]$",
            )
        axis_s.axvline(args.kmin, color="0.6", ls=":", lw=1.0)
        axis_s.axvline(args.kmax, color="0.6", ls=":", lw=1.0)
        axis_s.set_xlim(1, args.kplot)
        axis_s.set_ylim(y_low, y_high)
        axis_s.set_xlabel(r"wavenumber $k$")
        axis_s.set_ylabel(r"magnetic spectrum $S_B(k)$")
        axis_s.set_title(f"Magnetic spectrum, t = {time[index]:.2f}")
        axis_s.grid(True, which="both", alpha=0.2)
        axis_s.legend(frameon=False, loc="lower left", fontsize="small")

    figure, (axis_j, axis_s) = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    figure.colorbar(ScalarMappable(norm=current_norm, cmap=cmap), ax=axis_j, shrink=0.8,
                    label=r"$j_z$ (code units)")
    draw(axis_j, axis_s, len(indices) - 1)
    figure.savefig(outdir / "turbulence_2d_final.png", dpi=160)

    animation = FuncAnimation(
        figure, lambda i: draw(axis_j, axis_s, i), frames=len(indices), interval=150,
    )
    animation.save(outdir / "turbulence_2d_dynamic.gif", writer=PillowWriter(fps=6), dpi=90)
    plt.close(figure)

    with (outdir / "slope_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "slope_S_B", "intercept_log10"])
        for index, (slope, intercept) in zip(indices, slopes, strict=True):
            writer.writerow([time[index], slope, intercept])

    print(f"fit k in [{args.kmin}, {args.kmax}] over {len(indices)} frames")
    print(f"final-frame slope = {slopes[-1][0]:.3f}")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
