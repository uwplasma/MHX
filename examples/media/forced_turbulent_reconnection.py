"""Retain the separate forced-current-sheet validation media workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import encode_frames, figure_frame, require_source, sample_indices, source_dir

CASE = "forced_turbulent_reconnection"


def simulate(outdir: Path) -> Path:
    """Run the documented validation-only forced current-sheet case."""
    from mhx.benchmarks import write_forced_turbulent_reconnection_validation

    write_forced_turbulent_reconnection_validation(
        outdir,
        shape=(64, 64),
        width=0.35,
        resistivity=1.0e-3,
        viscosity=1.0e-3,
        perturbation_amplitude=1.0e-1,
        turbulent_flux_amplitude=1.0e-1,
        turbulent_flow_amplitude=1.0e-1,
        forcing_amplitude=2.0e-2,
        dt=2.0e-2,
        t_end=80.0,
        save_every=100,
        max_relative_energy_growth=30.0,
        movies=False,
    )
    return outdir / "forced_turbulent_reconnection.npz"


def render(source: Path) -> dict[str, Path]:
    """Render current density with magnetic-flux contours for validation pages."""
    import matplotlib.pyplot as plt
    import numpy as np

    source = require_source(source, CASE)
    with np.load(source, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)
    indices = sample_indices(len(time), 20)
    limit = max(float(np.percentile(np.abs(current[indices]), 99.5)), np.finfo(float).eps)
    coordinates = np.linspace(0.0, 2.0 * np.pi, current.shape[1], endpoint=False)
    frames = []
    for index in indices:
        figure, axis = plt.subplots(figsize=(6.2, 5.4), dpi=100, constrained_layout=True)
        image = axis.imshow(
            current[index].T,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        axis.contour(
            coordinates,
            coordinates,
            psi[index].T,
            colors="black",
            levels=12,
            linewidths=0.4,
        )
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(f"Forced turbulent-reconnection validation, t = {time[index]:.1f}")
        figure.colorbar(image, ax=axis, label=r"$j_z$ (code units)", shrink=0.82)
        frames.append(figure_frame(figure))
        plt.close(figure)
    return encode_frames(
        frames,
        stem="forced_turbulent_reconnection",
        source=source,
        source_metadata={"case": CASE, "shape": list(current.shape[1:]), "t_end": time[-1]},
        times=time[indices],
        write_gif=True,
        write_mp4=True,
        fps=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--outdir", type=Path, default=source_dir(CASE, "final"))
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.command == "simulate":
        print(simulate(args.outdir))
    else:
        source = args.source or source_dir(CASE, "final") / "forced_turbulent_reconnection.npz"
        for path in render(source).values():
            print(path)


if __name__ == "__main__":
    main()
