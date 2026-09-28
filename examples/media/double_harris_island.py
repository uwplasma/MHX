"""Run and render a high-resolution low-mode Double-Harris island."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .common import (
        ROOT,
        encode_frames,
        figure_frame,
        load_source_metadata,
        require_source,
        sample_indices,
        source_dir,
        validate_array_shape,
        write_source_metadata,
    )
except ImportError:  # Direct script execution.
    from common import (
        ROOT,
        encode_frames,
        figure_frame,
        load_source_metadata,
        require_source,
        sample_indices,
        source_dir,
        validate_array_shape,
        write_source_metadata,
    )

CASE = "double_harris_island"
PRESETS = {
    "preview": {
        "nx": 64,
        "ny": 256,
        "dt": 2.0e-3,
        "t_end": 10.0,
        "save_every": 250,
        "sheet_sep": 3.141592653589793,
        "width": 0.25,
        "eta": 1.0e-5,
        "nu": 1.0e-5,
        "perturbation_amplitude": 1.0e-3,
        "mode_y": 1,
    },
    "final": {
        "nx": 256,
        "ny": 1024,
        "dt": 1.0e-3,
        "t_end": 400.0,
        "save_every": 2000,
        "sheet_sep": 3.141592653589793,
        "width": 0.25,
        "eta": 1.0e-5,
        "nu": 1.0e-5,
        "perturbation_amplitude": 1.0e-3,
        "mode_y": 1,
    },
}


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run the plasmoid initializer with one long-wavelength tearing seed."""
    config = PRESETS[preset]
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/plasmoid_chain/plasmoids.py"),
            "--nx",
            str(config["nx"]),
            "--ny",
            str(config["ny"]),
            "--sheet-sep",
            str(config["sheet_sep"]),
            "--width",
            str(config["width"]),
            "--seed-type",
            "single",
            "--mode-y",
            str(config["mode_y"]),
            "--perturbation-amp",
            str(config["perturbation_amplitude"]),
            "--eta",
            str(config["eta"]),
            "--nu",
            str(config["nu"]),
            "--dt",
            str(config["dt"]),
            "--t-end",
            str(config["t_end"]),
            "--save-every",
            str(config["save_every"]),
            "--field",
            "island",
            "--scale",
            "fixed",
            "--outdir",
            str(outdir),
        ],
        check=True,
    )
    source = require_source(outdir / "plasmoids_trajectory.npz", CASE)
    write_source_metadata(
        outdir,
        {
            "case": CASE,
            "preset": preset,
            **config,
            "shape": [config["nx"], config["ny"]],
            "domain": [0.0, 2.0 * config["sheet_sep"], 0.0, 8.0 * 3.141592653589793],
            "trajectory": source.name,
            "initialization": "single low-wavenumber plasmoid seed",
        },
    )
    return source


def render(*, source: Path, maximum_frames: int = 14) -> dict[str, Path]:
    """Render the growing island without unreliable X/O annotations."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    source = require_source(source, CASE)
    metadata = load_source_metadata(source)
    with np.load(source, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        lx = float(data["lx"])
        ly = float(data["ly"])
    if metadata.get("shape"):
        validate_array_shape(psi.shape[1:], metadata["shape"], label=CASE)

    indices = sample_indices(len(time), maximum_frames)
    norm = Normalize(
        vmin=float(np.percentile(psi[indices], 0.3)),
        vmax=float(np.percentile(psi[indices], 99.7)),
    )
    contour_levels = np.linspace(
        float(np.percentile(psi[0], 2.0)),
        float(np.percentile(psi[0], 98.0)),
        28,
    )

    frames = []
    for index in indices:
        figure, axis = plt.subplots(
            figsize=(5.2, 8.6),
            dpi=90,
            constrained_layout=True,
        )
        image = axis.imshow(
            psi[index].T,
            origin="lower",
            extent=(0.0, lx, 0.0, ly),
            cmap="RdBu_r",
            norm=norm,
            aspect="equal",
        )
        axis.contour(
            psi[index].T,
            levels=contour_levels,
            colors="black",
            linewidths=0.5,
            alpha=0.65,
            extent=(0.0, lx, 0.0, ly),
            origin="lower",
        )
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(
            rf"Low-mode Double-Harris island, $t={time[index]:.1f}$"
        )
        figure.colorbar(
            image,
            ax=axis,
            label=r"$\psi$",
            shrink=0.82,
        )
        frames.append(figure_frame(figure))
        plt.close(figure)

    return encode_frames(
        frames,
        stem="double_harris_island",
        source=source,
        source_metadata=metadata,
        times=time[indices],
        write_gif=True,
        write_mp4=True,
        fps=5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--preset", choices=PRESETS, default="preview")
    simulate_parser.add_argument("--outdir", type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--source", type=Path)
    render_parser.add_argument("--preset", choices=PRESETS, default="preview")
    render_parser.add_argument("--max-frames", type=int, default=14)
    args = parser.parse_args()
    if args.command == "simulate":
        outdir = args.outdir or source_dir(CASE, args.preset)
        print(simulate(preset=args.preset, outdir=outdir))
    else:
        source = args.source or source_dir(CASE, args.preset) / "plasmoids_trajectory.npz"
        for path in render(source=source, maximum_frames=args.max_frames).values():
            print(path)


if __name__ == "__main__":
    main()
