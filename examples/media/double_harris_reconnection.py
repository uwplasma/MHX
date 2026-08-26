"""Run the plasmoid-chain example and render reconnection/current-sheet media."""

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

CASE = "double_harris_reconnection"
PRESETS = {
    "preview": {
        "nx": 128,
        "ny": 512,
        "sheet_sep": 3.141592653589793,
        "width": 0.15,
        "dt": 2.0e-3,
        "t_end": 10.0,
        "save_every": 250,
        "eta": 5.0e-5,
        "nu": 5.0e-5,
        "noise_amplitude": 2.0e-3,
        "seed": 42,
    },
    "final": {
        "nx": 1024,
        "ny": 1024,
        "sheet_sep": 12.566370614359172,
        "width": 0.05,
        "dt": 2.0e-3,
        "t_end": 60.0,
        "save_every": 500,
        "eta": 5.0e-5,
        "nu": 5.0e-5,
        "noise_amplitude": 2.0e-3,
        "seed": 42,
    },
}


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run the established broadband-seeded plasmoid-chain example."""
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
            "noise",
            "--noise-amp",
            str(config["noise_amplitude"]),
            "--seed",
            str(config["seed"]),
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
            "psi",
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
            "domain": [
                0.0,
                2.0 * config["sheet_sep"],
                0.0,
                8.0 * 3.141592653589793,
            ],
            "trajectory": source.name,
            "initialization": "broadband-seeded examples/plasmoid_chain/plasmoids.py",
        },
    )
    return source


def _render_frame(
    field: object,
    contour_psi: object,
    *,
    lx: float,
    ly: float,
    title: str,
    label: str,
    norm: object,
    contour_levels: object,
) -> object:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(
        figsize=(7.4, 6.5),
        dpi=100,
        constrained_layout=True,
    )
    image = axis.imshow(
        field.T,
        origin="lower",
        extent=(0.0, lx, 0.0, ly),
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
    )
    axis.contour(
        contour_psi.T,
        levels=contour_levels,
        colors="black",
        linewidths=0.4,
        alpha=0.6,
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
    )
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=label, shrink=0.82)
    frame = figure_frame(figure)
    plt.close(figure)
    return frame


def render(*, source: Path, maximum_frames: int = 24) -> dict[str, Path]:
    """Render flux and current from one plasmoid-example trajectory."""
    import numpy as np
    from matplotlib.colors import Normalize, SymLogNorm

    source = require_source(source, CASE)
    metadata = load_source_metadata(source)
    with np.load(source, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)
        lx = float(data["lx"])
        ly = float(data["ly"])
    if metadata.get("shape"):
        validate_array_shape(psi.shape[1:], metadata["shape"], label=CASE)

    indices = sample_indices(len(time), maximum_frames)
    flux_norm = Normalize(
        vmin=float(np.percentile(psi[indices], 0.3)),
        vmax=float(np.percentile(psi[indices], 99.7)),
    )
    current_limit = max(
        float(np.percentile(np.abs(current[indices]), 99.5)),
        np.finfo(float).eps,
    )
    current_norm = SymLogNorm(
        linthresh=max(0.02 * current_limit, np.finfo(float).eps),
        linscale=0.8,
        vmin=-current_limit,
        vmax=current_limit,
        base=10,
    )
    contour_levels = np.linspace(
        float(np.percentile(psi[0], 2.0)),
        float(np.percentile(psi[0], 98.0)),
        40,
    )

    flux_frames = []
    current_frames = []
    for index in indices:
        flux_frames.append(
            _render_frame(
                psi[index],
                psi[index],
                lx=lx,
                ly=ly,
                title=rf"Double-Harris plasmoid-chain reconnection, $t={time[index]:.1f}$",
                label=r"$\psi$",
                norm=flux_norm,
                contour_levels=contour_levels,
            )
        )
        current_frames.append(
            _render_frame(
                current[index],
                psi[index],
                lx=lx,
                ly=ly,
                title=rf"Double-Harris plasmoid current sheets, $t={time[index]:.1f}$",
                label=r"$j_z$",
                norm=current_norm,
                contour_levels=contour_levels,
            )
        )

    outputs = encode_frames(
        flux_frames,
        stem="double_harris_reconnection",
        source=source,
        source_metadata=metadata,
        times=time[indices],
        write_gif=True,
        write_mp4=True,
        fps=6,
    )
    current_outputs = encode_frames(
        current_frames,
        stem="double_harris_current_sheet",
        source=source,
        source_metadata=metadata,
        times=time[indices],
        write_gif=True,
        write_mp4=True,
        fps=6,
    )
    outputs.update({f"current_{name}": path for name, path in current_outputs.items()})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--preset", choices=PRESETS, default="preview")
    simulate_parser.add_argument("--outdir", type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--source", type=Path)
    render_parser.add_argument("--preset", choices=PRESETS, default="preview")
    render_parser.add_argument("--max-frames", type=int, default=24)
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
