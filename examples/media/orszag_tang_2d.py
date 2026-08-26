"""Run and render the labeled two-dimensional Orszag--Tang current movie."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import (
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
        encode_frames,
        figure_frame,
        load_source_metadata,
        require_source,
        sample_indices,
        source_dir,
        validate_array_shape,
        write_source_metadata,
    )

CASE = "orszag_tang_2d"
PRESETS = {
    "preview": {"shape": (48, 48), "t_end": 2.0, "save_every": 20},
    "final": {"shape": (96, 96), "t_end": 10.0, "save_every": 40},
}


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run one explicit reduced-MHD Orszag--Tang source campaign."""
    from mhx.benchmarks import write_orszag_tang_vortex_validation

    config = PRESETS[preset]
    write_orszag_tang_vortex_validation(
        outdir,
        shape=config["shape"],
        t_end=config["t_end"],
        save_every=config["save_every"],
        movies=False,
    )
    source = outdir / "orszag_tang_vortex.npz"
    write_source_metadata(
        outdir,
        {"case": CASE, "preset": preset, **config, "trajectory": source.name},
    )
    return source


def render(*, source: Path, maximum_frames: int = 36) -> dict[str, Path]:
    """Render the sole curated 2-D Orszag--Tang current-density animation."""
    import matplotlib.pyplot as plt
    import numpy as np

    source = require_source(source, CASE)
    metadata = load_source_metadata(source)
    with np.load(source, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)
    if metadata.get("shape"):
        validate_array_shape(current.shape[1:], metadata["shape"], label=CASE)

    indices = sample_indices(len(time), maximum_frames)
    limit = max(float(np.percentile(np.abs(current[indices]), 99.5)), np.finfo(float).eps)
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
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(
            "2-D reduced-MHD Orszag--Tang vortex\n"
            rf"out-of-plane current density $j_z$, t = {time[index]:.2f}"
        )
        figure.colorbar(image, ax=axis, label=r"$j_z$ (code units)", shrink=0.82)
        frames.append(figure_frame(figure))
        plt.close(figure)

    return encode_frames(
        frames,
        stem="orszag_tang_current",
        source=source,
        source_metadata=metadata,
        times=time[indices],
        write_gif=True,
        write_mp4=True,
        fps=9,
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
    render_parser.add_argument("--max-frames", type=int, default=36)
    args = parser.parse_args()
    if args.command == "simulate":
        outdir = args.outdir or source_dir(CASE, args.preset)
        print(simulate(preset=args.preset, outdir=outdir))
    else:
        source = args.source or source_dir(CASE, args.preset) / "orszag_tang_vortex.npz"
        for path in render(source=source, maximum_frames=args.max_frames).values():
            print(path)


if __name__ == "__main__":
    main()
