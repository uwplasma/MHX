"""Run and render the forced three-dimensional MHD turbulence case."""

from __future__ import annotations

import argparse
import os
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
try:
    from .orszag_tang_3d import _current_magnitude
except ImportError:  # Direct script execution.
    from orszag_tang_3d import _current_magnitude

CASE = "turbulence_3d"
PRESETS = {
    "preview": {"shape": (16, 16, 16), "dt": 2.0e-3, "t_end": 0.04, "save_every": 10},
    "final": {"shape": (96, 96, 96), "dt": 1.6e-3, "t_end": 200.0, "save_every": 625},
}


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run the established forced-3D-turbulence gallery simulation."""
    config = PRESETS[preset]
    environment = os.environ.copy()
    environment.update(
        {
            "MHX_TURBULENCE_3D_N": str(config["shape"][0]),
            "MHX_TURBULENCE_3D_DT": str(config["dt"]),
            "MHX_TURBULENCE_3D_T_END": str(config["t_end"]),
            "MHX_TURBULENCE_3D_SAVE_EVERY": str(config["save_every"]),
            "MHX_TURBULENCE_3D_OUTDIR": str(outdir),
        }
    )
    subprocess.run(
        [sys.executable, str(ROOT / "examples/gallery/11_turbulence_3d.py")],
        check=True,
        env=environment,
    )
    source = require_source(outdir / "trajectory.npz", CASE)
    write_source_metadata(
        outdir,
        {"case": CASE, "preset": preset, **config, "trajectory": source.name},
    )
    return source


def render(*, source: Path, maximum_frames: int = 40) -> dict[str, Path]:
    """Render central-slice and maximum-projection views of current magnitude."""
    import matplotlib.pyplot as plt
    import numpy as np

    source = require_source(source, CASE)
    metadata = load_source_metadata(source)
    with np.load(source, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=float)
        magnetic = np.asarray(data["magnetic"], dtype=float)
    if metadata.get("shape"):
        validate_array_shape(magnetic.shape[2:], metadata["shape"], label=CASE)

    indices = sample_indices(len(times), maximum_frames)
    magnitudes = [_current_magnitude(magnetic[index]) for index in indices]
    limit = max(float(np.percentile(np.stack(magnitudes), 99.5)), np.finfo(float).eps)
    frames = []
    for index, magnitude in zip(indices, magnitudes, strict=True):
        midplane = magnitude[:, :, magnitude.shape[2] // 2]
        projection = magnitude.max(axis=2)
        figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), dpi=100, constrained_layout=True)
        for axis, values, title in (
            (axes[0], midplane, r"$|\mathbf{J}|$ at $z=L_z/2$"),
            (axes[1], projection, r"$\max_z |\mathbf{J}|$"),
        ):
            image = axis.imshow(
                values.T,
                origin="lower",
                extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
                cmap="RdBu_r",
                vmin=0.0,
                vmax=limit,
                aspect="equal",
            )
            axis.set_title(title)
            axis.set_xlabel(r"$x$")
            axis.set_ylabel(r"$y$")
        figure.colorbar(image, ax=axes, label=r"$|\mathbf{J}|$ (code units)", shrink=0.82)
        figure.suptitle(f"Continually forced 3-D MHD turbulence, t = {times[index]:.1f}")
        frames.append(figure_frame(figure))
        plt.close(figure)

    return encode_frames(
        frames,
        stem="forced_3d_turbulence_current",
        source=source,
        source_metadata=metadata,
        times=times[indices],
        write_gif=False,
        write_mp4=True,
        fps=10,
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
    render_parser.add_argument("--max-frames", type=int, default=40)
    args = parser.parse_args()
    if args.command == "simulate":
        outdir = args.outdir or source_dir(CASE, args.preset)
        print(simulate(preset=args.preset, outdir=outdir))
    else:
        source = args.source or source_dir(CASE, args.preset) / "trajectory.npz"
        for path in render(source=source, maximum_frames=args.max_frames).values():
            print(path)


if __name__ == "__main__":
    main()
