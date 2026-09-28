"""Run and render the two-panel three-dimensional Orszag--Tang movie."""

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

CASE = "orszag_tang_3d"
PRESETS = {
    "preview": {
        "shape": (16, 16, 16),
        "dt": 2.0e-3,
        "t_end": 0.04,
        "save_every": 10,
        "dissipation": 5.0e-3,
    },
    "final": {
        "shape": (192, 192, 192),
        "dt": 8.0e-4,
        "t_end": 4.0,
        "save_every": 250,
        "dissipation": 2.0e-3,
    },
}


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run and save one incompressible 3-D Orszag--Tang source campaign."""
    import mhx

    config = PRESETS[preset]
    simulation = mhx.Simulation(
        shape=config["shape"],
        equations="mhd3d",
        equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
        viscosity=config["dissipation"],
        resistivity=config["dissipation"],
        dt=config["dt"],
        t_end=config["t_end"],
        save_every=config["save_every"],
    )
    result = simulation.run()
    result.save(outdir)
    source = outdir / "trajectory.npz"
    write_source_metadata(
        outdir,
        {"case": CASE, "preset": preset, **config, "trajectory": source.name},
    )
    return require_source(source, CASE)


def _current_magnitude(magnetic: object) -> object:
    import numpy as np

    field = np.asarray(magnetic, dtype=float)
    shape = field.shape[1:]
    wavevectors = [np.fft.fftfreq(size, d=1.0 / size) for size in shape]
    kx, ky, kz = np.meshgrid(*wavevectors, indexing="ij")
    spectrum = np.fft.fftn(field, axes=(-3, -2, -1))
    current_hat = np.empty_like(spectrum, dtype=complex)
    current_hat[0] = 1j * (ky * spectrum[2] - kz * spectrum[1])
    current_hat[1] = 1j * (kz * spectrum[0] - kx * spectrum[2])
    current_hat[2] = 1j * (kx * spectrum[1] - ky * spectrum[0])
    current = np.fft.ifftn(current_hat, axes=(-3, -2, -1)).real
    return np.sqrt(np.sum(current**2, axis=0))


def render(*, source: Path, maximum_frames: int = 36) -> dict[str, Path]:
    """Render midplane and maximum-projection views of current magnitude."""
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
        figure.suptitle(f"3-D incompressible Orszag--Tang vortex, t = {times[index]:.2f}")
        frames.append(figure_frame(figure))
        plt.close(figure)

    return encode_frames(
        frames,
        stem="orszag_tang_3d_current",
        source=source,
        source_metadata=metadata,
        times=times[indices],
        write_gif=False,
        write_mp4=True,
        fps=8,
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
        source = args.source or source_dir(CASE, args.preset) / "trajectory.npz"
        for path in render(source=source, maximum_frames=args.max_frames).values():
            print(path)


if __name__ == "__main__":
    main()
