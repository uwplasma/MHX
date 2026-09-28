"""Run and render the decaying reduced-MHD turbulence documentation case."""

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

CASE = "decaying_turbulence"
PRESETS = {
    "preview": {"shape": (64, 64), "dt": 1.0e-2, "t_end": 2.0, "save_every": 20},
    "final": {"shape": (256, 256), "dt": 4.0e-3, "t_end": 10.0, "save_every": 100},
}


def simulate(*, preset: str, outdir: Path) -> Path:
    """Run one explicit decaying-turbulence source campaign."""
    from mhx.benchmarks import write_decaying_mhd_turbulence_validation

    config = PRESETS[preset]
    write_decaying_mhd_turbulence_validation(
        outdir,
        shape=config["shape"],
        dt=config["dt"],
        t_end=config["t_end"],
        save_every=config["save_every"],
        movies=False,
    )
    source = outdir / "decaying_mhd_turbulence.npz"
    write_source_metadata(
        outdir,
        {"case": CASE, "preset": preset, **config, "trajectory": source.name},
    )
    return source


def _magnetic_spectrum(psi: object) -> tuple[object, object]:
    import numpy as np

    values = np.asarray(psi, dtype=float)
    n = values.shape[0]
    coefficients = np.fft.fftn(values)
    kx = np.fft.fftfreq(values.shape[0]) * values.shape[0]
    ky = np.fft.fftfreq(values.shape[1]) * values.shape[1]
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    radius = np.rint(np.hypot(kx_grid, ky_grid)).astype(int)
    energy = (kx_grid**2 + ky_grid**2) * np.abs(coefficients) ** 2 / values.size**2
    shells = np.arange(1, n // 2)
    spectrum = np.asarray([energy[radius == shell].sum() for shell in shells])
    return shells, spectrum


def render(*, source: Path, maximum_frames: int = 36) -> dict[str, Path]:
    """Render current density and a live magnetic spectrum from one trajectory."""
    import matplotlib.pyplot as plt
    import numpy as np

    source = require_source(source, CASE)
    metadata = load_source_metadata(source)
    with np.load(source, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        current = np.asarray(data["current_density"], dtype=float)
    if metadata.get("shape"):
        validate_array_shape(current.shape[1:], metadata["shape"], label=CASE)

    indices = sample_indices(len(time), maximum_frames)
    current_limit = max(float(np.max(np.abs(current[indices]))), np.finfo(float).eps)
    spectra = [_magnetic_spectrum(psi[index]) for index in indices]
    positive = [values[values > 0.0] for _, values in spectra]
    spectrum_min = min(float(values.min()) for values in positive if values.size)
    spectrum_max = max(float(values.max()) for values in positive if values.size)
    k_max = max(int(k[-1]) for k, _ in spectra)

    frames = []
    for index, (wavenumber, spectrum) in zip(indices, spectra, strict=True):
        figure = plt.figure(figsize=(10.8, 4.8), dpi=100, layout="constrained")
        grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 0.045, 1.0))
        axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 2]))
        colorbar_axis = figure.add_subplot(grid[0, 1])
        image = axes[0].imshow(
            current[index].T,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap="RdBu_r",
            vmin=-current_limit,
            vmax=current_limit,
            aspect="equal",
        )
        axes[0].set_title(r"Current density $j_z$")
        axes[0].set_xlabel(r"$x$")
        axes[0].set_ylabel(r"$y$")
        figure.colorbar(image, cax=colorbar_axis, label=r"$j_z$ (code units)")

        mask = spectrum > 0.0
        axes[1].loglog(wavenumber[mask], spectrum[mask], color="#087f8c", linewidth=2.0)
        axes[1].axvline(
            current.shape[1] / 3.0,
            color="0.35",
            linestyle=":",
            linewidth=1.1,
            label="2/3 cutoff",
        )
        axes[1].set_xlim(1.0, k_max)
        axes[1].set_ylim(spectrum_min * 0.7, spectrum_max * 1.4)
        axes[1].set_xlabel(r"wavenumber $k$")
        axes[1].set_ylabel(r"magnetic spectrum $E_B(k)$")
        axes[1].set_title("Instantaneous magnetic spectrum")
        axes[1].grid(True, which="both", alpha=0.2)
        axes[1].legend(frameon=False, loc="lower left")
        figure.suptitle(f"Decaying reduced-MHD turbulence, t = {time[index]:.2f}")
        frames.append(figure_frame(figure))
        plt.close(figure)

    return encode_frames(
        frames,
        stem="decaying_mhd_turbulence_current_256",
        source=source,
        source_metadata=metadata,
        times=time[indices],
        write_gif=True,
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
    outdir = args.outdir or source_dir(CASE, args.preset) if args.command == "simulate" else None
    if args.command == "simulate":
        print(simulate(preset=args.preset, outdir=outdir))
    else:
        source = args.source or source_dir(CASE, args.preset) / "decaying_mhd_turbulence.npz"
        for path in render(source=source, maximum_frames=args.max_frames).values():
            print(path)


if __name__ == "__main__":
    main()
