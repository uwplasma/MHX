"""Render supplemental tearing and turbulence schematics outside the main campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import README_PREVIEW, figure_frame
except ImportError:  # Direct script execution.
    from common import README_PREVIEW, figure_frame


def harris_layer_frames() -> list[object]:
    """Return frames for the validated Harris eigenfunction-layer sweep."""
    import matplotlib.pyplot as plt

    from mhx.benchmarks import run_linear_tearing_layer_validation

    result = run_linear_tearing_layer_validation(grid_points=128)
    frames = []
    for marker, lundquist in enumerate(result.lundquist):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=100, constrained_layout=True)
        axes[0].loglog(result.lundquist, result.growth_rate, "o-", label=r"$\gamma$")
        axes[0].loglog(result.lundquist, result.stream_half_width, "s-", label="flow width")
        axes[0].scatter(
            [lundquist],
            [result.growth_rate[marker]],
            s=80,
            facecolors="none",
            edgecolors="black",
        )
        axes[0].set_xlabel("Lundquist number S")
        axes[0].set_ylabel("growth / width")
        axes[0].set_title("Harris tearing layer gate")
        axes[0].legend(frameon=False, fontsize=7)
        coordinate = result.selected_coordinate
        axes[1].plot(coordinate, result.selected_flux_eigenfunction, label=r"$\psi_1$")
        axes[1].plot(coordinate, result.selected_streamfunction_imag, label=r"Im $\phi_1$")
        axes[1].plot(coordinate, result.selected_current_density, label=r"$j_1$")
        axes[1].set_xlim(-4.0, 4.0)
        axes[1].set_xlabel(r"$x/a$")
        axes[1].set_title(f"reference profiles; frame S={lundquist:.0f}")
        axes[1].legend(frameon=False, fontsize=7)
        figure.suptitle("Literature-anchored Harris tearing eigenfunction localization")
        frames.append(figure_frame(figure))
        plt.close(figure)
    return frames


def plasmoid_scaling_frames() -> list[object]:
    """Return frames for the analytic Sweet--Parker plasmoid schematic."""
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0.0, 2.0 * np.pi, 240)
    y = np.linspace(-1.0, 1.0, 100)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="xy")
    values = np.geomspace(1.0e4, 1.0e6, 8)
    frames = []
    for lundquist in values:
        normalized = lundquist / values[0]
        modes = normalized ** (3.0 / 8.0)
        count = max(2, int(np.ceil(modes)))
        growth = normalized**0.25
        width = 0.22 * normalized**-0.5
        flux = np.tanh(y_mesh / width) + 0.18 * growth / (1.0 + growth) * np.cos(
            count * x_mesh
        ) * np.exp(-((y_mesh / (2.2 * width)) ** 2))
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=100, constrained_layout=True)
        axes[0].contour(x_mesh, y_mesh, flux, levels=20, linewidths=0.7, cmap="RdBu_r")
        axes[0].set_title(f"schematic chain: N≈{count}")
        axes[0].set_xlabel("current-sheet direction")
        axes[0].set_ylabel("inflow direction")
        axes[1].loglog(values, (values / values[0]) ** 0.25, label=r"$S^{1/4}$")
        axes[1].loglog(values, (values / values[0]) ** (3.0 / 8.0), label=r"$S^{3/8}$")
        axes[1].scatter([lundquist], [growth], s=40)
        axes[1].scatter([lundquist], [modes], s=40)
        axes[1].set_xlabel("global Lundquist number S")
        axes[1].set_ylabel("relative scaling")
        axes[1].set_title("Sweet--Parker plasmoid theory")
        axes[1].legend(frameon=False, fontsize=7)
        figure.suptitle("Loureiro--Schekochihin--Cowley plasmoid scaling schematic")
        frames.append(figure_frame(figure))
        plt.close(figure)
    return frames


def turbulence_schematic_frames() -> list[object]:
    """Return deterministic synthetic cascade-guide frames."""
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(20260513)
    n = 96
    coordinates = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    modes = []
    for kx in range(1, 7):
        for ky in range(1, 7):
            radius = float(np.hypot(kx, ky))
            modes.append(
                (
                    kx,
                    ky,
                    radius ** (-5.0 / 3.0),
                    rng.uniform(0, 2 * np.pi),
                    rng.normal(0, 0.35),
                    radius,
                )
            )
    frames = []
    spectrum_k = np.arange(1, 26, dtype=float)
    for frame_index, phase_shift in enumerate(np.linspace(0.0, 2.0 * np.pi, 8)):
        flux = np.zeros_like(x)
        current = np.zeros_like(x)
        for kx, ky, amplitude, phase, drift, radius in modes:
            angle = kx * x + ky * y + phase + drift * phase_shift
            flux += amplitude * np.sin(angle)
            current += radius**2 * amplitude * np.sin(angle)
        spectrum = spectrum_k ** (-5.0 / 3.0)
        cutoff = 1.0 / (1.0 + np.exp(-(spectrum_k - (5 + frame_index)) / 1.8))
        figure, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), dpi=100, constrained_layout=True)
        axes[0].imshow(flux.T, origin="lower", cmap="RdBu_r")
        axes[0].set_title("magnetic-flux eddies")
        axes[1].imshow(current.T, origin="lower", cmap="RdBu_r")
        axes[1].set_title("current filaments")
        for axis in axes[:2]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[2].loglog(spectrum_k, spectrum, label=r"$k^{-5/3}$ guide")
        axes[2].loglog(spectrum_k, spectrum * cutoff, label="animated cascade")
        axes[2].set_xlabel("wavenumber")
        axes[2].set_ylabel("relative power")
        axes[2].set_title("turbulent transfer")
        axes[2].legend(frameon=False, fontsize=7)
        figure.suptitle("MHD turbulence schematic")
        frames.append(figure_frame(figure))
        plt.close(figure)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=README_PREVIEW)
    args = parser.parse_args()
    import imageio.v2 as imageio

    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = (
        ("harris_layer_sweep.gif", harris_layer_frames(), 850),
        ("plasmoid_scaling_schematic.gif", plasmoid_scaling_frames(), 650),
        ("mhd_turbulence_cascade.gif", turbulence_schematic_frames(), 200),
    )
    for name, frames, duration in products:
        path = args.output_dir / name
        imageio.mimsave(path, frames, duration=duration, loop=0, palettesize=64)
        print(path)


if __name__ == "__main__":
    main()
