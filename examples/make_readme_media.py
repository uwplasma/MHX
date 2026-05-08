"""Generate compact README engagement movies from validated/literature workflows."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mhx.benchmarks import run_linear_tearing_layer_validation


def main() -> None:
    """Write small GIFs used by the README and docs landing pages."""
    output_dir = Path("docs/_static/readme")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_harris_layer_sweep(output_dir / "harris_layer_sweep.gif")
    _write_plasmoid_scaling_schematic(output_dir / "plasmoid_scaling_schematic.gif")


def _write_harris_layer_sweep(path: Path) -> None:
    """Animate the validated Harris eigenfunction-layer sweep over S."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    result = run_linear_tearing_layer_validation(grid_points=128)
    x = result.selected_coordinate
    frame_paths = []
    for lundquist in result.lundquist:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        marker = int(np.argmin(np.abs(result.lundquist - lundquist)))
        axes[0].loglog(
            result.lundquist,
            result.growth_rate,
            "o-",
            color="#3266a8",
            label=r"$\gamma$",
        )
        axes[0].loglog(
            result.lundquist,
            result.stream_half_width,
            "s-",
            color="#b54a4a",
            label="flow width",
        )
        axes[0].scatter(
            [result.lundquist[marker]],
            [result.growth_rate[marker]],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )
        axes[0].set_xlabel("Lundquist number S")
        axes[0].set_ylabel("growth / width")
        axes[0].set_xlim(220.0, 2300.0)
        axes[0].xaxis.set_major_locator(FixedLocator([250.0, 1000.0, 2000.0]))
        axes[0].set_xticklabels(["250", "1000", "2000"])
        axes[0].xaxis.set_minor_formatter(NullFormatter())
        axes[0].set_title("Harris tearing layer gate")
        axes[0].legend(frameon=False, fontsize=7)
        axes[1].plot(x, result.selected_flux_eigenfunction, label=r"$\psi_1$", lw=1.6)
        axes[1].plot(x, result.selected_streamfunction_imag, label=r"Im $\phi_1$", lw=1.6)
        axes[1].plot(x, result.selected_current_density, label=r"$j_1$", lw=1.2)
        axes[1].axvline(0.0, color="0.65", lw=0.8)
        axes[1].set_xlim(-4.0, 4.0)
        axes[1].set_xlabel(r"$x/a$")
        axes[1].set_title(f"reference profiles; frame S={lundquist:.0f}")
        axes[1].legend(frameon=False, fontsize=7)
        fig.suptitle("Literature-anchored Harris tearing eigenfunction localization")
        frame_paths.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frame_paths, duration=0.85, loop=0, palettesize=64)


def _write_plasmoid_scaling_schematic(path: Path) -> None:
    """Animate the Loureiro-Schekochihin-Cowley Sweet-Parker plasmoid scalings."""
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 2.0 * np.pi, 240)
    y = np.linspace(-1.0, 1.0, 100)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="xy")
    lundquist_values = np.geomspace(1.0e4, 1.0e6, 8)
    frames = []
    for lundquist in lundquist_values:
        normalized = lundquist / lundquist_values[0]
        mode_scaling = normalized ** (3.0 / 8.0)
        island_count = max(2, int(np.ceil(mode_scaling)))
        growth = normalized ** 0.25
        sheet_width = 0.22 * normalized ** -0.5
        perturbation = 0.18 * growth / (1.0 + growth)
        flux = np.tanh(y_mesh / sheet_width) + perturbation * np.cos(
            island_count * x_mesh
        ) * np.exp(-((y_mesh / (2.2 * sheet_width)) ** 2))
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        axes[0].contour(x_mesh, y_mesh, flux, levels=20, linewidths=0.7, cmap="viridis")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(f"schematic chain: N≈{island_count}")
        axes[0].set_xlabel("current-sheet direction")
        axes[0].set_ylabel("inflow direction")
        axes[1].loglog(
            lundquist_values,
            (lundquist_values / lundquist_values[0]) ** 0.25,
            "-",
            color="#3266a8",
            label=r"$\gamma_{\max}\tau_A\propto S^{1/4}$",
        )
        axes[1].loglog(
            lundquist_values,
            (lundquist_values / lundquist_values[0]) ** (3.0 / 8.0),
            "-",
            color="#b54a4a",
            label=r"$k_{\max}L\propto S^{3/8}$",
        )
        axes[1].scatter([lundquist], [growth], color="#3266a8", s=45)
        axes[1].scatter([lundquist], [mode_scaling], color="#b54a4a", s=45)
        axes[1].set_xlabel("global Lundquist number S")
        axes[1].set_ylabel("relative scaling")
        axes[1].set_title("Sweet-Parker plasmoid theory")
        axes[1].legend(frameon=False, fontsize=7)
        fig.suptitle("Loureiro-Schekochihin-Cowley plasmoid scaling schematic")
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=0.65, loop=0, palettesize=64)


def _figure_to_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


if __name__ == "__main__":
    main()
