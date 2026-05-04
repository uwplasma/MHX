"""Plotting helpers for reduced-MHD outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mhx.diagnostics import trajectory_energies, trajectory_mode_amplitude
from mhx.state import ReducedMHDState, ReducedMHDTrajectory


def plot_energy_history(
    trajectory: ReducedMHDTrajectory,
    *,
    lengths: tuple[float, float],
    path: str | Path,
) -> Path:
    """Plot magnetic, kinetic, and total energy time histories."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energies = trajectory_energies(trajectory, lengths=lengths)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(energies["time"], energies["magnetic"], label=r"$E_B$")
    ax.plot(energies["time"], energies["kinetic"], label=r"$E_K$")
    ax.plot(energies["time"], energies["total"], label=r"$E$")
    ax.set_xlabel("time")
    ax.set_ylabel("mean energy")
    ax.set_title("Reduced-MHD energy history")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_flux_contours(
    state: ReducedMHDState,
    *,
    path: str | Path,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> Path:
    """Plot final magnetic flux contours."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    psi = np.asarray(state.psi)
    if x is None or y is None:
        contours = ax.contour(psi, levels=20, linewidths=0.8)
        ax.set_xlabel("grid x-index")
        ax.set_ylabel("grid y-index")
    else:
        x_mesh, y_mesh = np.meshgrid(np.asarray(x), np.asarray(y), indexing="ij")
        contours = ax.contour(x_mesh, y_mesh, psi, levels=20, linewidths=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.clabel(contours, inline=True, fontsize=6)
    ax.set_title("Final magnetic flux")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_mode_amplitude(
    trajectory: ReducedMHDTrajectory,
    *,
    mode: tuple[int, int],
    path: str | Path,
) -> Path:
    """Plot a Fourier mode-amplitude time history."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    amplitudes = trajectory_mode_amplitude(trajectory, mode=mode)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.semilogy(trajectory.times, amplitudes)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$|\hat\psi_{k_x,k_y}|$")
    ax.set_title(f"Mode amplitude k={mode}")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_flux_gif(
    trajectory: ReducedMHDTrajectory,
    *,
    path: str | Path,
    extent: tuple[float, float, float, float] | None = None,
    duration: float = 0.15,
) -> Path:
    """Write an animated GIF of saved magnetic-flux frames."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    psi_values = np.asarray(trajectory.states.psi)
    vmin = float(np.min(psi_values))
    vmax = float(np.max(psi_values))
    for index, psi in enumerate(psi_values):
        fig, ax = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
        image = ax.imshow(
            psi.T,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        ax.set_title(f"Magnetic flux frame {index}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(output_path, frames, duration=duration)
    return output_path
