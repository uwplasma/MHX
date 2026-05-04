"""Plotting helpers for reduced-MHD outputs."""

from __future__ import annotations

from pathlib import Path

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
) -> Path:
    """Plot final magnetic flux contours."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    contours = ax.contour(state.psi, levels=20, linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=6)
    ax.set_xlabel("grid x-index")
    ax.set_ylabel("grid y-index")
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
