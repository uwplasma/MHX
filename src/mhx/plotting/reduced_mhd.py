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


def plot_decay_amplitude(times, numerical, exact, *, path: str | Path) -> Path:
    """Plot exact and numerical resistive-decay mode amplitudes."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.semilogy(np.asarray(times), np.asarray(numerical), "o", label="MHX RK4")
    ax.semilogy(np.asarray(times), np.asarray(exact), "-", label=r"exact $A_0 e^{-\eta k^2 t}$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$|\hat\psi_k|$")
    ax.set_title("Exact resistive decay gate")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_decay_relative_error(
    times,
    amplitude_error,
    energy_error,
    *,
    path: str | Path,
) -> Path:
    """Plot relative errors for exact resistive-decay validation."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.semilogy(np.asarray(times), np.asarray(amplitude_error), "o-", label="amplitude")
    ax.semilogy(np.asarray(times), np.asarray(energy_error), "s-", label="energy")
    ax.set_xlabel("time")
    ax.set_ylabel("relative error")
    ax.set_title("Resistive decay numerical error")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_decay_energy(times, numerical, exact, *, path: str | Path) -> Path:
    """Plot exact and numerical magnetic-energy decay."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.semilogy(np.asarray(times), np.asarray(numerical), "o", label="MHX RK4")
    ax.semilogy(np.asarray(times), np.asarray(exact), "-", label=r"exact $E_0 e^{-2\eta k^2 t}$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$E_B$")
    ax.set_title("Magnetic-energy decay")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_fkr_scaling(lundquist, gamma, inner_width, *, path: str | Path) -> Path:
    """Plot FKR constant-psi tearing scalings."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s_values = np.asarray(lundquist)
    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    ax.loglog(s_values, np.asarray(gamma), "o-", label=r"$\gamma\tau_a \sim S_a^{-3/5}$")
    ax.loglog(s_values, np.asarray(inner_width), "s-", label=r"$\delta/a \sim S_a^{-2/5}$")
    ax.set_xlabel(r"local Lundquist number $S_a$")
    ax.set_ylabel("dimensionless scaling")
    ax.set_title(r"FKR constant-$\psi$ scaling gate")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_fkr_validity_window(
    ka,
    gamma,
    constant_psi_product,
    *,
    max_constant_psi_product: float,
    path: str | Path,
) -> Path:
    """Plot a constant-psi FKR regime-window diagnostic."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ka_values = np.asarray(ka)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].plot(ka_values, np.asarray(gamma), "o-", color="#3266a8")
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_ylabel(r"$\gamma\tau_a$")
    axes[0].set_title(r"FKR growth estimate")
    axes[1].semilogy(
        ka_values,
        np.asarray(constant_psi_product),
        "s-",
        color="#8c4fb4",
        label=r"$\Delta'\delta$",
    )
    axes[1].axhline(
        max_constant_psi_product,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_ylabel(r"constant-$\psi$ product")
    axes[1].set_title(r"FKR validity gate")
    axes[1].legend(frameon=False)
    fig.suptitle(r"Constant-$\psi$ tearing regime window")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_plasmoid_scaling(lundquist, gamma, fastest_mode, *, path: str | Path) -> Path:
    """Plot Loureiro Sweet-Parker plasmoid scalings."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s_values = np.asarray(lundquist)
    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    ax.loglog(s_values, np.asarray(gamma), "o-", label=r"$\gamma_{\max}\tau_A \sim S^{1/4}$")
    ax.loglog(s_values, np.asarray(fastest_mode), "s-", label=r"$k_{\max}L \sim S^{3/8}$")
    ax.set_xlabel(r"global Lundquist number $S$")
    ax.set_ylabel("dimensionless scaling")
    ax.set_title("Sweet-Parker plasmoid scaling gate")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_ideal_tearing_scaling(lundquist, aspect_ratio, *, path: str | Path) -> Path:
    """Plot Pucci-Velli ideal-tearing aspect-ratio scaling."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s_values = np.asarray(lundquist)
    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    ax.loglog(s_values, np.asarray(aspect_ratio), "o-", label=r"$a/L \sim S^{-1/3}$")
    ax.set_xlabel(r"global Lundquist number $S$")
    ax.set_ylabel(r"sheet aspect ratio $a/L$")
    ax.set_title("Ideal-tearing aspect-ratio scaling gate")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_timing_summary(
    case_names,
    durations_seconds,
    peak_memory_mib,
    *,
    path: str | Path,
) -> Path:
    """Plot FAST timing wall-clock and Python memory summaries."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(case_names)
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].barh(positions, np.asarray(durations_seconds), color="#3266a8")
    axes[0].set_yticks(positions, labels=names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("median wall time [s]")
    axes[0].set_title("FAST runtime")
    axes[1].barh(positions, np.asarray(peak_memory_mib), color="#8c4fb4")
    axes[1].set_yticks(positions, labels=[])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("peak tracemalloc [MiB]")
    axes[1].set_title("Python allocation peak")
    fig.suptitle("MHX FAST benchmark timing")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
