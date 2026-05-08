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


def plot_fkr_growth_rate_validation(
    lundquist,
    gamma_vs_lundquist,
    ka,
    delta_prime,
    gamma_vs_delta_prime,
    gamma_relative_error,
    *,
    max_gamma_relative_error: float,
    path: str | Path,
) -> Path:
    """Plot the FKR growth-rate scaling validation."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s_values = np.asarray(lundquist)
    gamma_s = np.asarray(gamma_vs_lundquist)
    ka_values = np.asarray(ka)
    delta_prime_values = np.asarray(delta_prime)
    gamma_delta = np.asarray(gamma_vs_delta_prime)
    relative_error = np.asarray(gamma_relative_error)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)
    axes[0].loglog(s_values, gamma_s, "o-", color="#3266a8", label="assembled FKR")
    reference = gamma_s[0] * (s_values / s_values[0]) ** (-3.0 / 5.0)
    axes[0].loglog(s_values, reference, "--", color="black", label=r"$S_a^{-3/5}$")
    axes[0].set_xlabel(r"$S_a$")
    axes[0].set_ylabel(r"$\gamma\tau_a$")
    axes[0].set_title("Lundquist scaling")
    axes[0].legend(frameon=False)

    normalized = gamma_delta / (ka_values ** (2.0 / 5.0))
    axes[1].loglog(delta_prime_values, normalized, "s-", color="#4b8f5a")
    reference_delta = normalized[0] * (delta_prime_values / delta_prime_values[0]) ** (
        4.0 / 5.0
    )
    axes[1].loglog(
        delta_prime_values,
        reference_delta,
        "--",
        color="black",
        label=r"$(\Delta'a)^{4/5}$",
    )
    axes[1].set_xlabel(r"numerical $\Delta'a$")
    axes[1].set_ylabel(r"$\gamma\tau_a/(ka)^{2/5}$")
    axes[1].set_title(r"$\Delta'$ response")
    axes[1].legend(frameon=False)

    axes[2].semilogy(ka_values, relative_error, "o-", color="#8c4fb4")
    axes[2].axhline(
        max_gamma_relative_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    axes[2].set_xlabel(r"$ka$")
    axes[2].set_ylabel("relative growth error")
    axes[2].set_title("Numerical Δ′ propagation")
    axes[2].legend(frameon=False)
    fig.suptitle(r"FKR constant-$\psi$ growth-rate gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_harris_delta_prime(
    ka,
    numerical_delta_prime,
    analytic_delta_prime,
    relative_error,
    *,
    max_relative_error: float,
    path: str | Path,
) -> Path:
    """Plot numerical Harris-sheet outer-region Delta-prime validation."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ka_values = np.asarray(ka)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].plot(
        ka_values,
        np.asarray(analytic_delta_prime),
        "-",
        color="black",
        label=r"analytic $2[(ka)^{-1}-ka]$",
    )
    axes[0].plot(
        ka_values,
        np.asarray(numerical_delta_prime),
        "o",
        color="#3266a8",
        label="numerical outer solve",
    )
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_ylabel(r"$\Delta' a$")
    axes[0].set_title("Harris-sheet outer matching")
    axes[0].legend(frameon=False)
    axes[1].semilogy(ka_values, np.asarray(relative_error), "o-", color="#8c4fb4")
    axes[1].axhline(
        max_relative_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_ylabel("relative error")
    axes[1].set_title(r"Numerical $\Delta'$ error")
    axes[1].legend(frameon=False)
    fig.suptitle(r"FKR Harris-sheet $\Delta'$ gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_linear_tearing_eigenvalue_validation(
    dx,
    growth_rates,
    fitted_growth_rates,
    *,
    reference_growth_rate: float,
    extrapolated_growth_rate: float,
    spectrum,
    selected_eigenvalue: complex,
    coordinate,
    flux_eigenfunction,
    streamfunction_imag,
    path: str | Path,
) -> Path:
    """Plot the direct Harris-sheet tearing eigenvalue validation."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dx_values = np.asarray(dx)
    growth = np.asarray(growth_rates)
    fitted = np.asarray(fitted_growth_rates)
    spectrum_values = np.asarray(spectrum)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), constrained_layout=True)

    axes[0].plot(dx_values**2, growth, "o", color="#3266a8", label="dense FD eigenvalue")
    order = np.argsort(dx_values**2)
    axes[0].plot(dx_values[order] ** 2, fitted[order], "-", color="#4b8f5a", label="linear in Δx²")
    axes[0].axhline(
        reference_growth_rate,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="literature",
    )
    axes[0].axhline(
        extrapolated_growth_rate,
        color="#8c4fb4",
        linestyle=":",
        linewidth=1.4,
        label="Δx→0",
    )
    axes[0].set_xlabel(r"$\Delta x^2$")
    axes[0].set_ylabel(r"growth rate $\gamma$")
    axes[0].set_title("Grid extrapolation")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(spectrum_values.real, spectrum_values.imag, s=8, alpha=0.55, color="#3266a8")
    axes[1].scatter(
        [selected_eigenvalue.real],
        [selected_eigenvalue.imag],
        marker="x",
        s=58,
        color="#b54a4a",
        label="tearing mode",
    )
    axes[1].axhline(0.0, color="0.85", linewidth=0.8)
    axes[1].axvline(0.0, color="0.85", linewidth=0.8)
    axes[1].set_xlabel(r"$\operatorname{Re}\lambda$")
    axes[1].set_ylabel(r"$\operatorname{Im}\lambda$")
    axes[1].set_title("Selected spectrum")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(coordinate, flux_eigenfunction, color="#3266a8", label=r"$b(x)$")
    axes[2].plot(coordinate, streamfunction_imag, color="#8c4fb4", label=r"$\operatorname{Im}u(x)$")
    axes[2].axhline(0.0, color="0.85", linewidth=0.8)
    axes[2].axvline(0.0, color="0.85", linewidth=0.8)
    axes[2].set_xlabel(r"$x/a$")
    axes[2].set_ylabel("normalized eigenfunction")
    axes[2].set_title("Tearing parity")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Direct Harris-sheet tearing eigenvalue gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_linear_tearing_dispersion_validation(
    wavenumber,
    growth_rate,
    eigenvalue_imag,
    residual_norm,
    *,
    reference_wavenumber: float,
    reference_growth_rate: float,
    max_residual_norm: float,
    path: str | Path,
) -> Path:
    """Plot the finite-domain Harris-sheet tearing dispersion validation."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    k_values = np.asarray(wavenumber)
    growth_values = np.asarray(growth_rate)
    imag_values = np.asarray(eigenvalue_imag)
    residual_values = np.asarray(residual_norm)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), constrained_layout=True)

    axes[0].plot(k_values, growth_values, "o-", color="#3266a8", label="selected branch")
    axes[0].scatter(
        [reference_wavenumber],
        [reference_growth_rate],
        marker="x",
        s=64,
        color="#b54a4a",
        label="literature anchor",
    )
    axes[0].axhline(0.0, color="0.7", linewidth=0.8)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.0, label=r"$ka=1$")
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_ylabel(r"$\operatorname{Re}\lambda$")
    axes[0].set_title("Growth branch")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(k_values, np.abs(imag_values), "o-", color="#4b8f5a")
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_ylabel(r"$|\operatorname{Im}\lambda|$")
    axes[1].set_title("Oscillatory stable controls")

    axes[2].semilogy(k_values, residual_values, "s-", color="#8c4fb4", label="residual")
    axes[2].axhline(max_residual_norm, color="black", linestyle="--", linewidth=1.0, label="gate")
    axes[2].set_xlabel(r"$ka$")
    axes[2].set_ylabel(r"$\|Lv-\lambda v\|_2/\|v\|_2$")
    axes[2].set_title("Dense eigenpair residual")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Finite-domain Harris-sheet tearing dispersion gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_linear_tearing_layer_validation(
    lundquist,
    growth_rate,
    stream_half_width,
    current_half_width,
    flux_half_width,
    residual_norm,
    *,
    selected_coordinate,
    selected_flux_eigenfunction,
    selected_streamfunction_imag,
    selected_current_density,
    stream_width_slope: float,
    growth_rate_slope: float,
    max_residual_norm: float,
    path: str | Path,
) -> Path:
    """Plot the Harris-sheet tearing eigenfunction localization gate."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s_values = np.asarray(lundquist)
    growth_values = np.asarray(growth_rate)
    stream_width_values = np.asarray(stream_half_width)
    current_width_values = np.asarray(current_half_width)
    flux_width_values = np.asarray(flux_half_width)
    residual_values = np.asarray(residual_norm)
    coordinate = np.asarray(selected_coordinate)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    axes[0, 0].loglog(s_values, growth_values, "o-", color="#3266a8")
    axes[0, 0].set_xlabel(r"$S$")
    axes[0, 0].set_ylabel(r"$\operatorname{Re}\lambda$")
    axes[0, 0].set_title(rf"Growth trend, slope {growth_rate_slope:.2f}")

    axes[0, 1].loglog(
        s_values,
        stream_width_values,
        "o-",
        color="#8c4fb4",
        label=r"flow half width",
    )
    axes[0, 1].loglog(
        s_values,
        current_width_values,
        "s-",
        color="#4b8f5a",
        label=r"current half width",
    )
    axes[0, 1].loglog(
        s_values,
        flux_width_values,
        "^-",
        color="#b58b3b",
        label=r"flux half width",
    )
    axes[0, 1].set_xlabel(r"$S$")
    axes[0, 1].set_ylabel(r"half-maximum width")
    axes[0, 1].set_title(rf"Layer narrowing, flow slope {stream_width_slope:.2f}")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(
        coordinate,
        np.asarray(selected_flux_eigenfunction),
        color="#3266a8",
        label=r"$b(x)$",
    )
    axes[1, 0].plot(
        coordinate,
        np.asarray(selected_streamfunction_imag),
        color="#8c4fb4",
        label=r"$\operatorname{Im}u(x)$",
    )
    axes[1, 0].plot(
        coordinate,
        np.asarray(selected_current_density),
        color="#4b8f5a",
        label=r"$-\left(d_x^2-k^2\right)b$",
    )
    axes[1, 0].axhline(0.0, color="0.85", linewidth=0.8)
    axes[1, 0].axvline(0.0, color="0.85", linewidth=0.8)
    axes[1, 0].set_xlabel(r"$x/a$")
    axes[1, 0].set_ylabel("normalized profile")
    axes[1, 0].set_title("Reference eigenfunction localization")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].semilogy(s_values, residual_values, "s-", color="#8c4fb4", label="residual")
    axes[1, 1].axhline(
        max_residual_norm,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    axes[1, 1].set_xlabel(r"$S$")
    axes[1, 1].set_ylabel(r"$\|Lv-\lambda v\|_2/\|v\|_2$")
    axes[1, 1].set_title("Dense eigenpair residual")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("Harris-sheet tearing eigenfunction layer gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_linear_tearing_timedomain_validation(
    times,
    amplitude,
    exact_amplitude,
    relative_amplitude_error,
    *,
    expected_growth_rate: float,
    fitted_growth_rate: float,
    max_relative_amplitude_error: float,
    path: str | Path,
) -> Path:
    """Plot the Harris-sheet tearing time-domain eigenmode replay gate."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_values = np.asarray(times)
    amplitude_values = np.asarray(amplitude)
    exact_values = np.asarray(exact_amplitude)
    error_values = np.asarray(relative_amplitude_error)
    fitted_values = amplitude_values[0] * np.exp(fitted_growth_rate * time_values)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), constrained_layout=True)
    axes[0].semilogy(time_values, amplitude_values, "o", markersize=3.0, label="RK4 replay")
    axes[0].semilogy(time_values, exact_values, "-", label=r"$\exp(\gamma t)$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\|q(t)\|_2/\|q(0)\|_2$")
    axes[0].set_title("Eigenmode amplitude")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(time_values, np.log(amplitude_values), "o", markersize=3.0, label="measured")
    axes[1].plot(time_values, np.log(fitted_values), "-", label="fit")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$\log \|q(t)\|_2$")
    axes[1].set_title(
        rf"$\gamma_\mathrm{{eig}}={expected_growth_rate:.5f}$, "
        rf"$\gamma_\mathrm{{fit}}={fitted_growth_rate:.5f}$"
    )
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].semilogy(time_values, error_values, "s-", markersize=3.0, color="#8c4fb4")
    axes[2].axhline(
        max_relative_amplitude_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("relative amplitude error")
    axes[2].set_title("RK4 versus exact eigen-growth")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Time-domain Harris-sheet tearing eigenmode replay")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_linearized_rhs_errors(
    component_names,
    relative_errors,
    *,
    max_relative_error: float,
    path: str | Path,
) -> Path:
    """Plot JVP versus finite-difference linearized-RHS relative errors."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(component_names)
    errors = np.asarray(relative_errors)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.semilogy(names, errors, "o", color="#3266a8")
    ax.axhline(
        max_relative_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    ax.set_ylabel("relative L2 error")
    ax.set_title("Linearized RHS JVP consistency")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_diffusion_eigenvalue_error(
    quantities,
    values,
    thresholds,
    *,
    path: str | Path,
) -> Path:
    """Plot diffusion eigenvalue benchmark errors and gates."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(quantities)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.semilogy(positions, np.asarray(values), "o", color="#3266a8", label="measured")
    ax.semilogy(positions, np.asarray(thresholds), "x", color="#8c4fb4", label="gate")
    ax.set_xticks(positions, labels=names)
    ax.set_ylabel("absolute / relative error")
    ax.set_title("Matrix-free diffusion eigenvalue gate")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_power_iteration_history(
    iterations,
    rayleigh_history,
    residual_history,
    *,
    expected_eigenvalue: float,
    path: str | Path,
) -> Path:
    """Plot power-iteration Rayleigh quotient and eigen-residual histories."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(iterations)
    rayleigh_values = np.asarray(rayleigh_history)
    residual_values = np.asarray(residual_history)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].plot(steps, rayleigh_values, "o-", color="#3266a8")
    axes[0].axhline(expected_eigenvalue, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("Rayleigh quotient")
    axes[0].set_title("Dominant eigenvalue estimate")
    axes[1].semilogy(steps, residual_values, "s-", color="#8c4fb4")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title("Eigen-residual")
    fig.suptitle("Power-iteration smoke benchmark")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_arnoldi_ritz_values(
    expected_eigenvalues,
    ritz_values,
    residual_estimates,
    *,
    path: str | Path,
) -> Path:
    """Plot Arnoldi Ritz values and residual estimates for validation."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected = np.asarray(expected_eigenvalues)
    measured = np.asarray(ritz_values)
    residuals = np.asarray(residual_estimates)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].scatter(expected.real, expected.imag, marker="x", color="black", label="expected")
    axes[0].scatter(measured.real, measured.imag, marker="o", color="#3266a8", label="Ritz")
    axes[0].axhline(0.0, color="0.8", linewidth=0.8)
    axes[0].axvline(0.0, color="0.8", linewidth=0.8)
    axes[0].set_xlabel(r"$\operatorname{Re}\lambda$")
    axes[0].set_ylabel(r"$\operatorname{Im}\lambda$")
    axes[0].set_title("Fixture spectrum")
    axes[0].legend(frameon=False)
    axes[1].semilogy(np.arange(1, residuals.size + 1), residuals, "o-", color="#8c4fb4")
    axes[1].set_xlabel("Ritz value index")
    axes[1].set_ylabel("Arnoldi residual estimate")
    axes[1].set_title("Ritz residual estimates")
    fig.suptitle("Arnoldi matrix-free eigensolver gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_reduced_mhd_eigenmode_errors(
    quantities,
    values,
    thresholds,
    *,
    path: str | Path,
) -> Path:
    """Plot reduced-MHD linear eigenmode validation errors and gates."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(quantities)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    ax.semilogy(positions, np.asarray(values), "o", color="#3266a8", label="measured")
    ax.semilogy(positions, np.asarray(thresholds), "x", color="#8c4fb4", label="gate")
    ax.set_xticks(positions, labels=names, rotation=15, ha="right")
    ax.set_ylabel("absolute / relative error")
    ax.set_title("Reduced-MHD linear eigenmode gate")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_cosine_equilibrium_linearization_errors(
    quantities,
    values,
    thresholds,
    *,
    path: str | Path,
) -> Path:
    """Plot nonzero-equilibrium reduced-MHD linearization errors and gates."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(quantities)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.1), constrained_layout=True)
    ax.semilogy(positions, np.asarray(values), "o", color="#3266a8", label="measured")
    ax.semilogy(positions, np.asarray(thresholds), "x", color="#8c4fb4", label="gate")
    ax.set_xticks(positions, labels=names, rotation=18, ha="right")
    ax.set_ylabel("relative L2 error")
    ax.set_title(r"Nonzero-equilibrium JVP gate for $\psi_0=A\cos y$")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_periodic_current_sheet_spectrum(
    eigenvalues,
    *,
    selected_eigenvalue: complex,
    max_allowed_real_part: float,
    residual_norm: float,
    max_residual_norm: float,
    path: str | Path,
) -> Path:
    """Plot the tiny dense spectrum for the periodic current-sheet gate."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(eigenvalues)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    axes[0].scatter(values.real, values.imag, s=14, alpha=0.72, color="#3266a8")
    axes[0].scatter(
        [selected_eigenvalue.real],
        [selected_eigenvalue.imag],
        s=52,
        marker="x",
        color="#b54a4a",
        label="selected mode",
    )
    axes[0].axvline(
        max_allowed_real_part,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="non-gauge gate",
    )
    axes[0].axhline(0.0, color="0.85", linewidth=0.8)
    axes[0].axvline(0.0, color="0.85", linewidth=0.8)
    axes[0].set_xlabel(r"$\operatorname{Re}\lambda$")
    axes[0].set_ylabel(r"$\operatorname{Im}\lambda$")
    axes[0].set_title(r"Linear spectrum around $\psi_0=\cos y$")
    axes[0].legend(frameon=False)
    axes[1].semilogy(
        ["selected residual"],
        [residual_norm],
        "o",
        color="#3266a8",
        label="measured",
    )
    axes[1].semilogy(
        ["selected residual"],
        [max_residual_norm],
        "x",
        color="#8c4fb4",
        label="gate",
    )
    axes[1].set_ylabel(r"$\|Lv-\lambda v\|_2/\|v\|_2$")
    axes[1].set_title("Dense eigenpair residual")
    axes[1].legend(frameon=False)
    fig.suptitle("Periodic current-sheet eigenvalue gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_periodic_current_sheet_timedomain(
    times,
    amplitudes,
    exact_amplitudes,
    relative_state_error,
    *,
    selected_eigenvalue: float,
    fitted_decay_rate: float,
    max_relative_state_error: float,
    path: str | Path,
) -> Path:
    """Plot periodic-current-sheet linear eigenmode time-domain replay diagnostics."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_values = np.asarray(times)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True)
    axes[0].semilogy(
        time_values,
        np.asarray(amplitudes),
        "o",
        color="#3266a8",
        label="RK4 replay",
    )
    axes[0].semilogy(
        time_values,
        np.asarray(exact_amplitudes),
        "-",
        color="#b54a4a",
        label=r"$\exp(\lambda t)$",
    )
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\|q(t)\|_2$")
    axes[0].set_title(
        rf"Decaying mode: $\lambda={selected_eigenvalue:.4g}$, "
        rf"fit={fitted_decay_rate:.4g}"
    )
    axes[0].legend(frameon=False)
    axes[1].semilogy(
        time_values,
        np.asarray(relative_state_error),
        "o-",
        color="#3266a8",
        label="state error",
    )
    axes[1].axhline(
        max_relative_state_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$\|q_{\rm RK4}-q_{\rm exact}\|_2/\|q_{\rm exact}\|_2$")
    axes[1].set_title("Replay error")
    axes[1].legend(frameon=False)
    fig.suptitle("Periodic current-sheet time-domain eigenmode replay")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_nonlinear_current_sheet_bridge(
    epsilons,
    relative_errors,
    *,
    convergence_order: float,
    min_convergence_order: float,
    max_finest_relative_error: float,
    path: str | Path,
) -> Path:
    """Plot nonlinear current-sheet trajectory-map JVP convergence."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epsilon_values = np.asarray(epsilons)
    error_values = np.asarray(relative_errors)
    reference = error_values[-1] * (epsilon_values / epsilon_values[-1]) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), constrained_layout=True)
    axes[0].loglog(
        epsilon_values,
        error_values,
        "o-",
        color="#3266a8",
        label="centered FD vs JVP",
    )
    axes[0].loglog(
        epsilon_values,
        reference,
        "--",
        color="#b54a4a",
        label=r"$O(\epsilon^2)$",
    )
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"finite-difference amplitude $\epsilon$")
    axes[0].set_ylabel("relative tangent error")
    axes[0].set_title("Nonlinear RK4 map derivative")
    axes[0].legend(frameon=False)
    axes[1].bar(
        ["measured order", "gate"],
        [convergence_order, min_convergence_order],
        color=["#3266a8", "#8c4fb4"],
    )
    axes[1].axhline(2.0, color="0.4", linestyle=":", linewidth=1.0)
    axes[1].set_ylim(0.0, max(2.3, 1.15 * convergence_order))
    axes[1].set_ylabel("log-log slope")
    axes[1].set_title(
        f"finest error={error_values[-1]:.2e}, gate={max_finest_relative_error:.1e}"
    )
    fig.suptitle("Nonlinear current-sheet differentiability bridge")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_nonlinear_energy_budget(
    times,
    total_energy_values,
    current_dissipation,
    viscous_dissipation,
    relative_budget_residual,
    *,
    initial_psi,
    final_psi,
    x,
    y,
    max_budget_residual: float,
    path: str | Path,
) -> Path:
    """Plot nonlinear reduced-MHD energy-budget validation diagnostics."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_values = np.asarray(times)
    total = np.asarray(total_energy_values)
    current = np.asarray(current_dissipation)
    viscous = np.asarray(viscous_dissipation)
    residual = np.asarray(relative_budget_residual)
    x_mesh, y_mesh = np.meshgrid(np.asarray(x), np.asarray(y), indexing="ij")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    axes[0, 0].plot(time_values, total, "o-", color="#3266a8", label=r"$E$")
    axes[0, 0].plot(
        time_values,
        total[0] - _cumulative_trapezoid_for_plot(time_values, current + viscous),
        "--",
        color="#b54a4a",
        label=r"$E(0)-\int D\,dt$",
    )
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("mean energy")
    axes[0, 0].set_title("Energy and integrated dissipation")
    axes[0, 0].legend(frameon=False)
    residual_floor = max(max_budget_residual * 1.0e-7, np.finfo(float).eps)
    residual_for_plot = np.maximum(residual, residual_floor)
    axes[0, 1].semilogy(
        time_values,
        residual_for_plot,
        "o-",
        color="#3266a8",
        label="relative residual",
    )
    axes[0, 1].axhline(
        max_budget_residual,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="gate",
    )
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(r"$|E-E_0+\int Ddt|/E_0$")
    axes[0, 1].set_title("Budget residual")
    axes[0, 1].set_ylim(residual_floor * 0.5, max_budget_residual * 5.0)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(time_values, current, color="#3266a8", label=r"$\eta\langle j^2\rangle$")
    axes[1, 0].plot(time_values, viscous, color="#b54a4a", label=r"$\nu\langle\omega^2\rangle$")
    axes[1, 0].plot(time_values, current + viscous, color="black", label="total")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("dissipation rate")
    axes[1, 0].set_title("Dissipation channels")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].contour(
        x_mesh,
        y_mesh,
        np.asarray(initial_psi),
        levels=16,
        colors="#3266a8",
        linewidths=0.7,
    )
    axes[1, 1].contour(
        x_mesh,
        y_mesh,
        np.asarray(final_psi),
        levels=16,
        colors="#b54a4a",
        linewidths=0.7,
    )
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_title("Flux contours: initial blue, final red")
    fig.suptitle("Nonlinear reduced-MHD energy-budget gate")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _cumulative_trapezoid_for_plot(times, values) -> np.ndarray:
    time_values = np.asarray(times)
    value_array = np.asarray(values)
    integral = np.zeros_like(time_values, dtype=float)
    for index in range(1, time_values.size):
        integral[index] = integral[index - 1] + 0.5 * (
            value_array[index - 1] + value_array[index]
        ) * (time_values[index] - time_values[index - 1])
    return integral


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
