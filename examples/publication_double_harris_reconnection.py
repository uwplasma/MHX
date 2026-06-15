"""Publication-style FAST double-Harris nonlinear reconnection example.

Edit the parameters below, then run:

    python examples/publication_double_harris_reconnection.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

from mhx.benchmarks import write_periodic_double_harris_seeded_long_run_validation
from mhx.io import write_manifest
from mhx.runtime import configure_jax

CI_SMOKE = False
FAST_MODE = os.environ.get("MHX_EXAMPLE_FAST", "0") == "1"
OUTDIR_ROOT = Path(
    os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples/publication")
).expanduser()
RUN_DIR = OUTDIR_ROOT / "double_harris_reconnection"

SHAPE = (16, 16) if FAST_MODE else (64, 64)
WIDTH = 0.4 if FAST_MODE else 0.36
RESISTIVITY = 5.0e-3
VISCOSITY = 5.0e-3
PERTURBATION_AMPLITUDE = 1.0e-3 if FAST_MODE else 4.0e-3
PERTURBATION_MODE = (2, 1)
DT = 2.0e-2
T_END = 2.0 if FAST_MODE else 60.0
SAVE_EVERY = 10 if FAST_MODE else 100
FIT_WINDOW = (0.0, min(12.0, T_END))
WRITE_MOVIES = True
DELTA_MOVIE_MAX_FRAMES = 32
FIGURE_DPI = 220

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

configure_jax(enable_x64=True)
manifest_path, validation = write_periodic_double_harris_seeded_long_run_validation(
    RUN_DIR,
    movies=WRITE_MOVIES,
    shape=SHAPE,
    width=WIDTH,
    resistivity=RESISTIVITY,
    viscosity=VISCOSITY,
    perturbation_amplitude=PERTURBATION_AMPLITUDE,
    perturbation_mode=PERTURBATION_MODE,
    dt=DT,
    t_end=T_END,
    save_every=SAVE_EVERY,
    fit_window=FIT_WINDOW,
    min_saved_samples=5 if FAST_MODE else 10,
    min_early_growth_factor=1.001 if FAST_MODE else 1.05,
    min_max_growth_factor=1.001 if FAST_MODE else 1.05,
    min_reconnected_flux_amplification=1.000001,
    min_island_width_amplification=1.000001,
)

diagnostics = json.loads((RUN_DIR / "diagnostics.json").read_text(encoding="utf-8"))
with np.load(RUN_DIR / "periodic_double_harris_seeded_long_run.npz", allow_pickle=False) as history:
    time = np.asarray(history["time"], dtype=float)
    perturbation_norm = np.asarray(history["perturbation_norm"], dtype=float)
    reconnected_flux = np.asarray(history["reconnected_flux"], dtype=float)
    island_width = np.asarray(history["rutherford_island_width"], dtype=float)
    magnetic_energy = np.asarray(history["magnetic_energy"], dtype=float)
    kinetic_energy = np.asarray(history["kinetic_energy"], dtype=float)
    total_energy = np.asarray(history["total_energy"], dtype=float)
    current_linf = np.asarray(history["current_density_linf"], dtype=float)
    base_flux = np.asarray(history["base_psi"], dtype=float)
    perturbed_flux = np.asarray(history["perturbed_psi"], dtype=float)
    final_flux = np.asarray(history["perturbed_psi"][-1], dtype=float)
    final_flux_delta = perturbed_flux[-1] - base_flux[-1]

summary_path = RUN_DIR / "figures" / "publication_double_harris_reconnection_summary.png"
delta_movie_path = RUN_DIR / "figures" / "publication_double_harris_delta_flux.gif"
summary_path.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

growth_reference = perturbation_norm[0] * np.exp(
    diagnostics["fitted_early_growth_rate"] * (time - time[0])
)
axes[0, 0].semilogy(time, perturbation_norm, "o-", ms=3.5, label="nonlinear difference")
axes[0, 0].semilogy(time, growth_reference, "--", lw=2.0, label="early exponential fit")
axes[0, 0].set_title("Seeded double-Harris growth")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel(r"$||\Delta(\psi,\omega)||_2$")
axes[0, 0].grid(True, alpha=0.25)
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(time, reconnected_flux, color="tab:blue", label="reconnected flux")
island_axis = axes[0, 1].twinx()
island_axis.plot(time, island_width, color="tab:orange", label="island width")
axes[0, 1].set_title("Reconnection proxies")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("flux amplitude")
island_axis.set_ylabel("Rutherford width proxy")
axes[0, 1].grid(True, alpha=0.25)
handles, labels = axes[0, 1].get_legend_handles_labels()
extra_handles, extra_labels = island_axis.get_legend_handles_labels()
axes[0, 1].legend(handles + extra_handles, labels + extra_labels, frameon=False)

axes[1, 0].plot(time, magnetic_energy, label=r"$E_B$")
axes[1, 0].plot(time, kinetic_energy, label=r"$E_K$")
axes[1, 0].plot(time, total_energy, label=r"$E$")
current_axis = axes[1, 0].twinx()
current_axis.plot(time, current_linf, ":", color="tab:red", label=r"$||j_z||_\infty$")
axes[1, 0].set_title("Energy and current")
axes[1, 0].set_xlabel("time")
axes[1, 0].set_ylabel("mean energy")
current_axis.set_ylabel("current density")
axes[1, 0].grid(True, alpha=0.25)
handles, labels = axes[1, 0].get_legend_handles_labels()
extra_handles, extra_labels = current_axis.get_legend_handles_labels()
axes[1, 0].legend(handles + extra_handles, labels + extra_labels, frameon=False)

flux_limit = max(float(np.max(np.abs(final_flux_delta))), np.finfo(float).eps)
flux_image = axes[1, 1].imshow(
    final_flux_delta.T,
    origin="lower",
    cmap="RdBu_r",
    vmin=-flux_limit,
    vmax=flux_limit,
)
contour_levels = np.linspace(
    float(np.percentile(final_flux, 10.0)),
    float(np.percentile(final_flux, 90.0)),
    12,
)
axes[1, 1].contour(final_flux.T, levels=contour_levels, colors="white", linewidths=0.5)
axes[1, 1].set_title("Final perturbed flux residual")
axes[1, 1].set_xlabel("grid x")
axes[1, 1].set_ylabel("grid y")
figure.colorbar(flux_image, ax=axes[1, 1], shrink=0.75)

figure.suptitle(
    "MHX publication example: nonlinear periodic double-Harris reconnection",
    fontsize=14,
)
figure.savefig(summary_path, dpi=FIGURE_DPI)
plt.close(figure)

delta_flux = perturbed_flux - base_flux
movie_time = time[-delta_flux.shape[0] :]
movie_frame_indices = np.unique(
    np.linspace(
        0,
        delta_flux.shape[0] - 1,
        min(DELTA_MOVIE_MAX_FRAMES, delta_flux.shape[0]),
        dtype=int,
    )
)
delta_limit = max(float(np.max(np.abs(delta_flux))), np.finfo(float).eps)
frames = []
for frame_index in movie_frame_indices:
    frame_figure, frame_axis = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    image = frame_axis.imshow(
        delta_flux[frame_index].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-delta_limit,
        vmax=delta_limit,
    )
    frame_axis.contour(
        perturbed_flux[frame_index].T,
        levels=contour_levels,
        colors="black",
        linewidths=0.45,
        alpha=0.65,
    )
    frame_axis.set_title(
        "Double-Harris reconnecting perturbation "
        f"(t={movie_time[frame_index]:.2f}, frame {frame_index})"
    )
    frame_axis.set_xlabel("grid x")
    frame_axis.set_ylabel("grid y")
    frame_figure.colorbar(image, ax=frame_axis, shrink=0.75, label=r"$\Delta\psi$")
    frame_figure.canvas.draw()
    frames.append(np.asarray(frame_figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(frame_figure)
imageio.mimsave(delta_movie_path, frames, duration=0.11)

write_manifest(
    manifest_path,
    config=diagnostics,
    outputs={
        "diagnostics": "diagnostics.json",
        "validation": "validation.json",
        "history": "periodic_double_harris_seeded_long_run.npz",
        "periodic_double_harris_seeded_long_run": (
            "figures/periodic_double_harris_seeded_long_run.png"
        ),
        "periodic_double_harris_flux_movie": "figures/periodic_double_harris_flux.gif",
        "periodic_double_harris_current_movie": "figures/periodic_double_harris_current.gif",
        "publication_summary": "figures/publication_double_harris_reconnection_summary.png",
        "publication_delta_flux_movie": "figures/publication_double_harris_delta_flux.gif",
    },
    claim_level="validation",
    claim_scope=(
        "Standalone publication-style periodic double-Harris validation example. "
        "The residual-flux movie emphasizes reconnecting dynamics rather than the "
        "large static equilibrium field."
    ),
)

print(f"manifest: {manifest_path}")
print(f"validation_passed: {validation['passed']}")
print(f"summary_figure: {summary_path}")
print(f"benchmark_figure: {RUN_DIR / 'figures' / 'periodic_double_harris_seeded_long_run.png'}")
print(f"flux_movie: {RUN_DIR / 'figures' / 'periodic_double_harris_flux.gif'}")
print(f"current_movie: {RUN_DIR / 'figures' / 'periodic_double_harris_current.gif'}")
print(f"delta_flux_movie: {delta_movie_path}")
