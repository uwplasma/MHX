"""Nonlinear reduced-MHD turbulence and turbulence-mediated reconnection examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.diagnostics import (
    detect_flux_critical_points,
    kinetic_energy,
    magnetic_divergence_linf,
    magnetic_energy,
)
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics.spectral import laplacian
from mhx.physics import PeriodicDoubleHarrisEquilibrium
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4

TURBULENCE_DOMAIN = (2.0 * np.pi, 2.0 * np.pi)
DECAYING_MHD_TURBULENCE_SCHEMA = "mhx.validation.decaying_mhd_turbulence.v1"
FORCED_TURBULENT_RECONNECTION_SCHEMA = "mhx.validation.forced_turbulent_reconnection.v1"


@dataclass(frozen=True)
class TurbulenceResult:
    """Saved arrays and gates for nonlinear reduced-MHD turbulence examples."""

    schema: str
    time: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    current_density: np.ndarray
    magnetic_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    current_linf: np.ndarray
    vorticity_linf: np.ndarray
    current_high_k_fraction: np.ndarray
    magnetic_divergence_linf: np.ndarray
    reconnection_proxy: np.ndarray | None
    reconnection_rate_proxy: np.ndarray | None
    initial_state: ReducedMHDState
    final_state: ReducedMHDState
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def turbulent_initial_state(
    grid: CartesianGrid,
    *,
    seed: int = 0,
    flux_amplitude: float = 0.25,
    flow_amplitude: float = 0.25,
    kmin: int = 1,
    kmax: int = 5,
    spectral_slope: float = 5.0 / 3.0,
) -> ReducedMHDState:
    """Return a deterministic broadband reduced-MHD turbulent initial state."""
    flux = flux_amplitude * _broadband_scalar_field(
        grid,
        seed=seed,
        kmin=kmin,
        kmax=kmax,
        spectral_slope=spectral_slope,
    )
    stream = flow_amplitude * _broadband_scalar_field(
        grid,
        seed=seed + 1009,
        kmin=kmin,
        kmax=kmax,
        spectral_slope=spectral_slope,
    )
    omega = laplacian(jnp.asarray(stream), lengths=grid.lengths)
    return ReducedMHDState(psi=jnp.asarray(flux), omega=omega)


def run_decaying_mhd_turbulence_validation(
    *,
    shape: tuple[int, int] = (32, 32),
    resistivity: float = 2.0e-2,
    viscosity: float = 2.0e-2,
    dt: float = 1.0e-2,
    t_end: float = 4.0,
    save_every: int = 20,
    seed: int = 7,
    min_relative_energy_drop: float = 1.0e-3,
    max_relative_energy_growth: float = 1.0e-8,
    min_current_linf_growth: float = 1.0e-6,
    max_magnetic_divergence_linf: float = 1.0e-10,
) -> TurbulenceResult:
    """Run a deterministic decaying 2-D reduced-MHD turbulence validation."""
    _validate_turbulence_inputs(
        shape=shape,
        resistivity=resistivity,
        viscosity=viscosity,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
    )
    if min_relative_energy_drop < 0.0:
        raise ValueError("min_relative_energy_drop must be non-negative")
    if max_relative_energy_growth < 0.0:
        raise ValueError("max_relative_energy_growth must be non-negative")
    if min_current_linf_growth < 0.0:
        raise ValueError("min_current_linf_growth must be non-negative")
    if max_magnetic_divergence_linf <= 0.0:
        raise ValueError("max_magnetic_divergence_linf must be positive")

    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=TURBULENCE_DOMAIN)
    )
    initial_state = turbulent_initial_state(grid, seed=seed)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)
    result = _run_turbulence_trajectory(
        initial_state,
        params,
        grid=grid,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
    )
    relative_energy_drop, max_energy_growth = _energy_drop_and_growth(result.total_energy)
    current_linf_growth = float(np.max(result.current_linf) - result.current_linf[0])
    checks = {
        "finite_histories": _finite_result(result),
        "energy_nonincreasing": max_energy_growth <= max_relative_energy_growth,
        "net_dissipation_observed": relative_energy_drop >= min_relative_energy_drop,
        "current_sheet_activity_observed": current_linf_growth >= min_current_linf_growth,
        "magnetic_divergence_preserved": (
            float(np.max(result.magnetic_divergence_linf)) <= max_magnetic_divergence_linf
        ),
    }
    diagnostics = {
        "schema": DECAYING_MHD_TURBULENCE_SCHEMA,
        "shape": list(shape),
        "domain": [0.0, TURBULENCE_DOMAIN[0], 0.0, TURBULENCE_DOMAIN[1]],
        "resistivity": resistivity,
        "viscosity": viscosity,
        "dt": dt,
        "t_end": t_end,
        "save_every": save_every,
        "seed": seed,
        "samples": int(result.time.size),
        "initial_total_energy": float(result.total_energy[0]),
        "final_total_energy": float(result.total_energy[-1]),
        "relative_energy_drop": relative_energy_drop,
        "max_relative_energy_growth": max_energy_growth,
        "initial_current_linf": float(result.current_linf[0]),
        "max_current_linf": float(np.max(result.current_linf)),
        "current_linf_growth": current_linf_growth,
        "max_current_high_k_fraction": float(np.max(result.current_high_k_fraction)),
        "max_magnetic_divergence_linf": float(np.max(result.magnetic_divergence_linf)),
        "references": {
            "scope": (
                "Deterministic decaying incompressible reduced-MHD turbulence "
                "example with current-sheet formation and dissipative energy checks."
            ),
            "literature": (
                "Anchored to 2-D MHD turbulence/current-sheet studies and the "
                "Orszag--Tang vortex as a canonical nonlinear MHD test."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.decaying_mhd_turbulence.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_relative_energy_drop": min_relative_energy_drop,
            "max_relative_energy_growth": max_relative_energy_growth,
            "min_current_linf_growth": min_current_linf_growth,
            "max_magnetic_divergence_linf": max_magnetic_divergence_linf,
        },
        "diagnostics": diagnostics,
    }
    return _replace_result_metadata(result, DECAYING_MHD_TURBULENCE_SCHEMA, diagnostics, validation)


def run_forced_turbulent_reconnection_validation(
    *,
    shape: tuple[int, int] = (32, 32),
    width: float = 0.32,
    resistivity: float = 1.5e-3,
    viscosity: float = 1.5e-3,
    perturbation_amplitude: float = 1.0e-2,
    turbulent_flux_amplitude: float = 1.5e-2,
    turbulent_flow_amplitude: float = 1.5e-2,
    forcing_amplitude: float = 2.0e-3,
    dt: float = 2.0e-2,
    t_end: float = 20.0,
    save_every: int = 50,
    seed: int = 11,
    min_reconnection_proxy_change: float = 1.0e-3,
    min_current_linf_growth: float = 1.0e-6,
    max_relative_energy_growth: float = 2.0,
    max_magnetic_divergence_linf: float = 1.0e-10,
) -> TurbulenceResult:
    """Run a 2-D reduced-MHD current sheet with deterministic large-scale forcing."""
    _validate_turbulence_inputs(
        shape=shape,
        resistivity=resistivity,
        viscosity=viscosity,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
    )
    if width <= 0.0:
        raise ValueError("width must be positive")
    if perturbation_amplitude <= 0.0:
        raise ValueError("perturbation_amplitude must be positive")
    if turbulent_flux_amplitude < 0.0 or turbulent_flow_amplitude < 0.0:
        raise ValueError("turbulent amplitudes must be non-negative")
    if forcing_amplitude < 0.0:
        raise ValueError("forcing_amplitude must be non-negative")

    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=TURBULENCE_DOMAIN)
    )
    sheet_state = PeriodicDoubleHarrisEquilibrium(
        width=width,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_mode=(0, 1),
    ).initial_state(grid)
    turbulent_state = turbulent_initial_state(
        grid,
        seed=seed,
        flux_amplitude=turbulent_flux_amplitude,
        flow_amplitude=turbulent_flow_amplitude,
        kmin=1,
        kmax=4,
    )
    initial_state = ReducedMHDState(
        psi=sheet_state.psi + turbulent_state.psi,
        omega=sheet_state.omega + turbulent_state.omega,
    )
    forcing_stream = _broadband_scalar_field(grid, seed=seed + 2027, kmin=1, kmax=3)
    forcing_omega = forcing_amplitude * laplacian(jnp.asarray(forcing_stream), lengths=grid.lengths)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    def forcing(state: ReducedMHDState) -> ReducedMHDState:
        base = reduced_mhd_rhs(state, params, lengths=grid.lengths)
        return ReducedMHDState(psi=base.psi, omega=base.omega + forcing_omega)

    result = _run_turbulence_trajectory(
        initial_state,
        params,
        grid=grid,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        rhs=forcing,
        reconnection_proxy=True,
    )
    _, max_energy_growth = _energy_drop_and_growth(result.total_energy)
    current_linf_growth = float(np.max(result.current_linf) - result.current_linf[0])
    proxy_change = float(
        np.max(result.reconnection_proxy) - np.min(result.reconnection_proxy)
        if result.reconnection_proxy is not None
        else 0.0
    )
    checks = {
        "finite_histories": _finite_result(result),
        "reconnection_proxy_changes": proxy_change >= min_reconnection_proxy_change,
        "current_sheet_activity_observed": current_linf_growth >= min_current_linf_growth,
        "forced_energy_growth_bounded": max_energy_growth <= max_relative_energy_growth,
        "magnetic_divergence_preserved": (
            float(np.max(result.magnetic_divergence_linf)) <= max_magnetic_divergence_linf
        ),
    }
    diagnostics = {
        "schema": FORCED_TURBULENT_RECONNECTION_SCHEMA,
        "shape": list(shape),
        "domain": [0.0, TURBULENCE_DOMAIN[0], 0.0, TURBULENCE_DOMAIN[1]],
        "equilibrium": "periodic_double_harris_plus_broadband_flow",
        "width": width,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "perturbation_amplitude": perturbation_amplitude,
        "turbulent_flux_amplitude": turbulent_flux_amplitude,
        "turbulent_flow_amplitude": turbulent_flow_amplitude,
        "forcing_amplitude": forcing_amplitude,
        "dt": dt,
        "t_end": t_end,
        "save_every": save_every,
        "seed": seed,
        "samples": int(result.time.size),
        "initial_total_energy": float(result.total_energy[0]),
        "final_total_energy": float(result.total_energy[-1]),
        "max_relative_energy_growth": max_energy_growth,
        "initial_current_linf": float(result.current_linf[0]),
        "max_current_linf": float(np.max(result.current_linf)),
        "current_linf_growth": current_linf_growth,
        "reconnection_proxy_change": proxy_change,
        "max_abs_reconnection_rate_proxy": float(np.max(np.abs(result.reconnection_rate_proxy))),
        "max_magnetic_divergence_linf": float(np.max(result.magnetic_divergence_linf)),
        "references": {
            "scope": (
                "Pedagogical 2-D reduced-MHD forced-turbulence current-sheet "
                "example; not a 3-D LV99 production test."
            ),
            "literature": (
                "Anchored to turbulent reconnection ideas from Lazarian--Vishniac "
                "and 2-D MHD turbulence reconnection diagnostics from Servidio et al."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.forced_turbulent_reconnection.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_reconnection_proxy_change": min_reconnection_proxy_change,
            "min_current_linf_growth": min_current_linf_growth,
            "max_relative_energy_growth": max_relative_energy_growth,
            "max_magnetic_divergence_linf": max_magnetic_divergence_linf,
        },
        "diagnostics": diagnostics,
    }
    return _replace_result_metadata(
        result,
        FORCED_TURBULENT_RECONNECTION_SCHEMA,
        diagnostics,
        validation,
    )


def write_decaying_mhd_turbulence_validation(
    outdir: str | Path,
    *,
    movies: bool = False,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write decaying turbulence artifacts and optional movies."""
    result = run_decaying_mhd_turbulence_validation(**kwargs)
    return _write_turbulence_artifacts(
        result,
        Path(outdir),
        history_name="decaying_mhd_turbulence.npz",
        summary_name="decaying_mhd_turbulence_summary.png",
        movie_prefix="decaying_mhd_turbulence",
        claim_scope=(
            "Deterministic nonlinear reduced-MHD turbulence/current-sheet media "
            "with dissipative energy and divergence gates."
        ),
        movies=movies,
    )


def write_forced_turbulent_reconnection_validation(
    outdir: str | Path,
    *,
    movies: bool = False,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write forced turbulent reconnection artifacts and optional movies."""
    result = run_forced_turbulent_reconnection_validation(**kwargs)
    return _write_turbulence_artifacts(
        result,
        Path(outdir),
        history_name="forced_turbulent_reconnection.npz",
        summary_name="forced_turbulent_reconnection_summary.png",
        movie_prefix="forced_turbulent_reconnection",
        claim_scope=(
            "Pedagogical forced-turbulence current-sheet replay with reconnection "
            "proxy and finite-energy gates; not a converged 3-D turbulent "
            "reconnection result."
        ),
        movies=movies,
    )


def _run_turbulence_trajectory(
    initial_state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    grid: CartesianGrid,
    dt: float,
    t_end: float,
    save_every: int,
    rhs: Any | None = None,
    reconnection_proxy: bool = False,
) -> TurbulenceResult:
    steps = int(round(t_end / dt))
    active_rhs = rhs
    if active_rhs is None:

        def active_rhs(state: ReducedMHDState) -> ReducedMHDState:
            return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(initial_state, active_rhs, dt=dt, steps=steps, save_every=save_every)
    time = np.concatenate(([0.0], np.asarray(trajectory.times, dtype=np.float64)))
    psi = np.concatenate(
        (
            np.asarray(initial_state.psi, dtype=np.float64)[None, ...],
            np.asarray(trajectory.states.psi, dtype=np.float64),
        ),
        axis=0,
    )
    omega = np.concatenate(
        (
            np.asarray(initial_state.omega, dtype=np.float64)[None, ...],
            np.asarray(trajectory.states.omega, dtype=np.float64),
        ),
        axis=0,
    )
    current = np.asarray(
        [np.asarray(current_density(frame, lengths=grid.lengths)) for frame in psi],
        dtype=np.float64,
    )
    states = tuple(
        ReducedMHDState(psi=jnp.asarray(psi[index]), omega=jnp.asarray(omega[index]))
        for index in range(time.size)
    )
    magnetic = np.asarray(
        [float(magnetic_energy(state, lengths=grid.lengths)) for state in states],
        dtype=np.float64,
    )
    kinetic = np.asarray(
        [float(kinetic_energy(state, lengths=grid.lengths)) for state in states],
        dtype=np.float64,
    )
    divergence = np.asarray(
        [float(magnetic_divergence_linf(state, lengths=grid.lengths)) for state in states],
        dtype=np.float64,
    )
    reconnecting_flux = None
    reconnecting_rate = None
    if reconnection_proxy:
        reconnecting_flux = np.asarray(
            [_critical_flux_separation(frame, lengths=grid.lengths) for frame in psi],
            dtype=np.float64,
        )
        reconnecting_rate = np.gradient(reconnecting_flux, time)
    return TurbulenceResult(
        schema="",
        time=time,
        psi=psi,
        omega=omega,
        current_density=current,
        magnetic_energy=magnetic,
        kinetic_energy=kinetic,
        total_energy=magnetic + kinetic,
        current_linf=np.max(np.abs(current), axis=(1, 2)),
        vorticity_linf=np.max(np.abs(omega), axis=(1, 2)),
        current_high_k_fraction=np.asarray([_high_k_fraction(frame) for frame in current]),
        magnetic_divergence_linf=divergence,
        reconnection_proxy=reconnecting_flux,
        reconnection_rate_proxy=reconnecting_rate,
        initial_state=initial_state,
        final_state=states[-1],
        diagnostics={},
        validation={},
    )


def _write_turbulence_artifacts(
    result: TurbulenceResult,
    output_dir: Path,
    *,
    history_name: str,
    summary_name: str,
    movie_prefix: str,
    claim_scope: str,
    movies: bool,
) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / history_name
    diagnostics_path.write_text(json.dumps(result.diagnostics, indent=2, sort_keys=True))
    validation_path.write_text(json.dumps(result.validation, indent=2, sort_keys=True))
    payload: dict[str, Any] = {
        "schema": result.schema,
        "time": result.time,
        "psi": result.psi,
        "omega": result.omega,
        "current_density": result.current_density,
        "magnetic_energy": result.magnetic_energy,
        "kinetic_energy": result.kinetic_energy,
        "total_energy": result.total_energy,
        "current_linf": result.current_linf,
        "vorticity_linf": result.vorticity_linf,
        "current_high_k_fraction": result.current_high_k_fraction,
        "magnetic_divergence_linf": result.magnetic_divergence_linf,
    }
    if result.reconnection_proxy is not None and result.reconnection_rate_proxy is not None:
        payload["reconnection_proxy"] = result.reconnection_proxy
        payload["reconnection_rate_proxy"] = result.reconnection_rate_proxy
    np.savez_compressed(history_path, **payload)
    summary_path = _write_turbulence_summary(result, figure_dir / summary_name)
    outputs: dict[str, str] = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "history": history_path.name,
        "summary": str(summary_path.relative_to(output_dir)),
    }
    if movies:
        movie_paths = _write_turbulence_movies(
            result,
            figure_dir=figure_dir,
            movie_prefix=movie_prefix,
        )
        outputs.update(
            {name: str(path.relative_to(output_dir)) for name, path in movie_paths.items()}
        )
    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope=claim_scope,
    )
    return manifest_path, result.validation


def _write_turbulence_summary(result: TurbulenceResult, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    axes[0, 0].plot(result.time, result.magnetic_energy, label=r"$E_B$")
    axes[0, 0].plot(result.time, result.kinetic_energy, label=r"$E_K$")
    axes[0, 0].plot(result.time, result.total_energy, label=r"$E$")
    axes[0, 0].set_title("Energy")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(result.time, result.current_linf, label=r"$||j_z||_\infty$")
    if result.reconnection_rate_proxy is not None:
        axes[0, 1].plot(
            result.time,
            np.abs(result.reconnection_rate_proxy),
            label=r"$|\dot{\psi}_{rec}|$ proxy",
        )
    axes[0, 1].set_title("Current/reconnection proxy")
    axes[0, 1].legend(frameon=False)
    vmax_j = max(float(np.max(np.abs(result.current_density[-1]))), np.finfo(float).eps)
    image_j = axes[1, 0].imshow(
        result.current_density[-1].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax_j,
        vmax=vmax_j,
        extent=(0.0, TURBULENCE_DOMAIN[0], 0.0, TURBULENCE_DOMAIN[1]),
    )
    axes[1, 0].set_title("Final current density")
    fig.colorbar(image_j, ax=axes[1, 0], shrink=0.75)
    image_psi = axes[1, 1].imshow(
        result.psi[-1].T,
        origin="lower",
        cmap="viridis",
        extent=(0.0, TURBULENCE_DOMAIN[0], 0.0, TURBULENCE_DOMAIN[1]),
    )
    axes[1, 1].set_title("Final magnetic flux")
    fig.colorbar(image_psi, ax=axes[1, 1], shrink=0.75)
    fig.suptitle(result.schema)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_turbulence_movies(
    result: TurbulenceResult,
    *,
    figure_dir: Path,
    movie_prefix: str,
) -> dict[str, Path]:
    return {
        "current_movie": _write_scalar_movie(
            result.current_density,
            figure_dir / f"{movie_prefix}_current.gif",
            cmap="RdBu_r",
            symmetric=True,
        ),
        "flux_movie": _write_flux_contour_movie(
            result,
            figure_dir / f"{movie_prefix}_flux_contours.gif",
        ),
    }


def _write_scalar_movie(
    fields: np.ndarray,
    path: Path,
    *,
    cmap: str,
    symmetric: bool,
    max_frames: int = 36,
) -> Path:
    import imageio.v2 as imageio
    from matplotlib import colormaps

    indices = _sample_indices(fields.shape[0], max_frames)
    values = np.asarray(fields)[indices]
    if symmetric:
        vmax = max(float(np.percentile(np.abs(values), 99.5)), np.finfo(float).eps)
        vmin = -vmax
    else:
        vmin = float(np.percentile(values, 0.5))
        vmax = float(np.percentile(values, 99.5))
    colormap = colormaps[cmap]
    frames = []
    for field in values:
        normalized = np.clip((field.T - vmin) / (vmax - vmin), 0.0, 1.0)
        frames.append((255.0 * colormap(normalized)[..., :3]).astype(np.uint8))
    imageio.mimsave(path, frames, duration=90, loop=0, palettesize=48)
    return path


def _write_flux_contour_movie(
    result: TurbulenceResult,
    path: Path,
    *,
    max_frames: int = 36,
) -> Path:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, TURBULENCE_DOMAIN[0], result.psi.shape[1], endpoint=False)
    y = np.linspace(0.0, TURBULENCE_DOMAIN[1], result.psi.shape[2], endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    indices = _sample_indices(result.time.size, max_frames)
    current_values = result.current_density[indices]
    vmax = max(float(np.percentile(np.abs(current_values), 99.5)), np.finfo(float).eps)
    frames = []
    for index in indices:
        fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=72, constrained_layout=True)
        ax.imshow(
            result.current_density[index].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=(0.0, TURBULENCE_DOMAIN[0], 0.0, TURBULENCE_DOMAIN[1]),
        )
        levels = np.linspace(
            float(np.percentile(result.psi[index], 5.0)),
            float(np.percentile(result.psi[index], 95.0)),
            18,
        )
        ax.contour(
            x_mesh,
            y_mesh,
            result.psi[index],
            levels=levels,
            colors="black",
            linewidths=0.35,
        )
        ax.set_title(f"{result.schema.rsplit('.', maxsplit=2)[-2]}, t={result.time[index]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=90, loop=0, palettesize=48)
    return path


def _replace_result_metadata(
    result: TurbulenceResult,
    schema: str,
    diagnostics: dict[str, Any],
    validation: dict[str, Any],
) -> TurbulenceResult:
    return TurbulenceResult(
        schema=schema,
        time=result.time,
        psi=result.psi,
        omega=result.omega,
        current_density=result.current_density,
        magnetic_energy=result.magnetic_energy,
        kinetic_energy=result.kinetic_energy,
        total_energy=result.total_energy,
        current_linf=result.current_linf,
        vorticity_linf=result.vorticity_linf,
        current_high_k_fraction=result.current_high_k_fraction,
        magnetic_divergence_linf=result.magnetic_divergence_linf,
        reconnection_proxy=result.reconnection_proxy,
        reconnection_rate_proxy=result.reconnection_rate_proxy,
        initial_state=result.initial_state,
        final_state=result.final_state,
        diagnostics=diagnostics,
        validation=validation,
    )


def _broadband_scalar_field(
    grid: CartesianGrid,
    *,
    seed: int,
    kmin: int = 1,
    kmax: int = 5,
    spectral_slope: float = 5.0 / 3.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = (np.asarray(item, dtype=np.float64) for item in grid.mesh())
    field = np.zeros(grid.shape, dtype=np.float64)
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if kx == 0 and ky == 0:
                continue
            radius = float(np.hypot(kx, ky))
            if radius < kmin or radius > kmax:
                continue
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = radius ** (-0.5 * spectral_slope)
            field += amplitude * np.cos(kx * x + ky * y + phase)
    field -= float(np.mean(field))
    field /= max(float(np.std(field)), np.finfo(np.float64).eps)
    return field


def _critical_flux_separation(
    psi: np.ndarray,
    *,
    lengths: tuple[float, float],
) -> float:
    points = detect_flux_critical_points(
        psi,
        lengths=lengths,
        max_points=12,
        min_separation=0.35,
    )
    x_values = [point.psi for point in points if point.kind == "X"]
    o_values = [point.psi for point in points if point.kind == "O"]
    if x_values and o_values:
        return float(abs(np.mean(o_values) - np.mean(x_values)))
    return float(np.max(psi) - np.min(psi))


def _energy_drop_and_growth(total_energy: np.ndarray) -> tuple[float, float]:
    scale = max(abs(float(total_energy[0])), np.finfo(np.float64).tiny)
    relative_drop = float((total_energy[0] - total_energy[-1]) / scale)
    max_growth = float(max(0.0, np.max(total_energy) - total_energy[0]) / scale)
    return relative_drop, max_growth


def _finite_result(result: TurbulenceResult) -> bool:
    arrays = [
        result.time,
        result.psi,
        result.omega,
        result.current_density,
        result.total_energy,
        result.current_linf,
        result.magnetic_divergence_linf,
    ]
    if result.reconnection_proxy is not None:
        arrays.append(result.reconnection_proxy)
    if result.reconnection_rate_proxy is not None:
        arrays.append(result.reconnection_rate_proxy)
    return bool(all(np.isfinite(array).all() for array in arrays))


def _sample_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count <= max_frames:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _high_k_fraction(field: np.ndarray, *, cutoff: float = 4.0) -> float:
    values = np.asarray(field, dtype=np.float64)
    kx = np.fft.fftfreq(values.shape[0]) * values.shape[0]
    ky = np.fft.fftfreq(values.shape[1]) * values.shape[1]
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    power = np.abs(np.fft.fftn(values)) ** 2
    total_power = float(np.sum(power))
    if total_power <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.sum(power[radius >= cutoff]) / total_power)


def _figure_to_frame(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _validate_turbulence_inputs(
    *,
    shape: tuple[int, int],
    resistivity: float,
    viscosity: float,
    dt: float,
    t_end: float,
    save_every: int,
) -> None:
    if len(shape) != 2 or shape[0] < 8 or shape[1] < 8:
        raise ValueError("shape must contain at least 8 points in each periodic direction")
    if resistivity <= 0.0:
        raise ValueError("resistivity must be positive")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    steps = int(round(t_end / dt))
    if steps < 4:
        raise ValueError("t_end / dt must produce at least four RK4 steps")
    if not np.isclose(steps * dt, t_end):
        raise ValueError("t_end must be an integer multiple of dt")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if steps // save_every < 2:
        raise ValueError("configuration must save at least two non-initial samples")
