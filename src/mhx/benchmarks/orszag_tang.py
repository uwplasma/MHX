"""Nonlinear Orszag--Tang reduced-MHD vortex validation and media artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_divergence_linf, magnetic_energy
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4

ORSZAG_TANG_VORTEX_SCHEMA = "mhx.validation.orszag_tang_vortex.v1"
ORSZAG_TANG_DOMAIN = (2.0 * np.pi, 2.0 * np.pi)


@dataclass(frozen=True)
class OrszagTangVortexResult:
    """Saved arrays and validation gates for the reduced-MHD Orszag--Tang vortex."""

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
    vorticity_high_k_fraction: np.ndarray
    max_relative_energy_growth: float
    relative_energy_drop: float
    current_high_k_growth: float
    vorticity_high_k_growth: float
    final_magnetic_divergence_linf: float
    initial_state: ReducedMHDState
    final_state: ReducedMHDState
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def orszag_tang_initial_state(grid: CartesianGrid) -> ReducedMHDState:
    r"""Return the incompressible reduced-MHD Orszag--Tang initial condition.

    The stream and flux functions are

    ``phi = cos(x) + cos(y)``, ``psi = cos(y) + 0.5 cos(2x)``.

    With the MHX convention ``B_perp=(psi_y, -psi_x)`` and
    ``v_perp=(phi_y, -phi_x)``, this gives the classic 2-D Orszag--Tang
    large-scale vortex ``v=(-sin y, sin x)`` and ``B=(-sin y, sin 2x)``.
    """
    x, y = grid.mesh()
    psi = jnp.cos(y) + 0.5 * jnp.cos(2.0 * x)
    omega = -jnp.cos(x) - jnp.cos(y)
    return ReducedMHDState(psi=psi, omega=omega)


def run_orszag_tang_vortex_validation(
    *,
    shape: tuple[int, int] = (24, 24),
    resistivity: float = 1.0e-2,
    viscosity: float = 1.0e-2,
    dt: float = 5.0e-3,
    t_end: float = 2.0,
    save_every: int = 20,
    min_relative_energy_drop: float = 5.0e-3,
    max_relative_energy_growth: float = 1.0e-8,
    min_current_high_k_growth: float = 1.0e-6,
    min_vorticity_high_k_growth: float = 1.0e-6,
    max_magnetic_divergence_linf: float = 1.0e-10,
) -> OrszagTangVortexResult:
    r"""Run a periodic nonlinear Orszag--Tang reduced-MHD validation.

    This is a compact nonlinear benchmark rather than a compressible full-MHD
    shock-capturing Orszag--Tang run. It verifies that nonlinear advection,
    magnetic tension, diffusion, spectral current/vorticity diagnostics, and
    media generation operate coherently on a standard vortex testbed.
    """
    _validate_inputs(
        shape=shape,
        resistivity=resistivity,
        viscosity=viscosity,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        min_relative_energy_drop=min_relative_energy_drop,
        max_relative_energy_growth=max_relative_energy_growth,
        min_current_high_k_growth=min_current_high_k_growth,
        min_vorticity_high_k_growth=min_vorticity_high_k_growth,
        max_magnetic_divergence_linf=max_magnetic_divergence_linf,
    )
    steps = int(round(t_end / dt))
    if steps % save_every != 0:
        raise ValueError("t_end / dt must be divisible by save_every")
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=ORSZAG_TANG_DOMAIN)
    )
    initial_state = orszag_tang_initial_state(grid)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(
        initial_state,
        rhs,
        dt=dt,
        steps=steps,
        save_every=save_every,
    )
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
    total = magnetic + kinetic
    energy_scale = max(abs(float(total[0])), np.finfo(np.float64).tiny)
    relative_growth = np.diff(total) / energy_scale
    max_growth = float(np.max(relative_growth))
    relative_drop = float((total[0] - total[-1]) / energy_scale)
    current_linf = np.max(np.abs(current), axis=(1, 2))
    vorticity_linf = np.max(np.abs(omega), axis=(1, 2))
    current_high_k = np.asarray([_high_k_fraction(frame) for frame in current])
    vorticity_high_k = np.asarray([_high_k_fraction(frame) for frame in omega])
    current_high_k_growth = float(np.max(current_high_k) - current_high_k[0])
    vorticity_high_k_growth = float(np.max(vorticity_high_k) - vorticity_high_k[0])
    final_state = states[-1]
    final_divergence = float(magnetic_divergence_linf(final_state, lengths=grid.lengths))

    checks = {
        "finite_arrays": bool(
            np.isfinite(psi).all()
            and np.isfinite(omega).all()
            and np.isfinite(current).all()
            and np.isfinite(total).all()
        ),
        "energy_nonincreasing": max_growth <= max_relative_energy_growth,
        "net_dissipation_observed": relative_drop >= min_relative_energy_drop,
        "current_high_k_cascade_observed": current_high_k_growth >= min_current_high_k_growth,
        "vorticity_high_k_cascade_observed": (
            vorticity_high_k_growth >= min_vorticity_high_k_growth
        ),
        "magnetic_divergence_preserved": final_divergence <= max_magnetic_divergence_linf,
    }
    diagnostics = {
        "schema": ORSZAG_TANG_VORTEX_SCHEMA,
        "shape": list(shape),
        "domain": [0.0, ORSZAG_TANG_DOMAIN[0], 0.0, ORSZAG_TANG_DOMAIN[1]],
        "resistivity": resistivity,
        "viscosity": viscosity,
        "dt": dt,
        "t_end": t_end,
        "steps": steps,
        "save_every": save_every,
        "samples": int(time.size),
        "initial_total_energy": float(total[0]),
        "final_total_energy": float(total[-1]),
        "relative_energy_drop": relative_drop,
        "max_relative_energy_growth": max_growth,
        "initial_current_linf": float(current_linf[0]),
        "max_current_linf": float(np.max(current_linf)),
        "initial_vorticity_linf": float(vorticity_linf[0]),
        "max_vorticity_linf": float(np.max(vorticity_linf)),
        "initial_current_high_k_fraction": float(current_high_k[0]),
        "max_current_high_k_fraction": float(np.max(current_high_k)),
        "current_high_k_growth": current_high_k_growth,
        "initial_vorticity_high_k_fraction": float(vorticity_high_k[0]),
        "max_vorticity_high_k_fraction": float(np.max(vorticity_high_k)),
        "vorticity_high_k_growth": vorticity_high_k_growth,
        "final_magnetic_divergence_linf": final_divergence,
        "references": {
            "classic_test": "Orszag--Tang vortex adapted to incompressible reduced MHD.",
            "initial_condition": (
                "phi=cos(x)+cos(y), psi=cos(y)+0.5*cos(2x); "
                "v=(-sin y, sin x), B=(-sin y, sin 2x)."
            ),
            "claim_scope": (
                "Nonlinear reduced-MHD vortex/cascade media and invariant checks; "
                "not a compressible shock-capturing full-MHD benchmark."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.orszag_tang_vortex.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_relative_energy_drop": min_relative_energy_drop,
            "max_relative_energy_growth": max_relative_energy_growth,
            "min_current_high_k_growth": min_current_high_k_growth,
            "min_vorticity_high_k_growth": min_vorticity_high_k_growth,
            "max_magnetic_divergence_linf": max_magnetic_divergence_linf,
        },
        "diagnostics": diagnostics,
    }
    return OrszagTangVortexResult(
        time=time,
        psi=psi,
        omega=omega,
        current_density=current,
        magnetic_energy=magnetic,
        kinetic_energy=kinetic,
        total_energy=total,
        current_linf=current_linf,
        vorticity_linf=vorticity_linf,
        current_high_k_fraction=current_high_k,
        vorticity_high_k_fraction=vorticity_high_k,
        max_relative_energy_growth=max_growth,
        relative_energy_drop=relative_drop,
        current_high_k_growth=current_high_k_growth,
        vorticity_high_k_growth=vorticity_high_k_growth,
        final_magnetic_divergence_linf=final_divergence,
        initial_state=initial_state,
        final_state=final_state,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_orszag_tang_vortex_validation(
    outdir: str | Path,
    *,
    movies: bool = False,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write Orszag--Tang JSON, NPZ, figures, optional GIFs, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_orszag_tang_vortex_validation(**kwargs)
    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "orszag_tang_vortex.npz"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        history_path,
        schema=ORSZAG_TANG_VORTEX_SCHEMA,
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
        vorticity_high_k_fraction=result.vorticity_high_k_fraction,
        max_relative_energy_growth=result.max_relative_energy_growth,
        relative_energy_drop=result.relative_energy_drop,
        current_high_k_growth=result.current_high_k_growth,
        vorticity_high_k_growth=result.vorticity_high_k_growth,
        final_magnetic_divergence_linf=result.final_magnetic_divergence_linf,
    )
    summary_path = _write_summary_figure(result, figure_dir / "orszag_tang_summary.png")
    outputs: dict[str, str] = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "history": history_path.name,
        "summary": str(summary_path.relative_to(output_dir)),
    }
    if movies:
        outputs.update(
            _write_orszag_tang_movies(
                result,
                figure_dir=figure_dir,
                relative_to=output_dir,
            )
        )
    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope=(
            "Incompressible reduced-MHD Orszag--Tang vortex nonlinear cascade and "
            "energy/divergence validation."
        ),
    )
    return manifest_path, result.validation


def _write_summary_figure(result: OrszagTangVortexResult, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    axes[0, 0].plot(result.time, result.magnetic_energy, label=r"$E_B$")
    axes[0, 0].plot(result.time, result.kinetic_energy, label=r"$E_K$")
    axes[0, 0].plot(result.time, result.total_energy, label=r"$E$")
    axes[0, 0].set_title("Energy decay")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("mean energy")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(result.time, result.current_high_k_fraction, label="current")
    axes[0, 1].plot(result.time, result.vorticity_high_k_fraction, label="vorticity")
    axes[0, 1].set_title("High-wavenumber fraction")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("spectral power fraction")
    axes[0, 1].legend(frameon=False)
    vmax_j = max(float(np.max(np.abs(result.current_density[-1]))), np.finfo(float).eps)
    image_j = axes[1, 0].imshow(
        result.current_density[-1].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax_j,
        vmax=vmax_j,
        extent=(0.0, ORSZAG_TANG_DOMAIN[0], 0.0, ORSZAG_TANG_DOMAIN[1]),
    )
    axes[1, 0].set_title("Final current density")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(image_j, ax=axes[1, 0], shrink=0.75)
    vmax_w = max(float(np.max(np.abs(result.omega[-1]))), np.finfo(float).eps)
    image_w = axes[1, 1].imshow(
        result.omega[-1].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax_w,
        vmax=vmax_w,
        extent=(0.0, ORSZAG_TANG_DOMAIN[0], 0.0, ORSZAG_TANG_DOMAIN[1]),
    )
    axes[1, 1].set_title("Final vorticity")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    fig.colorbar(image_w, ax=axes[1, 1], shrink=0.75)
    fig.suptitle("Reduced-MHD Orszag--Tang vortex validation", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_orszag_tang_movies(
    result: OrszagTangVortexResult,
    *,
    figure_dir: Path,
    relative_to: Path,
) -> dict[str, str]:
    outputs = {
        "flux_movie": _write_scalar_movie(
            result.psi,
            figure_dir / "orszag_tang_flux.gif",
            cmap="viridis",
            symmetric=False,
        ),
        "current_movie": _write_scalar_movie(
            result.current_density,
            figure_dir / "orszag_tang_current.gif",
            cmap="RdBu_r",
            symmetric=True,
        ),
        "vorticity_movie": _write_scalar_movie(
            result.omega,
            figure_dir / "orszag_tang_vorticity.gif",
            cmap="RdBu_r",
            symmetric=True,
        ),
    }
    return {name: str(path.relative_to(relative_to)) for name, path in outputs.items()}


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

    path.parent.mkdir(parents=True, exist_ok=True)
    indices = _sample_indices(fields.shape[0], max_frames)
    values = np.asarray(fields)[indices]
    if symmetric:
        vmax = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        vmin = -vmax
    else:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    colormap = colormaps[cmap]
    frames = []
    for field in values:
        normalized = np.clip((field.T - vmin) / (vmax - vmin), 0.0, 1.0)
        frames.append((255.0 * colormap(normalized)[..., :3]).astype(np.uint8))
    imageio.mimsave(path, frames, duration=90, loop=0, palettesize=48)
    return path


def _sample_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count <= max_frames:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _high_k_fraction(field: np.ndarray, *, cutoff: float = 3.0) -> float:
    values = np.asarray(field, dtype=np.float64)
    kx = np.fft.fftfreq(values.shape[0]) * values.shape[0]
    ky = np.fft.fftfreq(values.shape[1]) * values.shape[1]
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    power = np.abs(np.fft.fftn(values)) ** 2
    total_power = float(np.sum(power))
    if total_power <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.sum(power[radius >= cutoff]) / total_power)


def _validate_inputs(
    *,
    shape: tuple[int, int],
    resistivity: float,
    viscosity: float,
    dt: float,
    t_end: float,
    save_every: int,
    min_relative_energy_drop: float,
    max_relative_energy_growth: float,
    min_current_high_k_growth: float,
    min_vorticity_high_k_growth: float,
    max_magnetic_divergence_linf: float,
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
    if min_relative_energy_drop < 0.0:
        raise ValueError("min_relative_energy_drop must be non-negative")
    if max_relative_energy_growth < 0.0:
        raise ValueError("max_relative_energy_growth must be non-negative")
    if min_current_high_k_growth < 0.0:
        raise ValueError("min_current_high_k_growth must be non-negative")
    if min_vorticity_high_k_growth < 0.0:
        raise ValueError("min_vorticity_high_k_growth must be non-negative")
    if max_magnetic_divergence_linf <= 0.0:
        raise ValueError("max_magnetic_divergence_linf must be positive")
