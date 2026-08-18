"""Inviscid Orszag--Tang reduced-MHD vortex energy conservation validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.benchmarks.orszag_tang import orszag_tang_initial_state
from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_divergence_linf, magnetic_energy
from mhx.equations.arakawa_reduced_mhd import arakawa_reduced_mhd_rhs
from mhx.equations.reduced_mhd import current_density
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4

INVISCID_ORSZAG_TANG_SCHEMA = "mhx.validation.inviscid_orszag_tang.v1"
ORSZAG_TANG_DOMAIN = (2.0 * np.pi, 2.0 * np.pi)


@dataclass(frozen=True)
class InviscidOrszagTangResult:
    """Saved arrays and validation gates for the inviscid Orszag--Tang vortex."""

    time: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    current_density: np.ndarray
    magnetic_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    relative_energy_error: float
    final_magnetic_divergence_linf: float
    initial_state: ReducedMHDState
    final_state: ReducedMHDState
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_inviscid_orszag_tang_validation(
    *,
    shape: tuple[int, int] = (128, 128),
    dt: float = 5.0e-4,
    t_end: float = 2.0,
    save_every: int = 40,
    max_relative_energy_error: float = 1.0e-6,
    max_magnetic_divergence_linf: float = 1.0e-10,
) -> InviscidOrszagTangResult:
    """Run a periodic inviscid Orszag--Tang reduced-MHD energy conservation validation."""
    _validate_inputs(
        shape=shape,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        max_relative_energy_error=max_relative_energy_error,
        max_magnetic_divergence_linf=max_magnetic_divergence_linf,
    )
    steps = int(round(t_end / dt))
    if steps % save_every != 0:
        raise ValueError("t_end / dt must be divisible by save_every")

    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=ORSZAG_TANG_DOMAIN)
    )
    initial_state = orszag_tang_initial_state(grid)
    params = ReducedMHDParams(resistivity=0.0, viscosity=0.0)

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return arakawa_reduced_mhd_rhs(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(initial_state, rhs, dt=dt, steps=steps, save_every=save_every)

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
        ReducedMHDState(psi=jnp.asarray(psi[i]), omega=jnp.asarray(omega[i]))
        for i in range(time.size)
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

    relative_energy_error = float(abs((total[-1] - total[0]) / total[0]))

    final_state = states[-1]
    final_divergence = float(magnetic_divergence_linf(final_state, lengths=grid.lengths))

    checks = {
        "finite_arrays": bool(
            np.isfinite(psi).all()
            and np.isfinite(omega).all()
            and np.isfinite(total).all()
        ),
        "energy_conserved": relative_energy_error < max_relative_energy_error,
        "magnetic_divergence_preserved": final_divergence <= max_magnetic_divergence_linf,
    }

    diagnostics = {
        "schema": INVISCID_ORSZAG_TANG_SCHEMA,
        "shape": list(shape),
        "domain": [0.0, ORSZAG_TANG_DOMAIN[0], 0.0, ORSZAG_TANG_DOMAIN[1]],
        "resistivity": 0.0,
        "viscosity": 0.0,
        "dt": dt,
        "t_end": t_end,
        "steps": steps,
        "save_every": save_every,
        "samples": int(time.size),
        "initial_total_energy": float(total[0]),
        "final_total_energy": float(total[-1]),
        "relative_energy_error": relative_energy_error,
        "final_magnetic_divergence_linf": final_divergence,
        "references": {
            "classic_test": "Inviscid Orszag--Tang vortex energy conservation check.",
        },
    }

    validation = {
        "schema": "mhx.validation.inviscid_orszag_tang.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_relative_energy_error": max_relative_energy_error,
            "max_magnetic_divergence_linf": max_magnetic_divergence_linf,
        },
        "diagnostics": diagnostics,
    }

    return InviscidOrszagTangResult(
        time=time,
        psi=psi,
        omega=omega,
        current_density=current,
        magnetic_energy=magnetic,
        kinetic_energy=kinetic,
        total_energy=total,
        relative_energy_error=relative_energy_error,
        final_magnetic_divergence_linf=final_divergence,
        initial_state=initial_state,
        final_state=final_state,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_inviscid_orszag_tang_validation(
    outdir: str | Path,
    *,
    movies: bool = False,
    save_npz: bool = True,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write inviscid Orszag--Tang JSON, NPZ, figures, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_inviscid_orszag_tang_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "inviscid_orszag_tang.npz"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    if save_npz:
        np.savez_compressed(
            history_path,
            schema=INVISCID_ORSZAG_TANG_SCHEMA,
            time=result.time,
            psi=result.psi,
            omega=result.omega,
            current_density=result.current_density,
            magnetic_energy=result.magnetic_energy,
            kinetic_energy=result.kinetic_energy,
            total_energy=result.total_energy,
            relative_energy_error=result.relative_energy_error,
            final_magnetic_divergence_linf=result.final_magnetic_divergence_linf,
        )

    summary_path = _write_summary_figure(
        result, figure_dir / "inviscid_orszag_tang_summary.png"
    )

    outputs: dict[str, str] = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "summary": str(summary_path.relative_to(output_dir)),
    }
    if save_npz:
        outputs["history"] = history_path.name

    if movies:
        outputs.update(
            _write_inviscid_orszag_tang_movies(
                result, figure_dir=figure_dir, relative_to=output_dir
            )
        )

    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope=(
            "Inviscid reduced-MHD Orszag--Tang vortex energy conservation validation."
        ),
    )
    return manifest_path, result.validation


def _write_summary_figure(result: InviscidOrszagTangResult, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)

    axes[0].plot(result.time, result.magnetic_energy, label=r"$E_B$", color="navy", linewidth=2)
    axes[0].plot(result.time, result.kinetic_energy, label=r"$E_K$", color="crimson", linewidth=2)
    axes[0].plot(result.time, result.total_energy, label=r"$E$", color="purple",
                 linestyle="--", linewidth=2)
    axes[0].set_title("Inviscid Orszag--Tang Energy Conservation")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("energy")
    axes[0].legend(frameon=False, fontsize="small")

    energy_error = (result.total_energy - result.total_energy[0]) / result.total_energy[0]
    axes[1].plot(result.time, energy_error, color="red", linewidth=2)
    axes[1].set_title("Relative Energy Error: $(E(t) - E(0)) / E(0)$")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("Relative Error")

    # Clip the color scale to 3 standard deviations to prevent intense current
    # sheets from washing out the entire background into solid green.
    vmax = 3.0 * float(np.std(result.current_density[-1]))
    vmax = max(vmax, 1.0)

    im = axes[2].imshow(
        result.current_density[-1].T,
        cmap="jet",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
    )
    axes[2].set_title("Final Current Density ($j_z$)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im, ax=axes[2], shrink=0.75)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_inviscid_orszag_tang_movies(
    result: InviscidOrszagTangResult,
    *,
    figure_dir: Path,
    relative_to: Path,
) -> dict[str, str]:
    outputs = {
        "flux_movie": _write_scalar_movie(
            result.psi, figure_dir / "inviscid_orszag_tang_flux.gif", cmap="viridis",
            symmetric=False,
        ),
        "current_movie": _write_scalar_movie(
            result.current_density, figure_dir / "inviscid_orszag_tang_current.gif",
            cmap="RdBu_r", symmetric=True,
        ),
        "vorticity_movie": _write_scalar_movie(
            result.omega, figure_dir / "inviscid_orszag_tang_vorticity.gif",
            cmap="RdBu_r", symmetric=True,
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


def _validate_inputs(
    *,
    shape: tuple[int, int],
    dt: float,
    t_end: float,
    save_every: int,
    max_relative_energy_error: float,
    max_magnetic_divergence_linf: float,
) -> None:
    if len(shape) != 2 or shape[0] < 8 or shape[1] < 8:
        raise ValueError("shape must contain at least 8 points in each periodic direction")
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
    if max_relative_energy_error < 0.0:
        raise ValueError("max_relative_energy_error must be non-negative")
    if max_magnetic_divergence_linf <= 0.0:
        raise ValueError("max_magnetic_divergence_linf must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inviscid Orszag-Tang Benchmark")
    parser.add_argument("--nx", type=int, default=128, help="Grid points in X")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in Y")
    parser.add_argument("--dt", type=float, default=5.0e-4, help="Time step")
    parser.add_argument("--t-end", dest="t_end", type=float, default=2.0, help="Simulation end time")
    parser.add_argument("--save-every", type=int, default=40, help="Save interval")
    parser.add_argument("--outdir", type=str, default="outputs/inviscid_orszag_tang_output", help="Output directory")
    parser.add_argument("--movies", action="store_true", help="Generate output GIFs")
    parser.add_argument("--no-npz", action="store_true", help="Do not save the heavy NPZ history file")
    args = parser.parse_args()

    print("Running Inviscid Orszag-Tang Benchmark...")
    print(f"Grid: {args.nx}x{args.ny} | dt: {args.dt} | t_end: {args.t_end} | Movies: {args.movies}")
    print(f"Output directory: {Path(args.outdir).resolve()}")

    manifest_path, validation = write_inviscid_orszag_tang_validation(
        outdir=args.outdir,
        shape=(args.nx, args.ny),
        dt=args.dt,
        t_end=args.t_end,
        save_every=args.save_every,
        movies=args.movies,
        save_npz=not args.no_npz,
    )

    passed = validation.get("passed", False)
    print(f"wrote {manifest_path}")
    print(f"passed={passed}")
    if not passed:
        print("Failed checks:")
        for check, passed_check in validation.get("checks", {}).items():
            if not passed_check:
                print(f"  - {check}")


if __name__ == "__main__":
    main()
