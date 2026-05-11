"""Validation-grade Rutherford campaign runner for reduced-MHD artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.benchmarks.campaigns import (
    RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA,
    build_rutherford_campaign_template,
)
from mhx.benchmarks.seed_robust_qi import seeded_tearing_initial_state
from mhx.config import DiagnosticsConfig, MeshConfig, PhysicsConfig, RunConfig, TimeConfig
from mhx.diagnostics import (
    island_width_from_mode,
    magnetic_divergence_linf,
    reconnected_flux_amplitude,
    trajectory_energies,
)
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest, write_reduced_mhd_trajectory_npz
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import evolve_rk4

RUTHERFORD_CAMPAIGN_RUN_SCHEMA = "mhx.validation.rutherford_campaign_run.v1"
RUTHERFORD_FAST_CAMPAIGN_SCHEMA = RUTHERFORD_CAMPAIGN_RUN_SCHEMA


@dataclass(frozen=True)
class RutherfordCampaignRunResult:
    """Saved histories and gates for a validation-grade Rutherford run."""

    trajectory: ReducedMHDTrajectory
    time: np.ndarray
    reconnected_flux: np.ndarray
    island_width: np.ndarray
    current_density_linf: np.ndarray
    magnetic_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


RutherfordFastCampaignResult = RutherfordCampaignRunResult


def _run_rutherford_campaign_fast_single(
    *,
    shape: tuple[int, int] = (24, 24),
    t_end: float = 0.24,
    dt: float = 1.0e-2,
    save_every: int = 1,
    seed: int = 0,
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    perturbation_amplitude: float = 1.0e-3,
    noise_amplitude: float = 1.0e-6,
    mode: tuple[int, int] = (1, 1),
    magnetic_shear: float = 1.0,
    max_relative_energy_growth: float = 1.0e-10,
    max_divergence_linf: float = 1.0e-10,
) -> RutherfordCampaignRunResult:
    """Run a short deterministic Rutherford-style validation campaign.

    This runner produces the output schema that the eventual long production
    campaign will use, but intentionally keeps the claim level at validation.
    It is long enough to test histories, figures, IO, and energy/divergence
    gates; it is not long enough for Rutherford-regime claims.
    """
    _validate_campaign_inputs(
        shape=shape,
        t_end=t_end,
        dt=dt,
        save_every=save_every,
        resistivity=resistivity,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        noise_amplitude=noise_amplitude,
        magnetic_shear=magnetic_shear,
        max_relative_energy_growth=max_relative_energy_growth,
        max_divergence_linf=max_divergence_linf,
    )
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=(2.0 * np.pi, 2.0 * np.pi))
    )
    state0 = seeded_tearing_initial_state(
        grid,
        seed=seed,
        perturbation_amplitude=perturbation_amplitude,
        noise_amplitude=noise_amplitude,
    )
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)
    steps = max(1, round(t_end / dt))

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(
        state0,
        rhs,
        dt=dt,
        steps=steps,
        save_every=save_every,
    )
    states = _trajectory_with_initial_state(state0, trajectory.states)
    time = np.concatenate(([0.0], np.asarray(trajectory.times, dtype=np.float64)))
    reconnected = np.asarray(
        [float(reconnected_flux_amplitude(state, mode=mode)) for state in states],
        dtype=np.float64,
    )
    width = np.asarray(
        [
            float(
                island_width_from_mode(
                    state,
                    mode=mode,
                    magnetic_shear=magnetic_shear,
                )
            )
            for state in states
        ],
        dtype=np.float64,
    )
    current_linf = np.asarray(
        [
            float(jnp.max(jnp.abs(current_density(state.psi, lengths=grid.lengths))))
            for state in states
        ],
        dtype=np.float64,
    )
    saved_energies = trajectory_energies(trajectory, lengths=grid.lengths)
    initial_energy = trajectory_energies(
        ReducedMHDTrajectory(
            times=jnp.asarray([0.0]),
            states=ReducedMHDState(
                psi=jnp.expand_dims(state0.psi, 0),
                omega=jnp.expand_dims(state0.omega, 0),
            ),
        ),
        lengths=grid.lengths,
    )
    magnetic = np.concatenate(
        (
            np.asarray(initial_energy["magnetic"], dtype=np.float64),
            np.asarray(saved_energies["magnetic"], dtype=np.float64),
        )
    )
    kinetic = np.concatenate(
        (
            np.asarray(initial_energy["kinetic"], dtype=np.float64),
            np.asarray(saved_energies["kinetic"], dtype=np.float64),
        )
    )
    total = magnetic + kinetic
    energy_scale = max(abs(float(total[0])), np.finfo(np.float64).tiny)
    relative_energy_growth = float(np.max(np.diff(total)) / energy_scale)
    final_state = states[-1]
    divergence = float(magnetic_divergence_linf(final_state, lengths=grid.lengths))
    checks = {
        "finite_histories": bool(
            np.isfinite(reconnected).all()
            and np.isfinite(width).all()
            and np.isfinite(current_linf).all()
            and np.isfinite(total).all()
        ),
        "enough_saved_samples": bool(time.size >= 5),
        "positive_island_width_proxy": bool(np.all(width > 0.0)),
        "energy_nonincreasing": relative_energy_growth <= max_relative_energy_growth,
        "divergence_below_tolerance": divergence <= max_divergence_linf,
        "reconnected_flux_changes": bool(abs(reconnected[-1] - reconnected[0]) > 0.0),
    }
    diagnostics = {
        "schema": RUTHERFORD_CAMPAIGN_RUN_SCHEMA,
        "shape": list(shape),
        "domain": [0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi],
        "seed": int(seed),
        "t_end": t_end,
        "dt": dt,
        "save_every": save_every,
        "samples": int(time.size),
        "resistivity": resistivity,
        "viscosity": viscosity,
        "perturbation_amplitude": perturbation_amplitude,
        "noise_amplitude": noise_amplitude,
        "mode": list(mode),
        "magnetic_shear": magnetic_shear,
        "initial_reconnected_flux": float(reconnected[0]),
        "final_reconnected_flux": float(reconnected[-1]),
        "initial_island_width": float(width[0]),
        "final_island_width": float(width[-1]),
        "initial_total_energy": float(total[0]),
        "final_total_energy": float(total[-1]),
        "max_relative_energy_growth": relative_energy_growth,
        "final_magnetic_divergence_linf": divergence,
        "claim_boundary": (
            "Short validation campaign for IO, diagnostics, and gates; not a "
            "long-time Rutherford-growth production result."
        ),
    }
    validation = {
        "schema": "mhx.validation.rutherford_campaign_run.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_relative_energy_growth": max_relative_energy_growth,
            "max_divergence_linf": max_divergence_linf,
        },
        "diagnostics": diagnostics,
    }
    return RutherfordCampaignRunResult(
        trajectory=trajectory,
        time=time,
        reconnected_flux=reconnected,
        island_width=width,
        current_density_linf=current_linf,
        magnetic_energy=magnetic,
        kinetic_energy=kinetic,
        total_energy=total,
        diagnostics=diagnostics,
        validation=validation,
    )


def run_rutherford_campaign_fast(
    outdir: str | Path | None = None,
    *,
    seeds: int | Sequence[int] | None = None,
    shape: tuple[int, int] = (16, 16),
    t_end: float | None = None,
    steps: int | None = 20,
    dt: float = 1.0e-2,
    save_every: int = 1,
    seed: int = 0,
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    perturbation_amplitude: float = 1.0e-3,
    noise_amplitude: float = 1.0e-6,
    mode: tuple[int, int] = (1, 1),
    magnetic_shear: float = 1.0,
    max_fast_time: float = 2.0,
    max_relative_energy_growth: float = 1.0e-10,
    max_divergence_linf: float = 1.0e-10,
    claim_level: str = "validation",
    make_figures: bool = True,
) -> RutherfordCampaignRunResult | tuple[Path, dict[str, Any]]:
    """Run or write a deterministic FAST Rutherford validation campaign.

    With ``outdir=None`` this preserves the in-memory single-seed runner. With
    ``outdir`` set, it writes seed-ensemble histories, validation JSON, a
    manifest, and cheap figures while keeping the claim level at ``validation``
    or ``smoke``.
    """
    if claim_level not in {"smoke", "validation"}:
        raise ValueError("claim_level must be 'smoke' or 'validation'")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if steps is not None and steps < 1:
        raise ValueError("steps must be >= 1")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    effective_t_end = float(t_end) if t_end is not None else float((steps or 1) * dt)
    effective_steps = max(1, round(effective_t_end / dt))
    if effective_steps % save_every != 0:
        raise ValueError("save_every must evenly divide steps for fixed final histories")

    if outdir is None and seeds is None:
        return _run_rutherford_campaign_fast_single(
            shape=shape,
            t_end=effective_t_end,
            dt=dt,
            save_every=save_every,
            seed=seed,
            resistivity=resistivity,
            viscosity=viscosity,
            perturbation_amplitude=perturbation_amplitude,
            noise_amplitude=noise_amplitude,
            mode=mode,
            magnetic_shear=magnetic_shear,
            max_relative_energy_growth=max_relative_energy_growth,
            max_divergence_linf=max_divergence_linf,
        )
    if outdir is None:
        raise ValueError("outdir is required when running a seed-list campaign")

    normalized_seeds = _normalize_campaign_seeds(seed if seeds is None else seeds)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = build_rutherford_campaign_template(
        shape=shape,
        dt=dt,
        run_output_dir=output_dir / "production_template_placeholder",
    )
    results = [
        _run_rutherford_campaign_fast_single(
            shape=shape,
            t_end=effective_t_end,
            dt=dt,
            save_every=save_every,
            seed=active_seed,
            resistivity=resistivity,
            viscosity=viscosity,
            perturbation_amplitude=perturbation_amplitude,
            noise_amplitude=noise_amplitude,
            mode=mode,
            magnetic_shear=magnetic_shear,
            max_relative_energy_growth=max_relative_energy_growth,
            max_divergence_linf=max_divergence_linf,
        )
        for active_seed in normalized_seeds
    ]
    time = results[0].time
    reconnected_flux = np.stack([result.reconnected_flux for result in results])
    island_width = np.stack([result.island_width for result in results])
    current_linf = np.stack([result.current_density_linf for result in results])
    magnetic = np.stack([result.magnetic_energy for result in results])
    kinetic = np.stack([result.kinetic_energy for result in results])
    total = np.stack([result.total_energy for result in results])
    divergence = np.asarray(
        [
            float(result.diagnostics["final_magnetic_divergence_linf"])
            for result in results
        ],
        dtype=np.float64,
    )
    reconnection_rate_proxy = np.stack(
        [np.gradient(row, time) for row in reconnected_flux]
    )
    production_t_end = float(template.config.time.t1)
    final_time = float(time[-1])
    max_growth = float(
        max(result.diagnostics["max_relative_energy_growth"] for result in results)
    )
    max_divergence = float(np.max(divergence))
    checks = {
        "finite_histories": bool(
            np.isfinite(reconnected_flux).all()
            and np.isfinite(island_width).all()
            and np.isfinite(total).all()
            and np.isfinite(current_linf).all()
        ),
        "saved_initial_and_final_samples": bool(time.size >= 2 and np.isclose(time[0], 0.0)),
        "duration_within_fast_limit": final_time <= max_fast_time,
        "duration_flagged_below_production_template": final_time < production_t_end,
        "energy_growth_within_tolerance": max_growth <= max_relative_energy_growth,
        "magnetic_divergence_within_tolerance": max_divergence <= max_divergence_linf,
        "claim_level_is_not_production": claim_level in {"smoke", "validation"},
    }
    diagnostics = {
        "schema": RUTHERFORD_FAST_CAMPAIGN_SCHEMA,
        "template_schema": RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA,
        "claim_level": claim_level,
        "claim_boundary": (
            "FAST nonlinear reduced-MHD campaign runner for deterministic smoke/"
            "validation artifacts only; not a production Rutherford-growth claim."
        ),
        "seeds": list(normalized_seeds),
        "shape": list(shape),
        "t_end": final_time,
        "dt": dt,
        "steps": int(effective_steps),
        "save_every": save_every,
        "samples": int(time.size),
        "mode": list(mode),
        "resistivity": resistivity,
        "viscosity": viscosity,
        "perturbation_amplitude": perturbation_amplitude,
        "noise_amplitude": noise_amplitude,
        "max_relative_energy_growth": max_growth,
        "max_magnetic_divergence_linf": max_divergence,
        "production_template_final_time": production_t_end,
        "duration_fraction_of_template": final_time / production_t_end,
        "template_claim_level": template.diagnostics["claim_level"],
        "required_runtime_outputs_inherited": template.diagnostics[
            "required_runtime_outputs"
        ],
    }
    validation = {
        "schema": "mhx.validation.rutherford_campaign_run.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_fast_time": max_fast_time,
            "production_template_final_time": production_t_end,
            "max_relative_energy_growth": max_relative_energy_growth,
            "max_magnetic_divergence_linf": max_divergence_linf,
        },
        "diagnostics": diagnostics,
    }

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    template_path = output_dir / "campaign_template.json"
    history_path = output_dir / "rutherford_fast_histories.npz"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    template_path.write_text(
        json.dumps(template.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        history_path,
        schema=np.asarray(RUTHERFORD_FAST_CAMPAIGN_SCHEMA),
        seed=np.asarray(normalized_seeds, dtype=np.int64),
        time=time,
        reconnected_flux=reconnected_flux,
        rutherford_island_width=island_width,
        reconnection_rate_proxy=reconnection_rate_proxy,
        current_density_linf=current_linf,
        magnetic_energy=magnetic,
        kinetic_energy=kinetic,
        total_energy=total,
        magnetic_divergence_linf=divergence,
        final_magnetic_divergence_linf=divergence,
    )
    figure_path = _write_ensemble_campaign_figure(
        output_dir / "figures" / "rutherford_fast_histories.png",
        time=time,
        seeds=normalized_seeds,
        reconnected_flux=reconnected_flux,
        island_width=island_width,
        total_energy=total,
        make_figures=make_figures,
    )
    outputs = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "campaign_template": template_path.name,
        "histories": history_path.name,
    }
    if figure_path is not None:
        outputs["rutherford_fast_histories"] = str(figure_path.relative_to(output_dir))
    write_manifest(
        manifest_path,
        config=diagnostics,
        outputs=outputs,
        claim_level=claim_level,
        claim_scope=(
            "Deterministic FAST Rutherford-campaign smoke/validation run; not a "
            "production nonlinear reconnection claim."
        ),
    )
    return manifest_path, validation


def write_rutherford_campaign_fast(
    outdir: str | Path,
    *,
    write_gif: bool = False,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write a validation-grade Rutherford campaign run artifact bundle."""
    if "seeds" in kwargs or "steps" in kwargs or "claim_level" in kwargs:
        kwargs.pop("write_gif", None)
        return run_rutherford_campaign_fast(outdir, **kwargs)
    from mhx.plotting import plot_flux_gif

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_rutherford_campaign_fast(**kwargs)
    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "rutherford_history.npz"
    trajectory_path = output_dir / "trajectory.npz"
    manifest_path = output_dir / "manifest.json"
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
        schema=np.asarray(RUTHERFORD_CAMPAIGN_RUN_SCHEMA),
        time=result.time,
        reconnected_flux=result.reconnected_flux,
        island_width=result.island_width,
        current_density_linf=result.current_density_linf,
        magnetic_energy=result.magnetic_energy,
        kinetic_energy=result.kinetic_energy,
        total_energy=result.total_energy,
    )
    config = _campaign_run_config(result.diagnostics)
    write_reduced_mhd_trajectory_npz(
        trajectory_path,
        trajectory=result.trajectory,
        config=config,
        diagnostics=result.diagnostics,
    )
    figure_paths = _write_campaign_figures(result, output_dir / "figures")
    if write_gif:
        figure_paths["flux_movie"] = plot_flux_gif(
            result.trajectory,
            path=output_dir / "figures" / "flux_movie.gif",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
        )
    outputs = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "history": history_path.name,
        "trajectory": trajectory_path.name,
        **{
            key: str(path.relative_to(output_dir))
            for key, path in figure_paths.items()
        },
    }
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope=(
            "Short Rutherford-style campaign runner validation; not a production "
            "nonlinear reconnection claim."
        ),
    )
    return manifest_path, result.validation


def _campaign_run_config(diagnostics: dict[str, Any]) -> dict[str, Any]:
    config = RunConfig(
        name="rutherford_campaign_fast",
        mesh=MeshConfig(shape=tuple(diagnostics["shape"])),
        time=TimeConfig(
            t1=float(diagnostics["t_end"]),
            dt=float(diagnostics["dt"]),
            save_every=int(diagnostics["save_every"]),
        ),
        physics=PhysicsConfig(
            model="reduced_mhd_rutherford_campaign_fast",
            resistivity=float(diagnostics["resistivity"]),
            viscosity=float(diagnostics["viscosity"]),
            equilibrium_parameters={
                "perturbation_amplitude": float(diagnostics["perturbation_amplitude"]),
                "noise_amplitude": float(diagnostics["noise_amplitude"]),
                "seed": float(diagnostics["seed"]),
            },
        ),
        diagnostics=DiagnosticsConfig(mode=tuple(diagnostics["mode"])),
    )
    return config.to_dict()


def _write_campaign_figures(
    result: RutherfordCampaignRunResult,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    histories_path = figure_dir / "rutherford_histories.png"
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), constrained_layout=True)
    axes[0, 0].plot(result.time, result.reconnected_flux)
    axes[0, 0].set_title("Reconnected-flux proxy")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(r"$2|\hat\psi_k|$")
    axes[0, 1].plot(result.time, result.island_width)
    axes[0, 1].set_title("Rutherford width proxy")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(r"$W$")
    axes[1, 0].plot(result.time, result.current_density_linf)
    axes[1, 0].set_title("Current-density proxy")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r"$\|j\|_\infty$")
    axes[1, 1].plot(result.time, result.magnetic_energy, label=r"$E_B$")
    axes[1, 1].plot(result.time, result.kinetic_energy, label=r"$E_K$")
    axes[1, 1].plot(result.time, result.total_energy, label=r"$E$")
    axes[1, 1].set_title("Energy history")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].legend(frameon=False)
    fig.savefig(histories_path, dpi=220)
    plt.close(fig)
    return {"rutherford_histories": histories_path}


def _normalize_campaign_seeds(seeds: int | Sequence[int]) -> tuple[int, ...]:
    normalized = (seeds,) if isinstance(seeds, int) else tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("seeds must contain at least one seed")
    return normalized


def _write_ensemble_campaign_figure(
    path: Path,
    *,
    time: np.ndarray,
    seeds: tuple[int, ...],
    reconnected_flux: np.ndarray,
    island_width: np.ndarray,
    total_energy: np.ndarray,
    make_figures: bool,
) -> Path | None:
    if not make_figures:
        return None
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for seed_index, active_seed in enumerate(seeds):
        label = f"seed {active_seed}"
        axes[0].plot(time, reconnected_flux[seed_index], label=label)
        axes[1].plot(time, island_width[seed_index], label=label)
        axes[2].plot(time, total_energy[seed_index], label=label)
    axes[0].set_title("Reconnected-flux proxy")
    axes[1].set_title("Rutherford width proxy")
    axes[2].set_title("Total energy")
    for axis in axes:
        axis.set_xlabel("time")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize="small")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _trajectory_with_initial_state(
    initial_state: ReducedMHDState,
    saved_states: ReducedMHDState,
) -> tuple[ReducedMHDState, ...]:
    return (initial_state,) + tuple(
        ReducedMHDState(psi=saved_states.psi[index], omega=saved_states.omega[index])
        for index in range(saved_states.psi.shape[0])
    )


def _validate_campaign_inputs(
    *,
    shape: tuple[int, int],
    t_end: float,
    dt: float,
    save_every: int,
    resistivity: float,
    viscosity: float,
    perturbation_amplitude: float,
    noise_amplitude: float,
    magnetic_shear: float,
    max_relative_energy_growth: float,
    max_divergence_linf: float,
) -> None:
    if len(shape) != 2 or shape[0] < 8 or shape[1] < 8:
        raise ValueError("shape must contain at least 8 points in each direction")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if dt <= 0.0 or dt > t_end:
        raise ValueError("dt must be positive and no larger than t_end")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if resistivity < 0.0 or viscosity < 0.0:
        raise ValueError("resistivity and viscosity must be non-negative")
    if perturbation_amplitude <= 0.0:
        raise ValueError("perturbation_amplitude must be positive")
    if noise_amplitude < 0.0:
        raise ValueError("noise_amplitude must be non-negative")
    if magnetic_shear <= 0.0:
        raise ValueError("magnetic_shear must be positive")
    if max_relative_energy_growth < 0.0:
        raise ValueError("max_relative_energy_growth must be non-negative")
    if max_divergence_linf < 0.0:
        raise ValueError("max_divergence_linf must be non-negative")
