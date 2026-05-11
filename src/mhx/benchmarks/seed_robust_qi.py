"""Seed-robust quality indicators for reduced-MHD FAST trajectories."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import DiagnosticsConfig, MeshConfig, PhysicsConfig, RunConfig, TimeConfig
from mhx.diagnostics import compute_reduced_mhd_diagnostics
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest, write_reduced_mhd_trajectory_npz
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import evolve_rk4

SEED_ROBUST_QI_SCHEMA = "mhx.validation.seed_robust_qi.v1"
SEED_ROBUST_QI_GATES_SCHEMA = "mhx.validation.seed_robust_qi.gates.v1"
DEFAULT_QI_METRICS = (
    "gamma_fit",
    "final_total_energy",
    "final_magnetic_energy",
    "final_kinetic_energy",
    "final_magnetic_divergence_linf",
)


@dataclass(frozen=True)
class QIMetricGate:
    """Simple pass/fail gate for one seed-ensemble metric."""

    metric: str = ""
    statistic: str = ""
    threshold: float | None = None
    sense: str = "less_equal"
    max_abs_cv: float | None = None
    max_std: float | None = None
    max_abs_mean: float | None = None
    max_abs_max: float | None = None
    cv_floor: float = 1.0e-300


@dataclass(frozen=True)
class SeedRobustQIResult:
    """Seed ensemble metrics and validation gates for a FAST reduced-MHD run."""

    seeds: tuple[int, ...]
    metrics: dict[str, np.ndarray]
    metric_summary: dict[str, dict[str, float]]
    trajectories: tuple[ReducedMHDTrajectory, ...]
    diagnostics_by_seed: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    validation: dict[str, Any]
    metric_names: tuple[str, ...] = ()
    metric_values: np.ndarray | None = None
    summaries: dict[str, dict[str, Any]] | None = None


def run_seed_robust_qi(
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    shape: tuple[int, int] = (16, 16),
    t_end: float = 0.12,
    dt: float = 1.0e-2,
    save_every: int = 1,
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    perturbation_amplitude: float = 1.0e-3,
    noise_amplitude: float = 1.0e-6,
    max_gamma_normalized_std: float = 5.0e-2,
    max_energy_normalized_std: float = 1.0e-3,
    max_relative_energy_growth: float = 1.0e-10,
    max_divergence_linf: float = 1.0e-10,
) -> SeedRobustQIResult:
    """Run a deterministic seed ensemble and gate metric sensitivity.

    The ensemble perturbs the same coherent tearing seed with smooth low-mode
    noise.  This is not a turbulent-statistics claim; it is a quality indicator
    that the FAST benchmark metrics are insensitive to tiny admissible seed
    perturbations.
    """
    _validate_seed_qi_inputs(
        seeds=seeds,
        shape=shape,
        t_end=t_end,
        dt=dt,
        save_every=save_every,
        resistivity=resistivity,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        noise_amplitude=noise_amplitude,
        max_gamma_normalized_std=max_gamma_normalized_std,
        max_energy_normalized_std=max_energy_normalized_std,
        max_relative_energy_growth=max_relative_energy_growth,
        max_divergence_linf=max_divergence_linf,
    )
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=(2.0 * np.pi, 2.0 * np.pi))
    )
    steps = max(1, round(t_end / dt))
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    trajectories: list[ReducedMHDTrajectory] = []
    diagnostics_by_seed: list[dict[str, Any]] = []
    metric_rows: dict[str, list[float]] = {
        "gamma_fit": [],
        "initial_total_energy": [],
        "final_total_energy": [],
        "relative_energy_growth": [],
        "final_mode_amplitude": [],
        "final_magnetic_divergence_linf": [],
    }
    for seed in seeds:
        state0 = seeded_tearing_initial_state(
            grid,
            seed=seed,
            perturbation_amplitude=perturbation_amplitude,
            noise_amplitude=noise_amplitude,
        )

        def rhs(state: ReducedMHDState) -> ReducedMHDState:
            return reduced_mhd_rhs(state, params, lengths=grid.lengths)

        trajectory = evolve_rk4(
            state0,
            rhs,
            dt=dt,
            steps=steps,
            save_every=save_every,
        )
        diagnostics = compute_reduced_mhd_diagnostics(
            trajectory,
            initial_state=state0,
            lengths=grid.lengths,
            quantities=("energy", "mode_growth", "divergence_error"),
            mode=(1, 1),
            fit_time_window=(min(dt, t_end), t_end),
        )
        initial_energy = float(diagnostics["initial_total_energy"])
        final_energy = float(diagnostics["final_total_energy"])
        relative_growth = (final_energy - initial_energy) / max(
            abs(initial_energy),
            np.finfo(np.float64).tiny,
        )
        diagnostics["seed"] = int(seed)
        diagnostics["relative_energy_growth"] = float(relative_growth)
        trajectories.append(trajectory)
        diagnostics_by_seed.append(diagnostics)
        for key in metric_rows:
            metric_rows[key].append(float(diagnostics[key]))

    metrics = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in metric_rows.items()
    }
    metric_summary = {
        key: _metric_summary(values)
        for key, values in metrics.items()
    }
    gamma_normalized_std = _normalized_std(metrics["gamma_fit"])
    energy_normalized_std = _normalized_std(metrics["final_total_energy"])
    checks = {
        "finite_metrics": bool(all(np.isfinite(values).all() for values in metrics.values())),
        "gamma_seed_spread_within_tolerance": gamma_normalized_std
        <= max_gamma_normalized_std,
        "energy_seed_spread_within_tolerance": energy_normalized_std
        <= max_energy_normalized_std,
        "energy_nonincreasing_all_seeds": bool(
            np.max(metrics["relative_energy_growth"]) <= max_relative_energy_growth
        ),
        "divergence_below_tolerance_all_seeds": bool(
            np.max(metrics["final_magnetic_divergence_linf"]) <= max_divergence_linf
        ),
    }
    diagnostics = {
        "schema": SEED_ROBUST_QI_SCHEMA,
        "seeds": list(seeds),
        "shape": list(shape),
        "t_end": t_end,
        "dt": dt,
        "save_every": save_every,
        "samples_per_seed": int(trajectories[0].times.shape[0]),
        "resistivity": resistivity,
        "viscosity": viscosity,
        "perturbation_amplitude": perturbation_amplitude,
        "noise_amplitude": noise_amplitude,
        "gamma_normalized_std": gamma_normalized_std,
        "final_total_energy_normalized_std": energy_normalized_std,
        "metrics": metric_summary,
        "references": {
            "scope": (
                "Seeded-perturbation ensemble quality indicator for deterministic "
                "reduced-MHD FAST metrics."
            ),
            "review_use": (
                "This gate is intended to catch seed-sensitive smoke metrics before "
                "training inverse-design or neural-ODE models on generated data."
            ),
        },
    }
    validation = {
        "schema": SEED_ROBUST_QI_GATES_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_gamma_normalized_std": max_gamma_normalized_std,
            "max_energy_normalized_std": max_energy_normalized_std,
            "max_relative_energy_growth": max_relative_energy_growth,
            "max_divergence_linf": max_divergence_linf,
        },
        "diagnostics": diagnostics,
    }
    return SeedRobustQIResult(
        seeds=seeds,
        metrics=metrics,
        metric_summary=metric_summary,
        trajectories=tuple(trajectories),
        diagnostics_by_seed=tuple(diagnostics_by_seed),
        diagnostics=diagnostics,
        validation=validation,
    )


def write_seed_robust_qi(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write seed-robust QI JSON, NPZ, figures, seed trajectories, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_seed_robust_qi(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    ensemble_path = output_dir / "seed_ensemble.npz"
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
        ensemble_path,
        schema=np.asarray(SEED_ROBUST_QI_SCHEMA),
        seeds=np.asarray(result.seeds, dtype=np.int64),
        **result.metrics,
    )
    seed_outputs: dict[str, str] = {}
    config = _seed_qi_run_config(result.diagnostics)
    for index, (seed, trajectory, diagnostics) in enumerate(
        zip(
            result.seeds,
            result.trajectories,
            result.diagnostics_by_seed,
            strict=True,
        )
    ):
        trajectory_path = output_dir / f"trajectory_seed_{seed}.npz"
        write_reduced_mhd_trajectory_npz(
            trajectory_path,
            trajectory=trajectory,
            config=config,
            diagnostics=diagnostics,
        )
        seed_outputs[f"trajectory_seed_{index}"] = trajectory_path.name

    figure_paths = _write_seed_qi_figures(result, output_dir / "figures")
    outputs = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "ensemble": ensemble_path.name,
        **seed_outputs,
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
            "Seed-robust reduced-MHD metric quality indicator; not a production "
            "nonlinear reconnection result."
        ),
    )
    return manifest_path, result.validation


def seeded_tearing_initial_state(
    grid: CartesianGrid,
    *,
    seed: int,
    perturbation_amplitude: float = 1.0e-3,
    noise_amplitude: float = 1.0e-6,
) -> ReducedMHDState:
    """Return a cosine-tearing state with deterministic smooth seed noise."""
    rng = np.random.default_rng(int(seed))
    x, y = grid.mesh()
    length_x, length_y = grid.lengths
    base = jnp.cos(2.0 * jnp.pi * y / length_y)
    coherent = perturbation_amplitude * jnp.cos(2.0 * jnp.pi * x / length_x) * jnp.cos(
        2.0 * jnp.pi * y / length_y
    )
    noise = jnp.zeros(grid.shape)
    for mode_x in (1, 2):
        for mode_y in (1, 2):
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            coefficient = float(rng.normal())
            wave = (
                mode_x * 2.0 * jnp.pi * x / length_x
                + mode_y * 2.0 * jnp.pi * y / length_y
                + phase
            )
            noise = noise + coefficient * jnp.cos(wave)
    rms = jnp.sqrt(jnp.mean(noise**2))
    noise = jnp.where(rms > 0.0, noise / rms, noise)
    return ReducedMHDState(
        psi=base + coherent + noise_amplitude * noise,
        omega=jnp.zeros_like(base),
    )


def seeded_perturbation(
    shape_or_grid: tuple[int, int] | CartesianGrid,
    seed: int | None = None,
    *,
    amplitude: float | None = None,
    noise_amplitude: float | None = None,
) -> np.ndarray | jnp.ndarray:
    """Return a deterministic zero-mean seed perturbation.

    Passing a ``CartesianGrid`` preserves the smooth low-mode perturbation used
    by the FAST trajectory runner. Passing a shape returns a pure NumPy
    zero-mean, unit-RMS field for statistical tests and downstream QI tooling.
    """
    if seed is None:
        raise ValueError("seed is required")
    active_amplitude = float(
        amplitude if amplitude is not None else (noise_amplitude or 0.0)
    )
    if active_amplitude < 0.0:
        raise ValueError("amplitude must be non-negative")
    if isinstance(shape_or_grid, CartesianGrid):
        state_without_noise = seeded_tearing_initial_state(
            shape_or_grid,
            seed=seed,
            perturbation_amplitude=1.0e-3,
            noise_amplitude=0.0,
        )
        state_with_noise = seeded_tearing_initial_state(
            shape_or_grid,
            seed=seed,
            perturbation_amplitude=1.0e-3,
            noise_amplitude=active_amplitude,
        )
        return state_with_noise.psi - state_without_noise.psi
    shape = tuple(int(item) for item in shape_or_grid)
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError("shape must contain at least 4 points in each direction")
    rng = np.random.default_rng(int(seed))
    perturbation = rng.standard_normal(shape)
    perturbation = perturbation - float(np.mean(perturbation))
    rms = float(np.sqrt(np.mean(perturbation**2)))
    if rms == 0.0:
        return np.zeros(shape, dtype=np.float64)
    return (active_amplitude / rms) * perturbation


def make_seeded_initial_state(
    grid_or_state: CartesianGrid | ReducedMHDState,
    *,
    seed: int,
    perturbation_amplitude: float = 1.0e-3,
    noise_amplitude: float = 1.0e-6,
    psi_noise_amplitude: float | None = None,
    omega_noise_amplitude: float = 0.0,
) -> ReducedMHDState:
    """Return a deterministic seeded reduced-MHD initial state."""
    if isinstance(grid_or_state, CartesianGrid):
        return seeded_tearing_initial_state(
            grid_or_state,
            seed=seed,
            perturbation_amplitude=perturbation_amplitude,
            noise_amplitude=noise_amplitude,
        )
    base_state = grid_or_state
    shape = tuple(int(item) for item in np.asarray(base_state.psi).shape)
    psi_amplitude = noise_amplitude if psi_noise_amplitude is None else psi_noise_amplitude
    psi_noise = seeded_perturbation(shape, seed, amplitude=psi_amplitude)
    omega_noise = seeded_perturbation(shape, seed + 1, amplitude=omega_noise_amplitude)
    return ReducedMHDState(
        psi=base_state.psi + jnp.asarray(psi_noise),
        omega=base_state.omega + jnp.asarray(omega_noise),
    )


def generate_seed_ensemble(
    base_seed_or_grid: int | CartesianGrid,
    count: int | None = None,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    perturbation_amplitude: float = 1.0e-3,
    noise_amplitude: float = 1.0e-6,
) -> np.ndarray | tuple[ReducedMHDState, ...]:
    """Return deterministic seeds, or seeded states when passed a grid."""
    if isinstance(base_seed_or_grid, CartesianGrid):
        return tuple(
            seeded_tearing_initial_state(
                base_seed_or_grid,
                seed=seed,
                perturbation_amplitude=perturbation_amplitude,
                noise_amplitude=noise_amplitude,
            )
            for seed in seeds
        )
    if count is None:
        raise ValueError("count is required when generating integer seeds")
    if count < 1:
        raise ValueError("count must be >= 1")
    base_seed = int(base_seed_or_grid)
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    seed_sequence = np.random.SeedSequence(base_seed)
    generated = np.asarray(
        [
            sequence.generate_state(1, dtype=np.uint32)[0]
            for sequence in seed_sequence.spawn(count)
        ],
        dtype=np.uint32,
    )
    if len(set(int(seed) for seed in generated)) != count:
        generated = (np.arange(count, dtype=np.uint64) + base_seed).astype(np.uint32)
    return generated


def compute_metric_statistics(
    values: np.ndarray,
    metric_names: Sequence[str] | None = None,
    *,
    gates: Mapping[str, QIMetricGate] | None = None,
) -> dict[str, float] | dict[str, dict[str, Any]]:
    """Return mean/std/CV summaries and optional pass/fail gates."""
    array = np.asarray(values, dtype=np.float64)
    if metric_names is None:
        return _metric_summary(array)
    names = tuple(str(name) for name in metric_names)
    if array.ndim != 2:
        raise ValueError("values must be a 2D array when metric_names is provided")
    if array.shape[1] != len(names):
        raise ValueError("values column count must match metric_names")
    summaries: dict[str, dict[str, Any]] = {}
    active_gates = dict(gates or {})
    for column, name in enumerate(names):
        samples = array[:, column]
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
        minimum = float(np.min(samples))
        maximum = float(np.max(samples))
        gate = active_gates.get(name)
        cv_floor = gate.cv_floor if gate is not None else 1.0e-300
        cv = float(std / max(abs(mean), cv_floor))
        checks = {"finite": bool(np.isfinite(samples).all())}
        if gate is not None:
            threshold = gate.threshold
            if threshold is not None and gate.statistic in {"normalized_std", "cv"}:
                checks["legacy_threshold_within_tolerance"] = abs(cv) <= threshold
            if threshold is not None and gate.statistic == "max":
                checks["legacy_max_within_tolerance"] = maximum <= threshold
            if gate.max_abs_cv is not None:
                checks["cv_within_tolerance"] = abs(cv) <= gate.max_abs_cv
            if gate.max_std is not None:
                checks["std_within_tolerance"] = std <= gate.max_std
            if gate.max_abs_mean is not None:
                checks["mean_within_tolerance"] = abs(mean) <= gate.max_abs_mean
            if gate.max_abs_max is not None:
                checks["max_abs_within_tolerance"] = max(abs(minimum), abs(maximum)) <= (
                    gate.max_abs_max
                )
        summaries[name] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": minimum,
            "max": maximum,
            "samples": [float(item) for item in samples],
            "checks": checks,
            "passed": all(checks.values()),
            "gate": _gate_to_dict(gate),
        }
    return summaries


def default_seed_robust_qi_gates() -> dict[str, QIMetricGate]:
    """Return default seed-robust QI gates in metadata-friendly form."""
    return {
        "gamma_fit": QIMetricGate(max_abs_cv=5.0e-2, max_std=1.0e-2),
        "final_total_energy": QIMetricGate(max_abs_cv=1.0e-3, max_std=1.0e-5),
        "final_magnetic_energy": QIMetricGate(max_abs_cv=1.0e-3, max_std=1.0e-5),
        "final_kinetic_energy": QIMetricGate(max_abs_cv=2.5e-1, max_abs_mean=1.0e-6),
        "relative_energy_growth": QIMetricGate(max_abs_max=1.0e-10),
        "final_magnetic_divergence_linf": QIMetricGate(
            max_abs_mean=1.0e-10,
            max_abs_max=1.0e-9,
        ),
    }


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "normalized_std": _normalized_std(values),
    }


def _normalized_std(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    scale = max(abs(mean), np.finfo(np.float64).eps)
    return float(np.std(values, ddof=0) / scale)


def _seed_qi_run_config(diagnostics: dict[str, Any]) -> dict[str, Any]:
    config = RunConfig(
        name="seed_robust_qi",
        mesh=MeshConfig(shape=tuple(diagnostics["shape"])),
        time=TimeConfig(t1=float(diagnostics["t_end"]), dt=float(diagnostics["dt"])),
        physics=PhysicsConfig(
            resistivity=float(diagnostics["resistivity"]),
            viscosity=float(diagnostics["viscosity"]),
            equilibrium_parameters={
                "perturbation_amplitude": float(diagnostics["perturbation_amplitude"]),
                "noise_amplitude": float(diagnostics["noise_amplitude"]),
            },
        ),
        diagnostics=DiagnosticsConfig(
            fit_time_window=(
                float(diagnostics["dt"]),
                float(diagnostics["t_end"]),
            )
        ),
    )
    return config.to_dict()


def _write_seed_qi_figures(
    result: SeedRobustQIResult,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.asarray(result.seeds, dtype=np.int64)
    metrics_path = figure_dir / "seed_robust_qi_metrics.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ("gamma_fit", "final_total_energy", "final_magnetic_divergence_linf"),
        (r"$\gamma_\mathrm{fit}$", r"$E(t_f)$", r"$\|\nabla\cdot B_\perp\|_\infty$"),
        strict=True,
    ):
        ax.plot(seeds, result.metrics[key], "o-")
        ax.set_xlabel("seed")
        ax.set_ylabel(label)
        ax.set_title(key.replace("_", " "))
    fig.savefig(metrics_path, dpi=220)
    plt.close(fig)

    energy_path = figure_dir / "seed_energy_spread.png"
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.plot(seeds, result.metrics["relative_energy_growth"], "o-")
    ax.set_xlabel("seed")
    ax.set_ylabel(r"$(E_f-E_0)/E_0$")
    ax.set_title("Seed ensemble energy monotonicity")
    fig.savefig(energy_path, dpi=220)
    plt.close(fig)
    return {
        "seed_robust_qi_metrics": metrics_path,
        "seed_energy_spread": energy_path,
    }


def _validate_seed_qi_inputs(
    *,
    seeds: tuple[int, ...],
    shape: tuple[int, int],
    t_end: float,
    dt: float,
    save_every: int,
    resistivity: float,
    viscosity: float,
    perturbation_amplitude: float,
    noise_amplitude: float,
    max_gamma_normalized_std: float,
    max_energy_normalized_std: float,
    max_relative_energy_growth: float,
    max_divergence_linf: float,
) -> None:
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
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
    if max_gamma_normalized_std < 0.0:
        raise ValueError("max_gamma_normalized_std must be non-negative")
    if max_energy_normalized_std < 0.0:
        raise ValueError("max_energy_normalized_std must be non-negative")
    if max_relative_energy_growth < 0.0:
        raise ValueError("max_relative_energy_growth must be non-negative")
    if max_divergence_linf < 0.0:
        raise ValueError("max_divergence_linf must be non-negative")


def run_seed_robust_qi_validation(
    *,
    shape: tuple[int, int] = (16, 16),
    seed_count: int = 6,
    base_seed: int = 20240511,
    seeds: Sequence[int] | None = None,
    perturbation_amplitude: float = 1.0e-3,
    psi_noise_amplitude: float = 1.0e-8,
    omega_noise_amplitude: float = 0.0,
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    dt: float = 1.0e-2,
    steps: int = 24,
    save_every: int = 1,
    mode: tuple[int, int] = (1, 1),
    fit_time_window: tuple[float, float] | None = None,
    metric_names: Sequence[str] = DEFAULT_QI_METRICS,
    gates: Mapping[str, QIMetricGate] | None = None,
) -> SeedRobustQIResult:
    """Run a tiny seeded reduced-MHD ensemble and validate QI robustness."""
    del omega_noise_amplitude
    del mode
    del fit_time_window
    active_seeds = (
        tuple(int(seed) for seed in seeds)
        if seeds is not None
        else tuple(int(seed) for seed in generate_seed_ensemble(base_seed, seed_count))
    )
    low_level = run_seed_robust_qi(
        seeds=active_seeds,
        shape=shape,
        t_end=steps * dt,
        dt=dt,
        save_every=save_every,
        resistivity=resistivity,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        noise_amplitude=psi_noise_amplitude,
    )
    names = tuple(str(name) for name in metric_names)
    rows = []
    per_seed = []
    for seed, diagnostics in zip(active_seeds, low_level.diagnostics_by_seed, strict=True):
        row = [float(diagnostics[name]) for name in names]
        rows.append(row)
        per_seed.append({"seed": int(seed), "metrics": dict(zip(names, row, strict=True))})
    metric_values = np.asarray(rows, dtype=np.float64)
    active_gates = dict(default_seed_robust_qi_gates())
    if gates is not None:
        active_gates.update(gates)
    summaries = compute_metric_statistics(metric_values, names, gates=active_gates)
    checks = {
        "ensemble_size": len(active_seeds) >= 2,
        "unique_seeds": len(set(active_seeds)) == len(active_seeds),
        "finite_metric_matrix": bool(np.isfinite(metric_values).all()),
    }
    checks.update({f"{name}_passed": summary["passed"] for name, summary in summaries.items()})
    diagnostics = {
        "schema": SEED_ROBUST_QI_SCHEMA,
        "shape": list(shape),
        "seed_count": len(active_seeds),
        "base_seed": int(base_seed),
        "seeds": [int(seed) for seed in active_seeds],
        "perturbation_amplitude": perturbation_amplitude,
        "psi_noise_amplitude": psi_noise_amplitude,
        "omega_noise_amplitude": 0.0,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "dt": dt,
        "steps": steps,
        "save_every": save_every,
        "metric_names": list(names),
        "per_seed": per_seed,
        "summaries": summaries,
        "references": {
            "scope": (
                "Seed-robust quality indicator for stochastic perturbation "
                "sensitivity of FAST reduced-MHD trajectories."
            ),
            "physics_motivation": (
                "Reduced-MHD trajectory diagnostics should be stable under "
                "small seeded flux perturbations, while B_perp remains "
                "solenoidal to spectral roundoff."
            ),
        },
    }
    validation = {
        "schema": SEED_ROBUST_QI_GATES_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            name: _gate_to_dict(gate) for name, gate in sorted(active_gates.items())
        },
        "diagnostics": diagnostics,
    }
    return SeedRobustQIResult(
        seeds=active_seeds,
        metrics={name: metric_values[:, index] for index, name in enumerate(names)},
        metric_summary=summaries,
        trajectories=low_level.trajectories,
        diagnostics_by_seed=low_level.diagnostics_by_seed,
        diagnostics=diagnostics,
        validation=validation,
        metric_names=names,
        metric_values=metric_values,
        summaries=summaries,
    )


def write_seed_robust_qi_validation(
    outdir: str | Path,
    *,
    write_figures: bool = True,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write QI diagnostics, validation, ensemble NPZ, optional figures, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_seed_robust_qi_validation(**kwargs)
    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    ensemble_path = output_dir / "ensemble.npz"
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
        ensemble_path,
        schema=np.asarray(SEED_ROBUST_QI_SCHEMA),
        seeds=np.asarray(result.seeds, dtype=np.int64),
        metric_names=np.asarray(result.metric_names),
        metric_values=result.metric_values,
    )
    outputs = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "ensemble": ensemble_path.name,
    }
    if write_figures:
        figure_path = _write_qi_summary_figure(result, output_dir / "figures" / "qi_summary.png")
        outputs["qi_summary"] = str(figure_path.relative_to(output_dir))
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope="Seed-robust QI validation for FAST reduced-MHD trajectories.",
    )
    return manifest_path, result.validation


def _write_qi_summary_figure(result: SeedRobustQIResult, path: Path) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(result.metric_names))
    means = np.asarray(
        [result.summaries[name]["mean"] for name in result.metric_names],
        dtype=np.float64,
    )
    stds = np.asarray(
        [result.summaries[name]["std"] for name in result.metric_names],
        dtype=np.float64,
    )
    cvs = np.asarray(
        [
            max(abs(result.summaries[name]["cv"]), np.finfo(np.float64).tiny)
            for name in result.metric_names
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), constrained_layout=True)
    axes[0].errorbar(x, means, yerr=stds, fmt="o", capsize=3)
    axes[0].set_yscale("symlog", linthresh=1.0e-14)
    axes[0].set_ylabel("mean ± sample std")
    axes[0].set_title("Seed-robust QI metric ensemble")
    axes[1].bar(x, cvs)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("|CV|")
    axes[1].set_xticks(x, result.metric_names, rotation=25, ha="right")
    axes[1].set_xlabel("metric")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _gate_to_dict(gate: QIMetricGate | None) -> dict[str, float | str | None] | None:
    if gate is None:
        return None
    return {
        "metric": gate.metric,
        "statistic": gate.statistic,
        "threshold": gate.threshold,
        "sense": gate.sense,
        "max_abs_cv": gate.max_abs_cv,
        "max_std": gate.max_std,
        "max_abs_mean": gate.max_abs_mean,
        "max_abs_max": gate.max_abs_max,
        "cv_floor": gate.cv_floor,
    }
