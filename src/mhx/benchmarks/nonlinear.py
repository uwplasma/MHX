"""Nonlinear reduced-MHD validation gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.diagnostics import kinetic_energy, magnetic_energy
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.plotting import plot_nonlinear_energy_budget
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4

NONLINEAR_ENERGY_BUDGET_SCHEMA = "mhx.validation.nonlinear_energy_budget.v1"


@dataclass(frozen=True)
class NonlinearEnergyBudgetResult:
    """Saved arrays and validation gates for a nonlinear reduced-MHD energy budget."""

    time: np.ndarray
    magnetic_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    current_dissipation: np.ndarray
    viscous_dissipation: np.ndarray
    total_dissipation: np.ndarray
    cumulative_dissipation: np.ndarray
    budget_residual: np.ndarray
    relative_budget_residual: np.ndarray
    nonlinear_rhs_ratio: float
    max_relative_budget_residual: float
    max_relative_energy_growth: float
    relative_energy_drop: float
    initial_state: ReducedMHDState
    final_state: ReducedMHDState
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_nonlinear_energy_budget_validation(
    *,
    shape: tuple[int, int] = (16, 16),
    resistivity: float = 2.0e-2,
    viscosity: float = 2.0e-2,
    dt: float = 1.0e-2,
    steps: int = 80,
    save_every: int = 1,
    max_budget_residual: float = 2.0e-5,
    max_relative_energy_growth: float = 1.0e-9,
    min_nonlinear_rhs_ratio: float = 5.0e-2,
    min_relative_energy_drop: float = 1.0e-2,
) -> NonlinearEnergyBudgetResult:
    r"""Validate the nonlinear reduced-MHD energy/dissipation budget.

    For periodic resistive-viscous reduced MHD,

    ``dE/dt = -η <j²> - ν <ω²>``,

    where ``E = 0.5<|∇ψ|² + |∇φ|²>`` and ``j=-∇²ψ``.  This benchmark starts
    from a deliberately nonlinear multi-mode state, advances the full nonlinear
    RHS with RK4, and gates both monotone energy decay and the integrated
    budget residual.
    """
    _validate_energy_budget_inputs(
        shape=shape,
        resistivity=resistivity,
        viscosity=viscosity,
        dt=dt,
        steps=steps,
        save_every=save_every,
        max_budget_residual=max_budget_residual,
        max_relative_energy_growth=max_relative_energy_growth,
        min_nonlinear_rhs_ratio=min_nonlinear_rhs_ratio,
        min_relative_energy_drop=min_relative_energy_drop,
    )
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=(2.0 * np.pi, 2.0 * np.pi))
    )
    initial_state = _nonlinear_budget_initial_state(grid)
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
    states = _trajectory_with_initial_state(initial_state, trajectory.states)
    time = np.concatenate(([0.0], np.asarray(trajectory.times, dtype=np.float64)))

    magnetic = []
    kinetic = []
    current_dissipation = []
    viscous_dissipation = []
    for state in states:
        magnetic.append(float(magnetic_energy(state, lengths=grid.lengths)))
        kinetic.append(float(kinetic_energy(state, lengths=grid.lengths)))
        current = current_density(state.psi, lengths=grid.lengths)
        current_dissipation.append(float(resistivity * jnp.mean(current**2)))
        viscous_dissipation.append(float(viscosity * jnp.mean(state.omega**2)))

    magnetic_array = np.asarray(magnetic, dtype=np.float64)
    kinetic_array = np.asarray(kinetic, dtype=np.float64)
    total_array = magnetic_array + kinetic_array
    current_dissipation_array = np.asarray(current_dissipation, dtype=np.float64)
    viscous_dissipation_array = np.asarray(viscous_dissipation, dtype=np.float64)
    total_dissipation_array = current_dissipation_array + viscous_dissipation_array
    cumulative_dissipation = _cumulative_trapezoid(time, total_dissipation_array)
    budget_residual = total_array - total_array[0] + cumulative_dissipation
    energy_scale = max(abs(float(total_array[0])), np.finfo(np.float64).tiny)
    relative_budget_residual = np.abs(budget_residual) / energy_scale
    max_budget = float(np.max(relative_budget_residual))
    max_growth = float(np.max(np.diff(total_array)) / energy_scale)
    relative_energy_drop = float((total_array[0] - total_array[-1]) / energy_scale)
    nonlinear_rhs_ratio = _initial_nonlinear_rhs_ratio(
        initial_state,
        params,
        lengths=grid.lengths,
    )

    checks = {
        "finite_arrays": bool(
            np.isfinite(total_array).all()
            and np.isfinite(total_dissipation_array).all()
            and np.isfinite(relative_budget_residual).all()
        ),
        "nonlinear_rhs_active": nonlinear_rhs_ratio >= min_nonlinear_rhs_ratio,
        "energy_nonincreasing": max_growth <= max_relative_energy_growth,
        "budget_residual_within_tolerance": max_budget <= max_budget_residual,
        "net_dissipation_observed": relative_energy_drop >= min_relative_energy_drop,
        "positive_dissipation": bool(np.all(total_dissipation_array > 0.0)),
    }
    diagnostics = {
        "schema": NONLINEAR_ENERGY_BUDGET_SCHEMA,
        "shape": list(shape),
        "domain": [0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi],
        "resistivity": resistivity,
        "viscosity": viscosity,
        "dt": dt,
        "steps": steps,
        "save_every": save_every,
        "samples": int(time.size),
        "initial_total_energy": float(total_array[0]),
        "final_total_energy": float(total_array[-1]),
        "relative_energy_drop": relative_energy_drop,
        "max_relative_energy_growth": max_growth,
        "max_relative_budget_residual": max_budget,
        "nonlinear_rhs_ratio": nonlinear_rhs_ratio,
        "references": {
            "scope": (
                "Periodic nonlinear reduced-MHD energy and dissipation budget."
            ),
            "equation": "dE/dt = -eta <j^2> - nu <omega^2>",
            "review_use": (
                "This gate checks the nonlinear Poisson-bracket cancellation, "
                "spectral current operator, RK4 integration, and dissipative signs."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.nonlinear_energy_budget.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_budget_residual": max_budget_residual,
            "max_relative_energy_growth": max_relative_energy_growth,
            "min_nonlinear_rhs_ratio": min_nonlinear_rhs_ratio,
            "min_relative_energy_drop": min_relative_energy_drop,
        },
        "diagnostics": diagnostics,
    }
    final_state = states[-1]
    return NonlinearEnergyBudgetResult(
        time=time,
        magnetic_energy=magnetic_array,
        kinetic_energy=kinetic_array,
        total_energy=total_array,
        current_dissipation=current_dissipation_array,
        viscous_dissipation=viscous_dissipation_array,
        total_dissipation=total_dissipation_array,
        cumulative_dissipation=cumulative_dissipation,
        budget_residual=budget_residual,
        relative_budget_residual=relative_budget_residual,
        nonlinear_rhs_ratio=nonlinear_rhs_ratio,
        max_relative_budget_residual=max_budget,
        max_relative_energy_growth=max_growth,
        relative_energy_drop=relative_energy_drop,
        initial_state=initial_state,
        final_state=final_state,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_nonlinear_energy_budget_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write nonlinear energy-budget JSON, NPZ, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_nonlinear_energy_budget_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "nonlinear_energy_budget.npz"
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
        schema=NONLINEAR_ENERGY_BUDGET_SCHEMA,
        time=result.time,
        magnetic_energy=result.magnetic_energy,
        kinetic_energy=result.kinetic_energy,
        total_energy=result.total_energy,
        current_dissipation=result.current_dissipation,
        viscous_dissipation=result.viscous_dissipation,
        total_dissipation=result.total_dissipation,
        cumulative_dissipation=result.cumulative_dissipation,
        budget_residual=result.budget_residual,
        relative_budget_residual=result.relative_budget_residual,
        nonlinear_rhs_ratio=result.nonlinear_rhs_ratio,
        max_relative_budget_residual=result.max_relative_budget_residual,
        max_relative_energy_growth=result.max_relative_energy_growth,
        relative_energy_drop=result.relative_energy_drop,
        initial_psi=np.asarray(result.initial_state.psi),
        initial_omega=np.asarray(result.initial_state.omega),
        final_psi=np.asarray(result.final_state.psi),
        final_omega=np.asarray(result.final_state.omega),
    )
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(
            shape=tuple(result.diagnostics["shape"]),
            lower=(0.0, 0.0),
            upper=(2.0 * np.pi, 2.0 * np.pi),
        )
    )
    x, y = grid.axes()
    figure_path = plot_nonlinear_energy_budget(
        result.time,
        result.total_energy,
        result.current_dissipation,
        result.viscous_dissipation,
        result.relative_budget_residual,
        initial_psi=result.initial_state.psi,
        final_psi=result.final_state.psi,
        x=x,
        y=y,
        max_budget_residual=float(result.validation["thresholds"]["max_budget_residual"]),
        path=output_dir / "figures" / "nonlinear_energy_budget.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "nonlinear_energy_budget": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Nonlinear reduced-MHD energy/dissipation budget validation.",
    )
    return manifest_path, result.validation


def _nonlinear_budget_initial_state(grid: CartesianGrid) -> ReducedMHDState:
    x, y = grid.mesh()
    psi = (
        0.40 * jnp.cos(y)
        + 0.12 * jnp.cos(x) * jnp.cos(y)
        + 0.05 * jnp.cos(2.0 * x - y)
    )
    omega = 0.20 * jnp.sin(x) * jnp.sin(y) + 0.08 * jnp.cos(2.0 * x) * jnp.sin(y)
    return ReducedMHDState(psi=psi, omega=omega)


def _trajectory_with_initial_state(
    initial_state: ReducedMHDState,
    saved_states: ReducedMHDState,
) -> tuple[ReducedMHDState, ...]:
    return (initial_state,) + tuple(
        ReducedMHDState(psi=saved_states.psi[index], omega=saved_states.omega[index])
        for index in range(saved_states.psi.shape[0])
    )


def _cumulative_trapezoid(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    integral = np.zeros_like(time)
    for index in range(1, time.size):
        integral[index] = integral[index - 1] + 0.5 * (
            values[index - 1] + values[index]
        ) * (time[index] - time[index - 1])
    return integral


def _initial_nonlinear_rhs_ratio(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
) -> float:
    full_rhs = reduced_mhd_rhs(state, params, lengths=lengths)
    ideal_rhs = reduced_mhd_rhs(
        state,
        ReducedMHDParams(resistivity=0.0, viscosity=0.0),
        lengths=lengths,
    )
    full_norm = _state_norm(full_rhs)
    ideal_norm = _state_norm(ideal_rhs)
    return float(ideal_norm / max(full_norm, np.finfo(np.float64).tiny))


def _state_norm(state: ReducedMHDState) -> float:
    return float(jnp.sqrt(jnp.mean(state.psi**2) + jnp.mean(state.omega**2)))


def _validate_energy_budget_inputs(
    *,
    shape: tuple[int, int],
    resistivity: float,
    viscosity: float,
    dt: float,
    steps: int,
    save_every: int,
    max_budget_residual: float,
    max_relative_energy_growth: float,
    min_nonlinear_rhs_ratio: float,
    min_relative_energy_drop: float,
) -> None:
    if len(shape) != 2 or shape[0] < 8 or shape[1] < 8:
        raise ValueError("shape must contain at least 8 points in each periodic direction")
    if resistivity <= 0.0:
        raise ValueError("resistivity must be positive")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if steps < 4:
        raise ValueError("steps must be >= 4")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if max_budget_residual <= 0.0:
        raise ValueError("max_budget_residual must be positive")
    if max_relative_energy_growth < 0.0:
        raise ValueError("max_relative_energy_growth must be non-negative")
    if min_nonlinear_rhs_ratio <= 0.0:
        raise ValueError("min_nonlinear_rhs_ratio must be positive")
    if min_relative_energy_drop < 0.0:
        raise ValueError("min_relative_energy_drop must be non-negative")
