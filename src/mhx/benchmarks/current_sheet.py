"""Periodic current-sheet eigenvalue validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import linearized_reduced_mhd_operator, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.physics import CosineTearingEquilibrium
from mhx.plotting import (
    plot_nonlinear_current_sheet_bridge,
    plot_periodic_current_sheet_spectrum,
    plot_periodic_current_sheet_timedomain,
)
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    flatten_reduced_mhd_state,
    unflatten_reduced_mhd_state,
)
from mhx.time_integrators import evolve_rk4

PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA = (
    "mhx.validation.periodic_current_sheet_eigenvalue.v1"
)
PERIODIC_CURRENT_SHEET_TIMEDOMAIN_SCHEMA = (
    "mhx.validation.periodic_current_sheet_timedomain.v1"
)
PERIODIC_CURRENT_SHEET_NONLINEAR_BRIDGE_SCHEMA = (
    "mhx.validation.periodic_current_sheet_nonlinear_bridge.v1"
)


@dataclass(frozen=True)
class PeriodicCurrentSheetEigenvalueResult:
    """Tiny dense-spectrum validation around a periodic current-sheet equilibrium."""

    matrix: np.ndarray
    eigenvalues: np.ndarray
    selected_eigenvalue: complex
    selected_eigenvector: np.ndarray
    selected_residual_norm: float
    gauge_residual_norms: dict[str, float]
    gauge_mode_count: int
    max_real_part: float
    max_non_gauge_real_part: float
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class PeriodicCurrentSheetTimeDomainResult:
    """Time-domain replay of one periodic current-sheet JVP eigenmode."""

    times: np.ndarray
    amplitudes: np.ndarray
    exact_amplitudes: np.ndarray
    relative_state_error: np.ndarray
    selected_eigenvalue: float
    fitted_decay_rate: float
    relative_decay_rate_error: float
    selected_residual_norm: float
    initial_state: ReducedMHDState
    final_state: ReducedMHDState
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class PeriodicCurrentSheetNonlinearBridgeResult:
    """Differentiability bridge for the nonlinear reduced-MHD RK4 time map."""

    epsilons: np.ndarray
    relative_errors: np.ndarray
    convergence_order: float
    tangent_norm: float
    finest_relative_error: float
    perturbation: ReducedMHDState
    tangent_final: ReducedMHDState
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_periodic_current_sheet_eigenvalue_validation(
    *,
    shape: tuple[int, int] = (8, 8),
    amplitude: float = 1.0,
    resistivity: float = 2.0e-2,
    viscosity: float = 2.0e-2,
    gauge_eigenvalue_radius: float = 1.0e-9,
    max_spurious_growth: float = 1.0e-9,
    min_diffusive_decay_fraction: float = 0.25,
    max_selected_residual_norm: float = 1.0e-9,
    max_gauge_residual_norm: float = 1.0e-10,
) -> PeriodicCurrentSheetEigenvalueResult:
    r"""Solve a tiny dense JVP spectrum around ``ψ₀=A cos(y)``.

    This is deliberately not advertised as an FKR growth-rate benchmark. It is
    a stricter bridge between analytic JVP gates and future tearing calculations:
    the test assembles the flattened nonzero-equilibrium reduced-MHD operator,
    checks the two gauge/mean modes, solves the complete dense spectrum on a
    tiny grid, and verifies that the non-gauge spectrum is damped rather than
    showing spurious positive growth.
    """
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    base = CosineTearingEquilibrium(perturbation_amplitude=0.0).initial_state(grid)
    base = ReducedMHDState(psi=amplitude * base.psi, omega=base.omega)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)
    operator = linearized_reduced_mhd_operator(base, params, lengths=grid.lengths)
    matrix = _dense_matrix(operator)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    non_gauge_mask = np.abs(eigenvalues) > gauge_eigenvalue_radius
    if not np.any(non_gauge_mask):
        raise RuntimeError("periodic current-sheet spectrum has no non-gauge modes")
    non_gauge_indices = np.flatnonzero(non_gauge_mask)
    selected_index = int(non_gauge_indices[np.argmax(eigenvalues[non_gauge_mask].real)])
    selected_eigenvalue = complex(eigenvalues[selected_index])
    selected_eigenvector = np.asarray(eigenvectors[:, selected_index])
    selected_residual_norm = _dense_eigen_residual(
        matrix,
        selected_eigenvector,
        selected_eigenvalue,
    )

    gauge_residual_norms = _gauge_residuals(operator, grid.shape)
    gauge_mode_count = int(np.count_nonzero(~non_gauge_mask))
    max_real_part = float(np.max(eigenvalues.real))
    max_non_gauge_real_part = float(np.max(eigenvalues[non_gauge_mask].real))
    kmin_squared = min((2.0 * np.pi / length) ** 2 for length in grid.lengths)
    max_allowed_non_gauge_real_part = (
        -min_diffusive_decay_fraction * min(resistivity, viscosity) * kmin_squared
    )
    checks = {
        "constant_flux_gauge_mode_residual_small": (
            gauge_residual_norms["constant_flux"] <= max_gauge_residual_norm
        ),
        "constant_vorticity_gauge_mode_residual_small": (
            gauge_residual_norms["constant_vorticity"] <= max_gauge_residual_norm
        ),
        "at_least_two_gauge_modes_detected": gauge_mode_count >= 2,
        "no_spurious_positive_growth": max_real_part <= max_spurious_growth,
        "non_gauge_modes_are_damped": (
            max_non_gauge_real_part <= max_allowed_non_gauge_real_part
        ),
        "selected_dense_eigenpair_residual_small": (
            selected_residual_norm <= max_selected_residual_norm
        ),
    }
    diagnostics = {
        "schema": PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA,
        "shape": list(shape),
        "equilibrium": "psi0 = A cos(2π y / Ly), omega0 = 0",
        "amplitude": amplitude,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "operator_size": int(matrix.shape[0]),
        "gauge_eigenvalue_radius": gauge_eigenvalue_radius,
        "gauge_mode_count": gauge_mode_count,
        "gauge_residual_norms": gauge_residual_norms,
        "selected_eigenvalue": {
            "real": float(selected_eigenvalue.real),
            "imag": float(selected_eigenvalue.imag),
        },
        "selected_residual_norm": selected_residual_norm,
        "max_real_part": max_real_part,
        "max_non_gauge_real_part": max_non_gauge_real_part,
        "max_allowed_non_gauge_real_part": max_allowed_non_gauge_real_part,
        "references": {
            "scope": (
                "Dense tiny-grid spectrum of the periodic current-sheet JVP; "
                "a conservative stability/operator gate, not an FKR growth claim."
            ),
            "next_validation_step": (
                "Calibrated FKR/Coppi tearing benchmarks require a documented "
                "asymptotic equilibrium, boundary conditions, and resolution study."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.periodic_current_sheet_eigenvalue.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "gauge_eigenvalue_radius": gauge_eigenvalue_radius,
            "max_spurious_growth": max_spurious_growth,
            "min_diffusive_decay_fraction": min_diffusive_decay_fraction,
            "max_selected_residual_norm": max_selected_residual_norm,
            "max_gauge_residual_norm": max_gauge_residual_norm,
            "max_allowed_non_gauge_real_part": max_allowed_non_gauge_real_part,
        },
        "diagnostics": diagnostics,
    }
    return PeriodicCurrentSheetEigenvalueResult(
        matrix=matrix,
        eigenvalues=eigenvalues,
        selected_eigenvalue=selected_eigenvalue,
        selected_eigenvector=selected_eigenvector,
        selected_residual_norm=selected_residual_norm,
        gauge_residual_norms=gauge_residual_norms,
        gauge_mode_count=gauge_mode_count,
        max_real_part=max_real_part,
        max_non_gauge_real_part=max_non_gauge_real_part,
        diagnostics=diagnostics,
        validation=validation,
    )


def run_periodic_current_sheet_timedomain_validation(
    *,
    shape: tuple[int, int] = (8, 8),
    amplitude: float = 1.0,
    resistivity: float = 2.0e-2,
    viscosity: float = 2.0e-2,
    dt: float = 5.0e-2,
    t_end: float = 5.0,
    save_every: int = 5,
    real_eigenvalue_imag_tolerance: float = 1.0e-8,
    min_decay_rate: float = 1.0e-3,
    max_selected_residual_norm: float = 1.0e-9,
    max_relative_state_error: float = 1.0e-8,
    max_relative_decay_rate_error: float = 1.0e-8,
) -> PeriodicCurrentSheetTimeDomainResult:
    r"""Replay a real decaying eigenmode of the periodic current-sheet JVP.

    The benchmark assembles the same dense matrix-free JVP operator used by the
    spectrum gate, selects the least-damped real non-gauge eigenmode, advances
    ``dq/dt=Lq`` with RK4, and compares the trajectory to
    ``q(t)=\exp(\lambda t)q(0)``. This validates the time-domain path without
    claiming a nonlinear FKR/Coppi tearing result.
    """
    _validate_timedomain_inputs(
        shape=shape,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        real_eigenvalue_imag_tolerance=real_eigenvalue_imag_tolerance,
        min_decay_rate=min_decay_rate,
        max_selected_residual_norm=max_selected_residual_norm,
        max_relative_state_error=max_relative_state_error,
        max_relative_decay_rate_error=max_relative_decay_rate_error,
    )
    spectrum = run_periodic_current_sheet_eigenvalue_validation(
        shape=shape,
        amplitude=amplitude,
        resistivity=resistivity,
        viscosity=viscosity,
        max_selected_residual_norm=max_selected_residual_norm,
    )
    eigenvalues, eigenvectors = np.linalg.eig(spectrum.matrix)
    selected_index = _select_real_decaying_eigenpair(
        eigenvalues,
        imag_tolerance=real_eigenvalue_imag_tolerance,
        min_decay_rate=min_decay_rate,
    )
    selected_eigenvalue_complex = complex(eigenvalues[selected_index])
    selected_eigenvalue = float(selected_eigenvalue_complex.real)
    selected_vector = np.asarray(eigenvectors[:, selected_index].real, dtype=np.float64)
    selected_vector /= np.linalg.norm(selected_vector.ravel())
    selected_residual_norm = _dense_eigen_residual(
        spectrum.matrix,
        selected_vector,
        selected_eigenvalue,
    )
    steps = round(t_end / dt)
    times, states = _rk4_linear_replay(
        spectrum.matrix,
        selected_vector,
        dt=dt,
        steps=steps,
        save_every=save_every,
    )
    exact_states = np.exp(selected_eigenvalue * times)[:, None] * selected_vector[None, :]
    amplitudes = np.linalg.norm(states, axis=1)
    exact_amplitudes = np.linalg.norm(exact_states, axis=1)
    relative_state_error = np.linalg.norm(states - exact_states, axis=1) / np.maximum(
        exact_amplitudes,
        1.0e-300,
    )
    fitted_decay_rate = float(np.polyfit(times, np.log(amplitudes), 1)[0])
    relative_decay_rate_error = abs(
        (fitted_decay_rate - selected_eigenvalue) / selected_eigenvalue
    )
    checks = {
        "spectrum_gate_passes": bool(spectrum.validation["passed"]),
        "selected_mode_is_real_and_decaying": (
            abs(selected_eigenvalue_complex.imag) <= real_eigenvalue_imag_tolerance
            and selected_eigenvalue <= -min_decay_rate
        ),
        "selected_dense_eigenpair_residual_small": (
            selected_residual_norm <= max_selected_residual_norm
        ),
        "rk4_replay_matches_exact_exponential": (
            float(np.max(relative_state_error)) <= max_relative_state_error
        ),
        "fitted_decay_rate_matches_eigenvalue": (
            relative_decay_rate_error <= max_relative_decay_rate_error
        ),
        "amplitude_decays": bool(amplitudes[-1] < amplitudes[0]),
    }
    diagnostics = {
        "schema": PERIODIC_CURRENT_SHEET_TIMEDOMAIN_SCHEMA,
        "shape": list(shape),
        "equilibrium": "psi0 = A cos(2π y / Ly), omega0 = 0",
        "amplitude": amplitude,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "dt": dt,
        "t_end": t_end,
        "save_every": save_every,
        "steps": steps,
        "selected_eigenvalue": selected_eigenvalue,
        "fitted_decay_rate": fitted_decay_rate,
        "relative_decay_rate_error": relative_decay_rate_error,
        "selected_residual_norm": selected_residual_norm,
        "max_relative_state_error": float(np.max(relative_state_error)),
        "initial_amplitude": float(amplitudes[0]),
        "final_amplitude": float(amplitudes[-1]),
        "references": {
            "scope": (
                "Linear time-domain replay of the periodic reduced-MHD JVP; "
                "a solver/operator bridge, not a nonlinear tearing benchmark."
            ),
            "equation": "dq/dt = L q, q(t)=exp(lambda t) q(0)",
            "review_use": (
                "This gate catches mismatches between assembled linear operators, "
                "RK4 time stepping, and eigenmode diagnostics before nonlinear runs."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.periodic_current_sheet_timedomain.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "real_eigenvalue_imag_tolerance": real_eigenvalue_imag_tolerance,
            "min_decay_rate": min_decay_rate,
            "max_selected_residual_norm": max_selected_residual_norm,
            "max_relative_state_error": max_relative_state_error,
            "max_relative_decay_rate_error": max_relative_decay_rate_error,
        },
        "diagnostics": diagnostics,
    }
    initial_state = unflatten_reduced_mhd_state(selected_vector, shape)
    final_state = unflatten_reduced_mhd_state(states[-1], shape)
    return PeriodicCurrentSheetTimeDomainResult(
        times=times,
        amplitudes=amplitudes,
        exact_amplitudes=exact_amplitudes,
        relative_state_error=relative_state_error,
        selected_eigenvalue=selected_eigenvalue,
        fitted_decay_rate=fitted_decay_rate,
        relative_decay_rate_error=relative_decay_rate_error,
        selected_residual_norm=selected_residual_norm,
        initial_state=initial_state,
        final_state=final_state,
        diagnostics=diagnostics,
        validation=validation,
    )


def run_periodic_current_sheet_nonlinear_bridge_validation(
    *,
    shape: tuple[int, int] = (8, 8),
    amplitude: float = 1.0,
    resistivity: float = 2.0e-2,
    viscosity: float = 2.0e-2,
    dt: float = 2.0e-2,
    steps: int = 8,
    save_every: int = 2,
    epsilons: tuple[float, ...] = (1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3),
    min_convergence_order: float = 1.8,
    max_finest_relative_error: float = 1.0e-7,
) -> PeriodicCurrentSheetNonlinearBridgeResult:
    r"""Validate the nonlinear RK4 time-map derivative around a current sheet.

    This benchmark compares the JAX JVP of the complete saved-trajectory map
    ``\Phi(q_0)`` to centered finite differences of nonlinear trajectories:

    ``[\Phi(q_0+\epsilon p)-\Phi(q_0-\epsilon p)]/(2\epsilon)``.

    For smooth reduced-MHD RHS terms and x64 arithmetic, the centered
    finite-difference error should converge as ``O(epsilon**2)`` until roundoff.
    """
    _validate_nonlinear_bridge_inputs(
        shape=shape,
        dt=dt,
        steps=steps,
        save_every=save_every,
        epsilons=epsilons,
        min_convergence_order=min_convergence_order,
        max_finest_relative_error=max_finest_relative_error,
    )
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    base = CosineTearingEquilibrium(perturbation_amplitude=0.0).initial_state(grid)
    base = ReducedMHDState(psi=amplitude * base.psi, omega=base.omega)
    perturbation = _normalized_bridge_perturbation(grid)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    def flattened_trajectory(state: ReducedMHDState) -> jnp.ndarray:
        trajectory = evolve_rk4(
            state,
            rhs,
            dt=dt,
            steps=steps,
            save_every=save_every,
        )
        return jnp.concatenate(
            (
                jnp.ravel(trajectory.states.psi),
                jnp.ravel(trajectory.states.omega),
            )
        )

    _, tangent = jax.jvp(flattened_trajectory, (base,), (perturbation,))
    tangent_norm = _array_norm(tangent)
    relative_errors = []
    for epsilon in epsilons:
        plus = ReducedMHDState(
            psi=base.psi + epsilon * perturbation.psi,
            omega=base.omega + epsilon * perturbation.omega,
        )
        minus = ReducedMHDState(
            psi=base.psi - epsilon * perturbation.psi,
            omega=base.omega - epsilon * perturbation.omega,
        )
        finite_difference = (
            flattened_trajectory(plus) - flattened_trajectory(minus)
        ) / (2.0 * epsilon)
        relative_errors.append(_array_norm(finite_difference - tangent) / tangent_norm)
    epsilon_array = np.asarray(epsilons, dtype=np.float64)
    error_array = np.asarray(relative_errors, dtype=np.float64)
    convergence_order = float(np.polyfit(np.log(epsilon_array), np.log(error_array), 1)[0])
    finest_relative_error = float(error_array[-1])
    saved_count = steps // min(save_every, steps)
    final_split = shape[0] * shape[1]
    tangent_final_vector = np.asarray(tangent[-2 * final_split :])
    tangent_final = unflatten_reduced_mhd_state(jnp.asarray(tangent_final_vector), shape)
    checks = {
        "finite_positive_errors": bool(np.all(np.isfinite(error_array) & (error_array > 0.0))),
        "errors_decrease_with_epsilon": bool(np.all(np.diff(error_array) < 0.0)),
        "centered_difference_is_second_order": convergence_order >= min_convergence_order,
        "finest_error_within_tolerance": (
            finest_relative_error <= max_finest_relative_error
        ),
        "nonzero_tangent": tangent_norm > 0.0,
    }
    diagnostics = {
        "schema": PERIODIC_CURRENT_SHEET_NONLINEAR_BRIDGE_SCHEMA,
        "shape": list(shape),
        "equilibrium": "psi0 = A cos(2π y / Ly), omega0 = 0",
        "amplitude": amplitude,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "dt": dt,
        "steps": steps,
        "save_every": save_every,
        "saved_samples": saved_count,
        "epsilons": list(epsilon_array),
        "relative_errors": list(error_array),
        "convergence_order": convergence_order,
        "tangent_norm": tangent_norm,
        "finest_relative_error": finest_relative_error,
        "references": {
            "scope": (
                "Nonlinear reduced-MHD trajectory-map differentiability gate "
                "around a periodic current sheet."
            ),
            "equation": (
                "Centered finite differences of the nonlinear RK4 map must "
                "converge to the JAX JVP tangent at second order."
            ),
            "review_use": (
                "This is the validation layer needed before differentiable "
                "inverse design, adjoints, and neural-ODE training on solver data."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.periodic_current_sheet_nonlinear_bridge.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_convergence_order": min_convergence_order,
            "max_finest_relative_error": max_finest_relative_error,
        },
        "diagnostics": diagnostics,
    }
    return PeriodicCurrentSheetNonlinearBridgeResult(
        epsilons=epsilon_array,
        relative_errors=error_array,
        convergence_order=convergence_order,
        tangent_norm=tangent_norm,
        finest_relative_error=finest_relative_error,
        perturbation=perturbation,
        tangent_final=tangent_final,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_periodic_current_sheet_eigenvalue_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write periodic-current-sheet eigenvalue JSON, NPZ, figure, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_periodic_current_sheet_eigenvalue_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "periodic_current_sheet_eigenvalue.npz"
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
        schema=PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA,
        matrix=result.matrix,
        eigenvalues_real=result.eigenvalues.real,
        eigenvalues_imag=result.eigenvalues.imag,
        selected_eigenvalue_real=result.selected_eigenvalue.real,
        selected_eigenvalue_imag=result.selected_eigenvalue.imag,
        selected_eigenvector_real=result.selected_eigenvector.real,
        selected_eigenvector_imag=result.selected_eigenvector.imag,
        selected_residual_norm=result.selected_residual_norm,
        max_real_part=result.max_real_part,
        max_non_gauge_real_part=result.max_non_gauge_real_part,
    )

    figure_path = plot_periodic_current_sheet_spectrum(
        result.eigenvalues,
        selected_eigenvalue=result.selected_eigenvalue,
        max_allowed_real_part=float(
            result.validation["thresholds"]["max_allowed_non_gauge_real_part"]
        ),
        residual_norm=result.selected_residual_norm,
        max_residual_norm=float(result.validation["thresholds"]["max_selected_residual_norm"]),
        path=output_dir / "figures" / "periodic_current_sheet_spectrum.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "periodic_current_sheet_spectrum": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Periodic current-sheet spectrum validation.",
    )
    return manifest_path, result.validation


def write_periodic_current_sheet_nonlinear_bridge_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write nonlinear current-sheet differentiability JSON, NPZ, figure, manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_periodic_current_sheet_nonlinear_bridge_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "periodic_current_sheet_nonlinear_bridge.npz"
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
        schema=PERIODIC_CURRENT_SHEET_NONLINEAR_BRIDGE_SCHEMA,
        epsilon=result.epsilons,
        relative_error=result.relative_errors,
        convergence_order=result.convergence_order,
        tangent_norm=result.tangent_norm,
        finest_relative_error=result.finest_relative_error,
        perturbation_psi=np.asarray(result.perturbation.psi),
        perturbation_omega=np.asarray(result.perturbation.omega),
        tangent_final_psi=np.asarray(result.tangent_final.psi),
        tangent_final_omega=np.asarray(result.tangent_final.omega),
    )

    figure_path = plot_nonlinear_current_sheet_bridge(
        result.epsilons,
        result.relative_errors,
        convergence_order=result.convergence_order,
        min_convergence_order=float(result.validation["thresholds"]["min_convergence_order"]),
        max_finest_relative_error=float(
            result.validation["thresholds"]["max_finest_relative_error"]
        ),
        path=output_dir / "figures" / "periodic_current_sheet_nonlinear_bridge.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "periodic_current_sheet_nonlinear_bridge": str(
                figure_path.relative_to(output_dir)
            ),
        },
        claim_level="validation",
        claim_scope="Nonlinear current-sheet differentiability bridge validation.",
    )
    return manifest_path, result.validation


def write_periodic_current_sheet_timedomain_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write periodic-current-sheet time-domain JSON, NPZ, figure, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_periodic_current_sheet_timedomain_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "periodic_current_sheet_timedomain.npz"
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
        schema=PERIODIC_CURRENT_SHEET_TIMEDOMAIN_SCHEMA,
        time=result.times,
        amplitude=result.amplitudes,
        exact_amplitude=result.exact_amplitudes,
        relative_state_error=result.relative_state_error,
        selected_eigenvalue=result.selected_eigenvalue,
        fitted_decay_rate=result.fitted_decay_rate,
        relative_decay_rate_error=result.relative_decay_rate_error,
        selected_residual_norm=result.selected_residual_norm,
        initial_psi=np.asarray(result.initial_state.psi),
        initial_omega=np.asarray(result.initial_state.omega),
        final_psi=np.asarray(result.final_state.psi),
        final_omega=np.asarray(result.final_state.omega),
    )

    figure_path = plot_periodic_current_sheet_timedomain(
        result.times,
        result.amplitudes,
        result.exact_amplitudes,
        result.relative_state_error,
        selected_eigenvalue=result.selected_eigenvalue,
        fitted_decay_rate=result.fitted_decay_rate,
        max_relative_state_error=float(
            result.validation["thresholds"]["max_relative_state_error"]
        ),
        path=output_dir / "figures" / "periodic_current_sheet_timedomain.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "periodic_current_sheet_timedomain": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Periodic current-sheet time-domain operator replay validation.",
    )
    return manifest_path, result.validation


def _dense_matrix(operator) -> np.ndarray:
    vector_size = int(operator.shape[0])
    basis = np.eye(vector_size, dtype=np.float64)
    columns = [
        np.asarray(operator(jnp.asarray(basis[:, column])), dtype=np.float64)
        for column in range(vector_size)
    ]
    return np.column_stack(columns)


def _gauge_residuals(operator, shape: tuple[int, int]) -> dict[str, float]:
    ones = jnp.ones(shape)
    zeros = jnp.zeros(shape)
    vectors = {
        "constant_flux": flatten_reduced_mhd_state(ReducedMHDState(psi=ones, omega=zeros)),
        "constant_vorticity": flatten_reduced_mhd_state(
            ReducedMHDState(psi=zeros, omega=ones)
        ),
    }
    return {
        name: _dense_eigen_residual(
            None,
            np.asarray(vector),
            0.0,
            action=np.asarray(operator(vector)),
        )
        for name, vector in vectors.items()
    }


def _dense_eigen_residual(
    matrix: np.ndarray | None,
    vector: np.ndarray,
    eigenvalue: complex | float,
    *,
    action: np.ndarray | None = None,
) -> float:
    vector_array = np.asarray(vector)
    operator_action = action if action is not None else np.asarray(matrix @ vector_array)
    residual = operator_action - eigenvalue * vector_array
    return float(np.linalg.norm(residual.ravel()) / np.linalg.norm(vector_array.ravel()))


def _select_real_decaying_eigenpair(
    eigenvalues: np.ndarray,
    *,
    imag_tolerance: float,
    min_decay_rate: float,
) -> int:
    mask = (np.abs(eigenvalues.imag) <= imag_tolerance) & (
        eigenvalues.real <= -min_decay_rate
    )
    if not np.any(mask):
        raise RuntimeError(
            "periodic current-sheet spectrum has no real decaying eigenpair "
            f"with |Im(lambda)| <= {imag_tolerance} and Re(lambda) <= {-min_decay_rate}"
        )
    candidates = np.flatnonzero(mask)
    return int(candidates[np.argmax(eigenvalues[mask].real)])


def _rk4_linear_replay(
    matrix: np.ndarray,
    vector0: np.ndarray,
    *,
    dt: float,
    steps: int,
    save_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(vector0, dtype=np.float64).copy()
    saved_times: list[float] = []
    saved_states: list[np.ndarray] = []
    for step_index in range(steps):
        k1 = matrix @ state
        k2 = matrix @ (state + 0.5 * dt * k1)
        k3 = matrix @ (state + 0.5 * dt * k2)
        k4 = matrix @ (state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if (step_index + 1) % save_every == 0:
            saved_times.append((step_index + 1) * dt)
            saved_states.append(state.copy())
    return np.asarray(saved_times), np.vstack(saved_states)


def _normalized_bridge_perturbation(grid: CartesianGrid) -> ReducedMHDState:
    x, y = grid.mesh()
    length_x, length_y = grid.lengths
    perturbation = ReducedMHDState(
        psi=jnp.cos(2.0 * jnp.pi * x / length_x) * jnp.cos(2.0 * jnp.pi * y / length_y),
        omega=jnp.sin(2.0 * jnp.pi * x / length_x)
        * jnp.sin(2.0 * jnp.pi * y / length_y),
    )
    scale = jnp.sqrt(jnp.mean(perturbation.psi**2) + jnp.mean(perturbation.omega**2))
    return ReducedMHDState(
        psi=perturbation.psi / scale,
        omega=perturbation.omega / scale,
    )


def _array_norm(values) -> float:
    return float(jnp.linalg.norm(jnp.asarray(values).ravel()))


def _validate_timedomain_inputs(
    *,
    shape: tuple[int, int],
    dt: float,
    t_end: float,
    save_every: int,
    real_eigenvalue_imag_tolerance: float,
    min_decay_rate: float,
    max_selected_residual_norm: float,
    max_relative_state_error: float,
    max_relative_decay_rate_error: float,
) -> None:
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError("shape must contain two dimensions, each at least 4")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_end <= dt:
        raise ValueError("t_end must be greater than dt")
    steps = round(t_end / dt)
    if steps < 2:
        raise ValueError("t_end/dt must give at least two RK4 steps")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if steps // save_every < 2:
        raise ValueError("save_every must leave at least two saved samples")
    if real_eigenvalue_imag_tolerance <= 0.0:
        raise ValueError("real_eigenvalue_imag_tolerance must be positive")
    if min_decay_rate <= 0.0:
        raise ValueError("min_decay_rate must be positive")
    if max_selected_residual_norm <= 0.0:
        raise ValueError("max_selected_residual_norm must be positive")
    if max_relative_state_error <= 0.0:
        raise ValueError("max_relative_state_error must be positive")
    if max_relative_decay_rate_error <= 0.0:
        raise ValueError("max_relative_decay_rate_error must be positive")


def _validate_nonlinear_bridge_inputs(
    *,
    shape: tuple[int, int],
    dt: float,
    steps: int,
    save_every: int,
    epsilons: tuple[float, ...],
    min_convergence_order: float,
    max_finest_relative_error: float,
) -> None:
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError("shape must contain two dimensions, each at least 4")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if steps // min(save_every, steps) < 2:
        raise ValueError("save_every must leave at least two saved samples")
    if len(epsilons) < 3:
        raise ValueError("epsilons must contain at least three values")
    if any(epsilon <= 0.0 for epsilon in epsilons):
        raise ValueError("epsilons must be positive")
    if any(left <= right for left, right in zip(epsilons, epsilons[1:], strict=False)):
        raise ValueError("epsilons must be strictly decreasing")
    if min_convergence_order <= 0.0:
        raise ValueError("min_convergence_order must be positive")
    if max_finest_relative_error <= 0.0:
        raise ValueError("max_finest_relative_error must be positive")
