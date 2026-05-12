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
from mhx.diagnostics import kinetic_energy, magnetic_energy
from mhx.equations.reduced_mhd import (
    current_density,
    linearized_reduced_mhd_operator,
    reduced_mhd_rhs,
)
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.physics import CosineTearingEquilibrium, PeriodicDoubleHarrisEquilibrium
from mhx.plotting import (
    plot_current_density_gif,
    plot_double_harris_nonlinear_growth,
    plot_double_harris_seeded_long_run,
    plot_flux_gif,
    plot_nonlinear_current_sheet_bridge,
    plot_periodic_current_sheet_spectrum,
    plot_periodic_current_sheet_timedomain,
)
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    ReducedMHDTrajectory,
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
PERIODIC_DOUBLE_HARRIS_NONLINEAR_GROWTH_SCHEMA = (
    "mhx.validation.periodic_double_harris_nonlinear_growth.v1"
)
PERIODIC_DOUBLE_HARRIS_SEEDED_LONG_RUN_SCHEMA = (
    "mhx.validation.periodic_double_harris_seeded_long_run.v1"
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


@dataclass(frozen=True)
class PeriodicDoubleHarrisNonlinearGrowthResult:
    """Nonlinear replay of an unstable periodic double-Harris eigenmode."""

    matrix: np.ndarray
    eigenvalues: np.ndarray
    selected_eigenvalue: complex
    selected_eigenvector: np.ndarray
    selected_residual_norm: float
    time: np.ndarray
    perturbation_norm: np.ndarray
    expected_perturbation_norm: np.ndarray
    fitted_growth_rate: float
    relative_growth_error: float
    growth_factor: float
    base_initial_state: ReducedMHDState
    base_final_state: ReducedMHDState
    perturbed_final_state: ReducedMHDState
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class PeriodicDoubleHarrisSeededLongRunResult:
    """Resolution-oriented nonlinear replay of a seeded periodic double-Harris sheet."""

    time: np.ndarray
    perturbation_norm: np.ndarray
    magnetic_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    current_density_linf: np.ndarray
    fitted_early_growth_rate: float
    early_growth_factor: float
    max_growth_factor: float
    base_trajectory: ReducedMHDTrajectory
    perturbed_trajectory: ReducedMHDTrajectory
    base_initial_state: ReducedMHDState
    perturbed_initial_state: ReducedMHDState
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


def run_periodic_double_harris_nonlinear_growth_validation(
    *,
    shape: tuple[int, int] = (8, 8),
    width: float = 0.4,
    amplitude: float = 1.0,
    resistivity: float = 5.0e-3,
    viscosity: float = 5.0e-3,
    perturbation_amplitude: float = 1.0e-6,
    dt: float = 1.0e-2,
    t_end: float = 4.0,
    save_every: int = 20,
    gauge_eigenvalue_radius: float = 1.0e-9,
    min_linear_growth_rate: float = 5.0e-2,
    min_nonlinear_growth_factor: float = 2.0,
    max_relative_growth_error: float = 1.5e-1,
    max_selected_residual_norm: float = 1.0e-9,
    max_selected_eigenvalue_imag: float = 1.0e-8,
) -> PeriodicDoubleHarrisNonlinearGrowthResult:
    r"""Validate nonlinear growth from an unstable double-Harris eigenmode.

    The gate closes the gap left by the stable cosine-current-sheet checks. It
    builds a periodic double-Harris current sheet, assembles the dense
    matrix-free reduced-MHD Jacobian on a tiny grid, selects the most unstable
    real eigenmode, then advances two full nonlinear trajectories: the base
    sheet and the base sheet plus a small eigenmode perturbation.  The measured
    difference norm must grow exponentially at a rate close to the frozen-base
    eigenvalue.
    """
    _validate_double_harris_growth_inputs(
        shape=shape,
        width=width,
        amplitude=amplitude,
        resistivity=resistivity,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        min_linear_growth_rate=min_linear_growth_rate,
        min_nonlinear_growth_factor=min_nonlinear_growth_factor,
        max_relative_growth_error=max_relative_growth_error,
        max_selected_residual_norm=max_selected_residual_norm,
        max_selected_eigenvalue_imag=max_selected_eigenvalue_imag,
    )
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    base = PeriodicDoubleHarrisEquilibrium(
        width=width,
        amplitude=amplitude,
    ).initial_state(grid)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)
    operator = linearized_reduced_mhd_operator(base, params, lengths=grid.lengths)
    matrix = _dense_matrix(operator)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    selected_index = _select_unstable_eigenpair(
        eigenvalues,
        gauge_eigenvalue_radius=gauge_eigenvalue_radius,
        min_growth_rate=min_linear_growth_rate,
    )
    selected_eigenvalue = complex(eigenvalues[selected_index])
    selected_eigenvector = np.asarray(eigenvectors[:, selected_index].real, dtype=np.float64)
    selected_eigenvector /= np.linalg.norm(selected_eigenvector.ravel())
    selected_residual_norm = _dense_eigen_residual(
        matrix,
        selected_eigenvector,
        selected_eigenvalue,
    )
    perturbation = unflatten_reduced_mhd_state(jnp.asarray(selected_eigenvector), shape)
    perturbed_initial = ReducedMHDState(
        psi=base.psi + perturbation_amplitude * perturbation.psi,
        omega=base.omega + perturbation_amplitude * perturbation.omega,
    )

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    steps = round(t_end / dt)
    base_trajectory = evolve_rk4(
        base,
        rhs,
        dt=dt,
        steps=steps,
        save_every=save_every,
    )
    perturbed_trajectory = evolve_rk4(
        perturbed_initial,
        rhs,
        dt=dt,
        steps=steps,
        save_every=save_every,
    )
    time, perturbation_norm = _trajectory_difference_norms(
        base,
        base_trajectory.times,
        base_trajectory.states,
        perturbed_initial,
        perturbed_trajectory.states,
        perturbation_amplitude=perturbation_amplitude,
    )
    fitted_growth_rate = float(np.polyfit(time, np.log(perturbation_norm), 1)[0])
    expected_growth = perturbation_norm[0] * np.exp(selected_eigenvalue.real * time)
    relative_growth_error = abs(
        (fitted_growth_rate - selected_eigenvalue.real) / selected_eigenvalue.real
    )
    growth_factor = float(perturbation_norm[-1] / perturbation_norm[0])
    base_final = ReducedMHDState(
        psi=base_trajectory.states.psi[-1],
        omega=base_trajectory.states.omega[-1],
    )
    perturbed_final = ReducedMHDState(
        psi=perturbed_trajectory.states.psi[-1],
        omega=perturbed_trajectory.states.omega[-1],
    )
    checks = {
        "finite_spectrum_and_history": bool(
            np.isfinite(eigenvalues.real).all()
            and np.isfinite(eigenvalues.imag).all()
            and np.isfinite(perturbation_norm).all()
        ),
        "unstable_linear_mode_found": selected_eigenvalue.real >= min_linear_growth_rate,
        "selected_eigenvalue_is_real": abs(selected_eigenvalue.imag)
        <= max_selected_eigenvalue_imag,
        "selected_dense_eigenpair_residual_small": (
            selected_residual_norm <= max_selected_residual_norm
        ),
        "nonlinear_difference_grows": growth_factor >= min_nonlinear_growth_factor,
        "fitted_growth_rate_positive": fitted_growth_rate > 0.0,
        "fitted_growth_matches_frozen_linear_mode": (
            relative_growth_error <= max_relative_growth_error
        ),
    }
    diagnostics = {
        "schema": PERIODIC_DOUBLE_HARRIS_NONLINEAR_GROWTH_SCHEMA,
        "shape": list(shape),
        "equilibrium": (
            "periodic double Harris: By=A[tanh((x-Lx/4)/a)-"
            "tanh((x-3Lx/4)/a)-1]"
        ),
        "width": width,
        "amplitude": amplitude,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "perturbation_amplitude": perturbation_amplitude,
        "dt": dt,
        "t_end": t_end,
        "save_every": save_every,
        "steps": steps,
        "samples": int(time.size),
        "selected_eigenvalue": {
            "real": float(selected_eigenvalue.real),
            "imag": float(selected_eigenvalue.imag),
        },
        "selected_residual_norm": selected_residual_norm,
        "fitted_growth_rate": fitted_growth_rate,
        "relative_growth_error": float(relative_growth_error),
        "growth_factor": growth_factor,
        "initial_perturbation_norm": float(perturbation_norm[0]),
        "final_perturbation_norm": float(perturbation_norm[-1]),
        "references": {
            "scope": (
                "Nonlinear periodic reduced-MHD growth gate for an unstable "
                "double-Harris current sheet. This is a small-grid validation "
                "of the instability path, not a converged plasmoid/Rutherford "
                "production result."
            ),
            "literature": (
                "Harris-sheet tearing follows the FKR/Coppi current-sheet "
                "instability picture; long thin Sweet-Parker sheets transition "
                "to the Loureiro-Schekochihin-Cowley plasmoid regime."
            ),
            "claim_boundary": (
                "The gate demonstrates positive nonlinear growth from a dense "
                "unstable eigenmode. Publication claims still require resolution, "
                "duration, seed, and aspect-ratio sweeps."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.periodic_double_harris_nonlinear_growth.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "gauge_eigenvalue_radius": gauge_eigenvalue_radius,
            "min_linear_growth_rate": min_linear_growth_rate,
            "min_nonlinear_growth_factor": min_nonlinear_growth_factor,
            "max_relative_growth_error": max_relative_growth_error,
            "max_selected_residual_norm": max_selected_residual_norm,
            "max_selected_eigenvalue_imag": max_selected_eigenvalue_imag,
        },
        "diagnostics": diagnostics,
    }
    return PeriodicDoubleHarrisNonlinearGrowthResult(
        matrix=matrix,
        eigenvalues=eigenvalues,
        selected_eigenvalue=selected_eigenvalue,
        selected_eigenvector=selected_eigenvector,
        selected_residual_norm=selected_residual_norm,
        time=time,
        perturbation_norm=perturbation_norm,
        expected_perturbation_norm=expected_growth,
        fitted_growth_rate=fitted_growth_rate,
        relative_growth_error=float(relative_growth_error),
        growth_factor=growth_factor,
        base_initial_state=base,
        base_final_state=base_final,
        perturbed_final_state=perturbed_final,
        diagnostics=diagnostics,
        validation=validation,
    )


def run_periodic_double_harris_seeded_long_run_validation(
    *,
    shape: tuple[int, int] = (64, 64),
    width: float = 0.4,
    amplitude: float = 1.0,
    resistivity: float = 5.0e-3,
    viscosity: float = 5.0e-3,
    perturbation_amplitude: float = 1.0e-3,
    perturbation_mode: tuple[int, int] = (2, 1),
    dt: float = 1.0e-2,
    t_end: float = 30.0,
    save_every: int = 100,
    fit_window: tuple[float, float] = (0.0, 10.0),
    min_saved_samples: int = 8,
    min_early_growth_rate: float = 5.0e-2,
    min_early_growth_factor: float = 1.5,
    min_max_growth_factor: float = 2.0,
    max_relative_energy_increase: float = 1.0e-8,
) -> PeriodicDoubleHarrisSeededLongRunResult:
    r"""Run a longer seeded periodic double-Harris nonlinear replay.

    Unlike :func:`run_periodic_double_harris_nonlinear_growth_validation`, this
    benchmark does not assemble a dense Jacobian and does not require the whole
    trajectory to follow a frozen linear eigenvalue. It is a scalable nonlinear
    evidence run: the base sheet and a seeded sheet are advanced at the requested
    resolution, an early-time exponential window is fitted, and the full saved
    trajectory is checked for finite fields, dissipative total-energy behavior,
    and sustained resolved current-sheet diagnostics.
    """
    _validate_double_harris_seeded_long_run_inputs(
        shape=shape,
        width=width,
        amplitude=amplitude,
        resistivity=resistivity,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_mode=perturbation_mode,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        fit_window=fit_window,
        min_saved_samples=min_saved_samples,
        min_early_growth_rate=min_early_growth_rate,
        min_early_growth_factor=min_early_growth_factor,
        min_max_growth_factor=min_max_growth_factor,
        max_relative_energy_increase=max_relative_energy_increase,
    )
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    base_initial = PeriodicDoubleHarrisEquilibrium(
        width=width,
        amplitude=amplitude,
    ).initial_state(grid)
    perturbed_initial = PeriodicDoubleHarrisEquilibrium(
        width=width,
        amplitude=amplitude,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_mode=perturbation_mode,
    ).initial_state(grid)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    steps = round(t_end / dt)
    base_trajectory = evolve_rk4(
        base_initial,
        rhs,
        dt=dt,
        steps=steps,
        save_every=save_every,
    )
    perturbed_trajectory = evolve_rk4(
        perturbed_initial,
        rhs,
        dt=dt,
        steps=steps,
        save_every=save_every,
    )
    time, perturbation_norm = _trajectory_difference_norms(
        base_initial,
        base_trajectory.times,
        base_trajectory.states,
        perturbed_initial,
        perturbed_trajectory.states,
        perturbation_amplitude=perturbation_amplitude,
    )
    energies = _trajectory_energy_and_current_histories(
        perturbed_initial,
        perturbed_trajectory.states,
        lengths=grid.lengths,
    )
    fit_mask = (time >= fit_window[0]) & (time <= fit_window[1])
    fit_time = time[fit_mask]
    fit_norm = perturbation_norm[fit_mask]
    fitted_early_growth_rate = float(np.polyfit(fit_time, np.log(fit_norm), 1)[0])
    early_growth_factor = float(fit_norm[-1] / fit_norm[0])
    max_growth_factor = float(np.max(perturbation_norm) / perturbation_norm[0])
    initial_energy = float(energies["total"][0])
    relative_energy_increase = float(
        max(0.0, np.max(energies["total"]) - initial_energy) / max(initial_energy, 1.0e-300)
    )
    checks = {
        "finite_histories": bool(
            np.isfinite(perturbation_norm).all()
            and np.isfinite(energies["magnetic"]).all()
            and np.isfinite(energies["kinetic"]).all()
            and np.isfinite(energies["total"]).all()
            and np.isfinite(energies["current_density_linf"]).all()
        ),
        "full_duration_reached": bool(time[-1] >= t_end - 0.5 * dt),
        "enough_saved_samples": bool(time.size >= min_saved_samples),
        "early_growth_positive": fitted_early_growth_rate >= min_early_growth_rate,
        "early_difference_grows": early_growth_factor >= min_early_growth_factor,
        "nonlinear_difference_reaches_visible_growth": max_growth_factor >= min_max_growth_factor,
        "total_energy_is_dissipative": relative_energy_increase <= max_relative_energy_increase,
    }
    diagnostics = {
        "schema": PERIODIC_DOUBLE_HARRIS_SEEDED_LONG_RUN_SCHEMA,
        "shape": list(shape),
        "equilibrium": "periodic_double_harris",
        "width": width,
        "amplitude": amplitude,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "perturbation_amplitude": perturbation_amplitude,
        "perturbation_mode": list(perturbation_mode),
        "dt": dt,
        "t_end": t_end,
        "save_every": save_every,
        "steps": steps,
        "samples": int(time.size),
        "fit_window": list(fit_window),
        "fitted_early_growth_rate": fitted_early_growth_rate,
        "early_growth_factor": early_growth_factor,
        "max_growth_factor": max_growth_factor,
        "initial_total_energy": initial_energy,
        "final_total_energy": float(energies["total"][-1]),
        "relative_energy_increase": relative_energy_increase,
        "max_kinetic_energy": float(np.max(energies["kinetic"])),
        "max_current_density_linf": float(np.max(energies["current_density_linf"])),
        "references": {
            "scope": (
                "Longer seeded nonlinear reduced-MHD replay at scalable grid size. "
                "This is stronger than a dense tiny-grid eigenmode replay, but it is "
                "still a validation artifact until duration, resolution, seed, and "
                "aspect-ratio sweeps are completed."
            ),
            "literature": (
                "Anchored to Harris-sheet tearing and plasmoid-current-sheet "
                "validation practice; the command supplies histories and movies "
                "needed before FKR/Coppi or Sweet-Parker/plasmoid claims."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.periodic_double_harris_seeded_long_run.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_saved_samples": min_saved_samples,
            "min_early_growth_rate": min_early_growth_rate,
            "min_early_growth_factor": min_early_growth_factor,
            "min_max_growth_factor": min_max_growth_factor,
            "max_relative_energy_increase": max_relative_energy_increase,
        },
        "diagnostics": diagnostics,
    }
    return PeriodicDoubleHarrisSeededLongRunResult(
        time=time,
        perturbation_norm=perturbation_norm,
        magnetic_energy=energies["magnetic"],
        kinetic_energy=energies["kinetic"],
        total_energy=energies["total"],
        current_density_linf=energies["current_density_linf"],
        fitted_early_growth_rate=fitted_early_growth_rate,
        early_growth_factor=early_growth_factor,
        max_growth_factor=max_growth_factor,
        base_trajectory=base_trajectory,
        perturbed_trajectory=perturbed_trajectory,
        base_initial_state=base_initial,
        perturbed_initial_state=perturbed_initial,
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


def write_periodic_double_harris_nonlinear_growth_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write double-Harris nonlinear-growth JSON, NPZ, figure, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_periodic_double_harris_nonlinear_growth_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "periodic_double_harris_nonlinear_growth.npz"
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
        schema=PERIODIC_DOUBLE_HARRIS_NONLINEAR_GROWTH_SCHEMA,
        eigenvalues_real=result.eigenvalues.real,
        eigenvalues_imag=result.eigenvalues.imag,
        selected_eigenvalue_real=result.selected_eigenvalue.real,
        selected_eigenvalue_imag=result.selected_eigenvalue.imag,
        selected_eigenvector=result.selected_eigenvector,
        selected_residual_norm=result.selected_residual_norm,
        time=result.time,
        perturbation_norm=result.perturbation_norm,
        expected_perturbation_norm=result.expected_perturbation_norm,
        fitted_growth_rate=result.fitted_growth_rate,
        relative_growth_error=result.relative_growth_error,
        growth_factor=result.growth_factor,
        base_initial_psi=np.asarray(result.base_initial_state.psi),
        base_final_psi=np.asarray(result.base_final_state.psi),
        perturbed_final_psi=np.asarray(result.perturbed_final_state.psi),
        base_final_omega=np.asarray(result.base_final_state.omega),
        perturbed_final_omega=np.asarray(result.perturbed_final_state.omega),
    )
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=tuple(result.diagnostics["shape"])))
    x, y = grid.axes()
    figure_path = plot_double_harris_nonlinear_growth(
        result.time,
        result.perturbation_norm,
        result.expected_perturbation_norm,
        selected_eigenvalue=result.selected_eigenvalue.real,
        fitted_growth_rate=result.fitted_growth_rate,
        relative_growth_error=result.relative_growth_error,
        base_initial_psi=result.base_initial_state.psi,
        base_final_psi=result.base_final_state.psi,
        perturbed_final_psi=result.perturbed_final_state.psi,
        x=x,
        y=y,
        path=output_dir / "figures" / "periodic_double_harris_nonlinear_growth.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "periodic_double_harris_nonlinear_growth": str(
                figure_path.relative_to(output_dir)
            ),
        },
        claim_level="validation",
        claim_scope=(
            "Small-grid nonlinear growth validation for an unstable periodic "
            "double-Harris current sheet; not a production plasmoid/Rutherford claim."
        ),
    )
    return manifest_path, result.validation


def write_periodic_double_harris_seeded_long_run_validation(
    outdir: str | Path,
    *,
    movies: bool = False,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write seeded double-Harris long-run JSON, NPZ, figures, GIFs, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_periodic_double_harris_seeded_long_run_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "periodic_double_harris_seeded_long_run.npz"
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
        schema=PERIODIC_DOUBLE_HARRIS_SEEDED_LONG_RUN_SCHEMA,
        time=result.time,
        perturbation_norm=result.perturbation_norm,
        magnetic_energy=result.magnetic_energy,
        kinetic_energy=result.kinetic_energy,
        total_energy=result.total_energy,
        current_density_linf=result.current_density_linf,
        fitted_early_growth_rate=result.fitted_early_growth_rate,
        early_growth_factor=result.early_growth_factor,
        max_growth_factor=result.max_growth_factor,
        base_time=np.asarray(result.base_trajectory.times),
        perturbed_time=np.asarray(result.perturbed_trajectory.times),
        base_psi=np.asarray(result.base_trajectory.states.psi),
        base_omega=np.asarray(result.base_trajectory.states.omega),
        perturbed_psi=np.asarray(result.perturbed_trajectory.states.psi),
        perturbed_omega=np.asarray(result.perturbed_trajectory.states.omega),
        base_initial_psi=np.asarray(result.base_initial_state.psi),
        base_initial_omega=np.asarray(result.base_initial_state.omega),
        perturbed_initial_psi=np.asarray(result.perturbed_initial_state.psi),
        perturbed_initial_omega=np.asarray(result.perturbed_initial_state.omega),
    )
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=tuple(result.diagnostics["shape"])))
    figure_path = plot_double_harris_seeded_long_run(
        result.time,
        result.perturbation_norm,
        result.magnetic_energy,
        result.kinetic_energy,
        result.total_energy,
        result.current_density_linf,
        result.perturbed_initial_state.psi,
        result.perturbed_trajectory.states.psi[-1],
        result.base_trajectory.states.psi[-1],
        fitted_early_growth_rate=result.fitted_early_growth_rate,
        fit_window=tuple(result.diagnostics["fit_window"]),
        path=output_dir / "figures" / "periodic_double_harris_seeded_long_run.png",
    )
    outputs: dict[str, str] = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "history": history_path.name,
        "periodic_double_harris_seeded_long_run": str(
            figure_path.relative_to(output_dir)
        ),
    }
    if movies:
        extent = (0.0, grid.lengths[0], 0.0, grid.lengths[1])
        flux_path = plot_flux_gif(
            result.perturbed_trajectory,
            path=output_dir / "figures" / "periodic_double_harris_flux.gif",
            extent=extent,
        )
        current_path = plot_current_density_gif(
            result.perturbed_trajectory,
            path=output_dir / "figures" / "periodic_double_harris_current.gif",
            extent=extent,
            lengths=grid.lengths,
        )
        outputs["periodic_double_harris_flux_movie"] = str(
            flux_path.relative_to(output_dir)
        )
        outputs["periodic_double_harris_current_movie"] = str(
            current_path.relative_to(output_dir)
        )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope=(
            "Seeded longer nonlinear periodic double-Harris replay with finite, "
            "dissipative histories; not yet a converged plasmoid/Rutherford claim."
        ),
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


def _select_unstable_eigenpair(
    eigenvalues: np.ndarray,
    *,
    gauge_eigenvalue_radius: float,
    min_growth_rate: float,
) -> int:
    mask = (np.abs(eigenvalues) > gauge_eigenvalue_radius) & (
        eigenvalues.real >= min_growth_rate
    )
    if not np.any(mask):
        raise RuntimeError(
            "periodic double-Harris spectrum has no unstable non-gauge mode "
            f"with Re(lambda) >= {min_growth_rate}"
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


def _trajectory_difference_norms(
    base_initial: ReducedMHDState,
    saved_times,
    base_states: ReducedMHDState,
    perturbed_initial: ReducedMHDState,
    perturbed_states: ReducedMHDState,
    *,
    perturbation_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    states_base = (base_initial,) + tuple(
        ReducedMHDState(psi=base_states.psi[index], omega=base_states.omega[index])
        for index in range(base_states.psi.shape[0])
    )
    states_perturbed = (perturbed_initial,) + tuple(
        ReducedMHDState(
            psi=perturbed_states.psi[index],
            omega=perturbed_states.omega[index],
        )
        for index in range(perturbed_states.psi.shape[0])
    )
    times = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.asarray(saved_times, dtype=np.float64),
        )
    )
    amplitudes = []
    for base_state, perturbed_state in zip(states_base, states_perturbed, strict=True):
        difference = ReducedMHDState(
            psi=perturbed_state.psi - base_state.psi,
            omega=perturbed_state.omega - base_state.omega,
        )
        amplitudes.append(
            _array_norm(flatten_reduced_mhd_state(difference)) / perturbation_amplitude
        )
    return times, np.asarray(amplitudes, dtype=np.float64)


def _trajectory_energy_and_current_histories(
    initial_state: ReducedMHDState,
    saved_states: ReducedMHDState,
    *,
    lengths: tuple[float, float],
) -> dict[str, np.ndarray]:
    states = (initial_state,) + tuple(
        ReducedMHDState(psi=saved_states.psi[index], omega=saved_states.omega[index])
        for index in range(saved_states.psi.shape[0])
    )
    magnetic = []
    kinetic = []
    current_linf = []
    for state in states:
        magnetic.append(float(magnetic_energy(state, lengths=lengths)))
        kinetic.append(float(kinetic_energy(state, lengths=lengths)))
        current_linf.append(
            float(jnp.max(jnp.abs(current_density(state.psi, lengths=lengths))))
        )
    magnetic_array = np.asarray(magnetic, dtype=np.float64)
    kinetic_array = np.asarray(kinetic, dtype=np.float64)
    return {
        "magnetic": magnetic_array,
        "kinetic": kinetic_array,
        "total": magnetic_array + kinetic_array,
        "current_density_linf": np.asarray(current_linf, dtype=np.float64),
    }


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


def _validate_double_harris_growth_inputs(
    *,
    shape: tuple[int, int],
    width: float,
    amplitude: float,
    resistivity: float,
    viscosity: float,
    perturbation_amplitude: float,
    dt: float,
    t_end: float,
    save_every: int,
    min_linear_growth_rate: float,
    min_nonlinear_growth_factor: float,
    max_relative_growth_error: float,
    max_selected_residual_norm: float,
    max_selected_eigenvalue_imag: float,
) -> None:
    if len(shape) != 2 or min(shape) < 6:
        raise ValueError("shape must contain two dimensions, each at least 6")
    if width <= 0.0:
        raise ValueError("width must be positive")
    if amplitude == 0.0:
        raise ValueError("amplitude must be nonzero")
    if resistivity <= 0.0:
        raise ValueError("resistivity must be positive")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    if perturbation_amplitude <= 0.0:
        raise ValueError("perturbation_amplitude must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_end <= dt:
        raise ValueError("t_end must be greater than dt")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if round(t_end / dt) // save_every < 4:
        raise ValueError("save_every must leave at least four saved samples")
    if min_linear_growth_rate <= 0.0:
        raise ValueError("min_linear_growth_rate must be positive")
    if min_nonlinear_growth_factor <= 1.0:
        raise ValueError("min_nonlinear_growth_factor must be greater than one")
    if max_relative_growth_error <= 0.0:
        raise ValueError("max_relative_growth_error must be positive")
    if max_selected_residual_norm <= 0.0:
        raise ValueError("max_selected_residual_norm must be positive")
    if max_selected_eigenvalue_imag <= 0.0:
        raise ValueError("max_selected_eigenvalue_imag must be positive")


def _validate_double_harris_seeded_long_run_inputs(
    *,
    shape: tuple[int, int],
    width: float,
    amplitude: float,
    resistivity: float,
    viscosity: float,
    perturbation_amplitude: float,
    perturbation_mode: tuple[int, int],
    dt: float,
    t_end: float,
    save_every: int,
    fit_window: tuple[float, float],
    min_saved_samples: int,
    min_early_growth_rate: float,
    min_early_growth_factor: float,
    min_max_growth_factor: float,
    max_relative_energy_increase: float,
) -> None:
    if len(shape) != 2 or shape[0] < 8 or shape[1] < 8:
        raise ValueError("shape must contain two dimensions >= 8")
    if width <= 0.0:
        raise ValueError("width must be positive")
    if amplitude == 0.0:
        raise ValueError("amplitude must be nonzero")
    if resistivity < 0.0 or viscosity < 0.0:
        raise ValueError("resistivity and viscosity must be non-negative")
    if perturbation_amplitude <= 0.0:
        raise ValueError("perturbation_amplitude must be positive")
    if len(perturbation_mode) != 2 or perturbation_mode == (0, 0):
        raise ValueError("perturbation_mode must be a nonzero two-index mode")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    steps = round(t_end / dt)
    if steps < 2:
        raise ValueError("t_end / dt must allow at least two RK4 steps")
    if save_every <= 0 or save_every > steps:
        raise ValueError("save_every must be positive and no larger than steps")
    fit_start, fit_stop = fit_window
    if fit_start < 0.0 or fit_stop <= fit_start:
        raise ValueError("fit_window must be ordered and non-negative")
    if fit_stop > t_end:
        raise ValueError("fit_window stop must not exceed t_end")
    saved_fit_count = 1 + sum(
        1
        for step_index in range(steps)
        if (step_index + 1) % save_every == 0
        and fit_start <= (step_index + 1) * dt <= fit_stop
    )
    if saved_fit_count < 3:
        raise ValueError("fit_window must include at least three saved samples")
    if min_saved_samples < 3:
        raise ValueError("min_saved_samples must be at least three")
    if min_early_growth_rate <= 0.0:
        raise ValueError("min_early_growth_rate must be positive")
    if min_early_growth_factor <= 1.0:
        raise ValueError("min_early_growth_factor must exceed one")
    if min_max_growth_factor <= 1.0:
        raise ValueError("min_max_growth_factor must exceed one")
    if max_relative_energy_increase < 0.0:
        raise ValueError("max_relative_energy_increase must be non-negative")
