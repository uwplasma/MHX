"""Matrix-free eigenvalue and linearized-RHS validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.benchmarks.linear_stability._report import _l2_norm, _relative_l2_error
from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import (
    finite_difference_linearized_reduced_mhd_rhs,
    linearized_reduced_mhd_operator,
    linearized_reduced_mhd_rhs,
)
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics import (
    MatrixFreeOperator,
    arnoldi_iteration,
    eigen_residual_norm,
    power_iteration,
    rayleigh_quotient,
)
from mhx.numerics.spectral import laplacian
from mhx.physics import CosineTearingEquilibrium
from mhx.plotting import (
    plot_arnoldi_ritz_values,
    plot_cosine_equilibrium_linearization_errors,
    plot_diffusion_eigenvalue_error,
    plot_linearized_rhs_errors,
    plot_power_iteration_history,
    plot_reduced_mhd_eigenmode_errors,
)
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    flatten_reduced_mhd_state,
)

DIFFUSION_EIGENVALUE_SCHEMA = "mhx.validation.diffusion_eigenvalue.v1"
POWER_ITERATION_SCHEMA = "mhx.validation.power_iteration.v1"
ARNOLDI_SCHEMA = "mhx.validation.arnoldi.v1"
LINEARIZED_RHS_SCHEMA = "mhx.validation.linearized_rhs.v1"
REDUCED_MHD_LINEAR_EIGENMODE_SCHEMA = "mhx.validation.reduced_mhd_linear_eigenmode.v1"
COSINE_EQUILIBRIUM_LINEARIZATION_SCHEMA = (
    "mhx.validation.cosine_equilibrium_linearization.v1"
)


@dataclass(frozen=True)
class DiffusionEigenvalueResult:
    """Analytic diffusion eigenvalue result and validation gates."""

    eigenfunction: np.ndarray
    operator_action: np.ndarray
    expected_eigenvalue: float
    measured_eigenvalue: float
    eigenvalue_abs_error: float
    residual_norm: float
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class PowerIterationValidationResult:
    """Known-operator power-iteration result and validation gates."""

    expected_eigenvalue: float
    measured_eigenvalue: float
    eigenvalue_abs_error: float
    residual_norm: float
    rayleigh_history: np.ndarray
    residual_history: np.ndarray
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class ArnoldiValidationResult:
    """Known-operator Arnoldi result and validation gates."""

    expected_eigenvalues: np.ndarray
    ritz_values: np.ndarray
    max_ritz_abs_error: float
    max_imag_abs: float
    max_residual_estimate: float
    residual_estimates: np.ndarray
    hessenberg: np.ndarray
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class LinearizedRHSResult:
    """JVP/finite-difference consistency diagnostics for the reduced-MHD RHS."""

    jvp: ReducedMHDState
    finite_difference: ReducedMHDState
    absolute_errors: dict[str, float]
    relative_errors: dict[str, float]
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class ReducedMHDLinearEigenmodeResult:
    """Zero-state reduced-MHD linear eigenmode diagnostics and gates."""

    psi_eigenfunction: np.ndarray
    omega_eigenfunction: np.ndarray
    operator_psi_action: np.ndarray
    operator_omega_action: np.ndarray
    expected_eigenvalues: dict[str, float]
    measured_eigenvalues: dict[str, float]
    eigenvalue_abs_errors: dict[str, float]
    residual_norms: dict[str, float]
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class CosineEquilibriumLinearizationResult:
    """Analytic nonzero-equilibrium linearized-RHS coupling diagnostics."""

    flow_tangent: ReducedMHDState
    expected_flow_tangent: ReducedMHDState
    tension_tangent: ReducedMHDState
    expected_tension_tangent: ReducedMHDState
    relative_errors: dict[str, float]
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_diffusion_eigenvalue_validation(
    *,
    shape: tuple[int, int] = (32, 32),
    mode: tuple[int, int] = (2, 1),
    diffusivity: float = 2.5e-2,
    max_eigenvalue_abs_error: float = 1.0e-6,
    max_residual_norm: float = 5.0e-6,
) -> DiffusionEigenvalueResult:
    """Validate a matrix-free spectral diffusion eigenpair against theory."""
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    eigenfunction = grid.sinusoid(mode=mode)
    kx = 2.0 * np.pi * mode[0] / grid.lengths[0]
    ky = 2.0 * np.pi * mode[1] / grid.lengths[1]
    expected_eigenvalue = -diffusivity * (kx**2 + ky**2)
    operator = MatrixFreeOperator(
        shape=shape,
        name="spectral_diffusion",
        matvec=lambda vector: diffusivity * laplacian(vector, lengths=grid.lengths),
    )
    operator_action = operator(eigenfunction)
    measured_eigenvalue = float(rayleigh_quotient(operator, eigenfunction))
    eigenvalue_abs_error = abs(measured_eigenvalue - expected_eigenvalue)
    residual_norm = float(eigen_residual_norm(operator, eigenfunction, expected_eigenvalue))
    checks = {
        "rayleigh_quotient_matches_analytic_eigenvalue": (
            eigenvalue_abs_error <= max_eigenvalue_abs_error
        ),
        "eigen_residual_within_tolerance": residual_norm <= max_residual_norm,
    }
    diagnostics = {
        "schema": DIFFUSION_EIGENVALUE_SCHEMA,
        "shape": list(shape),
        "mode": list(mode),
        "diffusivity": diffusivity,
        "expected_eigenvalue": expected_eigenvalue,
        "measured_eigenvalue": measured_eigenvalue,
        "eigenvalue_abs_error": eigenvalue_abs_error,
        "residual_norm": residual_norm,
        "references": {
            "spectral_laplacian": "Fourier mode eigenvalue of the periodic Laplacian",
            "eigen_scaffold": "Matrix-free Rayleigh quotient and residual for future tearing modes",
        },
    }
    validation = {
        "schema": "mhx.validation.diffusion_eigenvalue.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_eigenvalue_abs_error": max_eigenvalue_abs_error,
            "max_residual_norm": max_residual_norm,
        },
        "diagnostics": diagnostics,
    }
    return DiffusionEigenvalueResult(
        eigenfunction=np.asarray(eigenfunction),
        operator_action=np.asarray(operator_action),
        expected_eigenvalue=expected_eigenvalue,
        measured_eigenvalue=measured_eigenvalue,
        eigenvalue_abs_error=eigenvalue_abs_error,
        residual_norm=residual_norm,
        diagnostics=diagnostics,
        validation=validation,
    )


def run_power_iteration_validation(
    *,
    iterations: int = 30,
    max_eigenvalue_abs_error: float = 1.0e-6,
    max_residual_norm: float = 1.0e-6,
) -> PowerIterationValidationResult:
    """Validate power iteration on a known diagonal matrix-free operator."""
    eigenvalues = jnp.asarray([3.0, -1.5, 0.5, 0.1])
    expected_eigenvalue = float(eigenvalues[0])
    operator = MatrixFreeOperator(
        shape=eigenvalues.shape,
        name="diagonal_power_iteration_fixture",
        matvec=lambda vector: eigenvalues * vector,
    )
    initial_vector = jnp.asarray([1.0, 0.5, -0.25, 0.125])
    result = power_iteration(operator, initial_vector, iterations=iterations)
    measured_eigenvalue = float(result.eigenvalue)
    residual_norm = float(result.residual_norm)
    eigenvalue_abs_error = abs(measured_eigenvalue - expected_eigenvalue)
    checks = {
        "dominant_rayleigh_quotient_matches_fixture": (
            eigenvalue_abs_error <= max_eigenvalue_abs_error
        ),
        "dominant_eigen_residual_within_tolerance": residual_norm <= max_residual_norm,
    }
    diagnostics = {
        "schema": POWER_ITERATION_SCHEMA,
        "iterations": iterations,
        "expected_eigenvalue": expected_eigenvalue,
        "measured_eigenvalue": measured_eigenvalue,
        "eigenvalue_abs_error": eigenvalue_abs_error,
        "residual_norm": residual_norm,
        "fixture_eigenvalues": [float(value) for value in eigenvalues],
        "references": {
            "power_iteration": "Dominant-eigenpair smoke test for matrix-free operators",
        },
    }
    validation = {
        "schema": "mhx.validation.power_iteration.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_eigenvalue_abs_error": max_eigenvalue_abs_error,
            "max_residual_norm": max_residual_norm,
        },
        "diagnostics": diagnostics,
    }
    return PowerIterationValidationResult(
        expected_eigenvalue=expected_eigenvalue,
        measured_eigenvalue=measured_eigenvalue,
        eigenvalue_abs_error=eigenvalue_abs_error,
        residual_norm=residual_norm,
        rayleigh_history=np.asarray(result.rayleigh_history),
        residual_history=np.asarray(result.residual_history),
        diagnostics=diagnostics,
        validation=validation,
    )


def run_arnoldi_validation(
    *,
    krylov_dim: int = 4,
    max_ritz_abs_error: float = 1.0e-6,
    max_imag_abs: float = 1.0e-8,
    max_residual_estimate: float = 1.0e-6,
) -> ArnoldiValidationResult:
    """Validate Arnoldi Ritz values on a known non-normal upper-triangular operator."""
    if krylov_dim != 4:
        raise ValueError("krylov_dim must be 4 for the full-spectrum Arnoldi fixture")
    matrix = jnp.asarray(
        [
            [2.0, 0.4, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.0, 0.0, -0.5, 0.2],
            [0.0, 0.0, 0.0, 0.1],
        ]
    )
    expected_eigenvalues = np.asarray([2.0, 1.0, -0.5, 0.1])
    operator = MatrixFreeOperator(
        shape=(matrix.shape[0],),
        name="upper_triangular_arnoldi_fixture",
        matvec=lambda vector: matrix @ vector,
    )
    initial_vector = jnp.asarray([1.0, 0.3, -0.2, 0.1])
    result = arnoldi_iteration(operator, initial_vector, krylov_dim=krylov_dim)
    ritz_values = np.asarray(result.ritz_values)
    sorted_ritz = np.sort(ritz_values.real)
    sorted_expected = np.sort(expected_eigenvalues)
    ritz_error = float(np.max(np.abs(sorted_ritz - sorted_expected)))
    imag_error = float(np.max(np.abs(ritz_values.imag)))
    residual_error = float(np.max(np.asarray(result.residual_estimates)))
    checks = {
        "ritz_values_match_fixture_spectrum": ritz_error <= max_ritz_abs_error,
        "ritz_imaginary_parts_negligible": imag_error <= max_imag_abs,
        "arnoldi_residual_estimates_within_tolerance": (
            residual_error <= max_residual_estimate
        ),
    }
    diagnostics = {
        "schema": ARNOLDI_SCHEMA,
        "krylov_dim": krylov_dim,
        "expected_eigenvalues": [float(value) for value in expected_eigenvalues],
        "ritz_values_real": [float(value) for value in ritz_values.real],
        "ritz_values_imag": [float(value) for value in ritz_values.imag],
        "max_ritz_abs_error": ritz_error,
        "max_imag_abs": imag_error,
        "max_residual_estimate": residual_error,
        "references": {
            "arnoldi": "Krylov Ritz-value scaffold for matrix-free tearing eigenmodes",
            "fixture": "Non-normal upper-triangular matrix with known diagonal spectrum",
        },
    }
    validation = {
        "schema": "mhx.validation.arnoldi.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_ritz_abs_error": max_ritz_abs_error,
            "max_imag_abs": max_imag_abs,
            "max_residual_estimate": max_residual_estimate,
        },
        "diagnostics": diagnostics,
    }
    return ArnoldiValidationResult(
        expected_eigenvalues=expected_eigenvalues,
        ritz_values=ritz_values,
        max_ritz_abs_error=ritz_error,
        max_imag_abs=imag_error,
        max_residual_estimate=residual_error,
        residual_estimates=np.asarray(result.residual_estimates),
        hessenberg=np.asarray(result.hessenberg),
        diagnostics=diagnostics,
        validation=validation,
    )


def write_diffusion_eigenvalue_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write diffusion eigenvalue JSON, NPZ, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_diffusion_eigenvalue_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "diffusion_eigenvalue.npz"
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
        schema=DIFFUSION_EIGENVALUE_SCHEMA,
        eigenfunction=result.eigenfunction,
        operator_action=result.operator_action,
        expected_eigenvalue=result.expected_eigenvalue,
        measured_eigenvalue=result.measured_eigenvalue,
    )

    figure_path = plot_diffusion_eigenvalue_error(
        ("eigenvalue", "residual"),
        (result.eigenvalue_abs_error, result.residual_norm),
        (
            result.validation["thresholds"]["max_eigenvalue_abs_error"],
            result.validation["thresholds"]["max_residual_norm"],
        ),
        path=output_dir / "figures" / "diffusion_eigenvalue_errors.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "diffusion_eigenvalue_errors": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Diffusion eigenvalue solver validation.",
    )
    return manifest_path, result.validation


def write_arnoldi_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write Arnoldi validation JSON, NPZ, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_arnoldi_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "arnoldi_spectrum.npz"
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
        schema=ARNOLDI_SCHEMA,
        expected_eigenvalues=result.expected_eigenvalues,
        ritz_values=result.ritz_values,
        residual_estimates=result.residual_estimates,
        hessenberg=result.hessenberg,
    )

    figure_path = plot_arnoldi_ritz_values(
        result.expected_eigenvalues,
        result.ritz_values,
        result.residual_estimates,
        path=output_dir / "figures" / "arnoldi_ritz_values.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "arnoldi_ritz_values": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Arnoldi Ritz-spectrum numerical scaffold validation.",
    )
    return manifest_path, result.validation


def write_power_iteration_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write power-iteration validation JSON, NPZ, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_power_iteration_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "power_iteration_history.npz"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    iterations = np.arange(1, result.rayleigh_history.shape[0] + 1)
    np.savez_compressed(
        history_path,
        schema=POWER_ITERATION_SCHEMA,
        iterations=iterations,
        rayleigh_history=result.rayleigh_history,
        residual_history=result.residual_history,
        expected_eigenvalue=result.expected_eigenvalue,
    )

    figure_path = plot_power_iteration_history(
        iterations,
        result.rayleigh_history,
        result.residual_history,
        expected_eigenvalue=result.expected_eigenvalue,
        path=output_dir / "figures" / "power_iteration_history.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "power_iteration_history": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Power-iteration numerical scaffold validation.",
    )
    return manifest_path, result.validation


def run_linearized_rhs_validation(
    *,
    shape: tuple[int, int] = (16, 16),
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    epsilon: float = 1.0e-3,
    max_relative_error: float = 1.0e-3,
) -> LinearizedRHSResult:
    """Compare JAX JVP linearization against a centered finite difference."""
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    state = CosineTearingEquilibrium(perturbation_amplitude=1.0e-3).initial_state(grid)
    perturbation = ReducedMHDState(
        psi=grid.sinusoid(mode=(2, 1)) + 0.25 * grid.cosinusoid(mode=(1, 2)),
        omega=0.5 * grid.cosinusoid(mode=(1, 1)),
    )
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)
    jvp = linearized_reduced_mhd_rhs(
        state,
        perturbation,
        params,
        lengths=grid.lengths,
    )
    finite_difference = finite_difference_linearized_reduced_mhd_rhs(
        state,
        perturbation,
        params,
        lengths=grid.lengths,
        epsilon=epsilon,
    )
    absolute_errors = {
        "psi": _l2_norm(jvp.psi - finite_difference.psi),
        "omega": _l2_norm(jvp.omega - finite_difference.omega),
    }
    relative_errors = {
        "psi": absolute_errors["psi"] / max(_l2_norm(jvp.psi), 1.0e-300),
        "omega": absolute_errors["omega"] / max(_l2_norm(jvp.omega), 1.0e-300),
    }
    checks = {
        "psi_jvp_matches_centered_finite_difference": (
            relative_errors["psi"] <= max_relative_error
        ),
        "omega_jvp_matches_centered_finite_difference": (
            relative_errors["omega"] <= max_relative_error
        ),
    }
    diagnostics = {
        "schema": LINEARIZED_RHS_SCHEMA,
        "shape": list(shape),
        "resistivity": resistivity,
        "viscosity": viscosity,
        "epsilon": epsilon,
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "references": {
            "matrix_free_jvp": "JAX forward-mode JVP for differentiable PDE linearization",
            "tearing_context": (
                "Linearized reduced-MHD operator is the basis for tearing eigenmodes"
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.linearized_rhs.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"max_relative_error": max_relative_error},
        "diagnostics": diagnostics,
    }
    return LinearizedRHSResult(
        jvp=jvp,
        finite_difference=finite_difference,
        absolute_errors=absolute_errors,
        relative_errors=relative_errors,
        diagnostics=diagnostics,
        validation=validation,
    )


def run_reduced_mhd_linear_eigenmode_validation(
    *,
    shape: tuple[int, int] = (24, 24),
    mode: tuple[int, int] = (2, 1),
    resistivity: float = 2.0e-2,
    viscosity: float = 3.0e-2,
    max_eigenvalue_abs_error: float = 1.0e-6,
    max_residual_norm: float = 5.0e-6,
) -> ReducedMHDLinearEigenmodeResult:
    """Validate zero-state reduced-MHD linear diffusion eigenmodes."""
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    eigenfunction = grid.sinusoid(mode=mode)
    zero = jnp.zeros_like(eigenfunction)
    base_state = ReducedMHDState(psi=zero, omega=zero)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)
    operator = linearized_reduced_mhd_operator(base_state, params, lengths=grid.lengths)
    psi_vector = flatten_reduced_mhd_state(ReducedMHDState(psi=eigenfunction, omega=zero))
    omega_vector = flatten_reduced_mhd_state(ReducedMHDState(psi=zero, omega=eigenfunction))
    kx = 2.0 * np.pi * mode[0] / grid.lengths[0]
    ky = 2.0 * np.pi * mode[1] / grid.lengths[1]
    wavenumber_squared = kx**2 + ky**2
    expected_eigenvalues = {
        "psi": -resistivity * wavenumber_squared,
        "omega": -viscosity * wavenumber_squared,
    }
    measured_eigenvalues = {
        "psi": float(rayleigh_quotient(operator, psi_vector)),
        "omega": float(rayleigh_quotient(operator, omega_vector)),
    }
    eigenvalue_abs_errors = {
        name: abs(measured_eigenvalues[name] - expected_eigenvalues[name])
        for name in expected_eigenvalues
    }
    residual_norms = {
        "psi": float(eigen_residual_norm(operator, psi_vector, expected_eigenvalues["psi"])),
        "omega": float(eigen_residual_norm(operator, omega_vector, expected_eigenvalues["omega"])),
    }
    checks = {
        "psi_eigenvalue_matches_resistive_diffusion": (
            eigenvalue_abs_errors["psi"] <= max_eigenvalue_abs_error
        ),
        "omega_eigenvalue_matches_viscous_diffusion": (
            eigenvalue_abs_errors["omega"] <= max_eigenvalue_abs_error
        ),
        "psi_eigen_residual_within_tolerance": residual_norms["psi"] <= max_residual_norm,
        "omega_eigen_residual_within_tolerance": residual_norms["omega"] <= max_residual_norm,
    }
    diagnostics = {
        "schema": REDUCED_MHD_LINEAR_EIGENMODE_SCHEMA,
        "shape": list(shape),
        "mode": list(mode),
        "resistivity": resistivity,
        "viscosity": viscosity,
        "wavenumber_squared": wavenumber_squared,
        "expected_eigenvalues": expected_eigenvalues,
        "measured_eigenvalues": measured_eigenvalues,
        "eigenvalue_abs_errors": eigenvalue_abs_errors,
        "residual_norms": residual_norms,
        "references": {
            "linear_limit": (
                "At zero flow and zero flux, reduced MHD decouples into "
                "resistive psi diffusion and viscous omega diffusion."
            ),
            "tearing_context": (
                "This validates the flattened reduced-MHD JVP operator before "
                "nonzero-equilibrium tearing eigenmode calculations."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.reduced_mhd_linear_eigenmode.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_eigenvalue_abs_error": max_eigenvalue_abs_error,
            "max_residual_norm": max_residual_norm,
        },
        "diagnostics": diagnostics,
    }
    return ReducedMHDLinearEigenmodeResult(
        psi_eigenfunction=np.asarray(eigenfunction),
        omega_eigenfunction=np.asarray(eigenfunction),
        operator_psi_action=np.asarray(operator(psi_vector)),
        operator_omega_action=np.asarray(operator(omega_vector)),
        expected_eigenvalues=expected_eigenvalues,
        measured_eigenvalues=measured_eigenvalues,
        eigenvalue_abs_errors=eigenvalue_abs_errors,
        residual_norms=residual_norms,
        diagnostics=diagnostics,
        validation=validation,
    )


def run_cosine_equilibrium_linearization_validation(
    *,
    shape: tuple[int, int] = (24, 24),
    amplitude: float = 1.0,
    resistivity: float = 1.0e-3,
    viscosity: float = 2.0e-3,
    max_relative_error: float = 1.0e-4,
) -> CosineEquilibriumLinearizationResult:
    r"""Validate analytic linearized couplings around ``ψ₀=A cos(y)``.

    The gate checks two Fourier perturbations with closed-form reduced-MHD JVPs:

    - ``δω = cos(k_x x)``, ``δψ=0`` gives flow advection of the equilibrium
      flux plus viscous vorticity diffusion.
    - ``δψ = cos(k_x x)cos(2k_y y)``, ``δω=0`` gives magnetic-tension coupling
      to vorticity plus resistive flux diffusion.
    """
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=shape))
    x, y = grid.mesh()
    length_x, length_y = grid.lengths
    kx = 2.0 * jnp.pi / length_x
    ky = 2.0 * jnp.pi / length_y
    ky2 = 2.0 * ky
    psi0 = amplitude * jnp.cos(ky * y)
    zero = jnp.zeros_like(psi0)
    base_state = ReducedMHDState(psi=psi0, omega=zero)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    flow_omega = jnp.cos(kx * x)
    flow_perturbation = ReducedMHDState(psi=zero, omega=flow_omega)
    flow_tangent = linearized_reduced_mhd_rhs(
        base_state,
        flow_perturbation,
        params,
        lengths=grid.lengths,
    )
    expected_flow_tangent = ReducedMHDState(
        psi=amplitude * ky / kx * jnp.sin(kx * x) * jnp.sin(ky * y),
        omega=-viscosity * kx**2 * flow_omega,
    )

    tension_psi = jnp.cos(kx * x) * jnp.cos(ky2 * y)
    tension_perturbation = ReducedMHDState(psi=tension_psi, omega=zero)
    tension_tangent = linearized_reduced_mhd_rhs(
        base_state,
        tension_perturbation,
        params,
        lengths=grid.lengths,
    )
    tension_wavenumber_squared = kx**2 + ky2**2
    expected_tension_tangent = ReducedMHDState(
        psi=-resistivity * tension_wavenumber_squared * tension_psi,
        omega=amplitude
        * kx
        * ky
        * (tension_wavenumber_squared - ky**2)
        * jnp.sin(kx * x)
        * jnp.sin(ky * y)
        * jnp.cos(ky2 * y),
    )

    relative_errors = {
        "flow_to_flux_psi": _relative_l2_error(
            flow_tangent.psi,
            expected_flow_tangent.psi,
        ),
        "flow_vorticity_diffusion": _relative_l2_error(
            flow_tangent.omega,
            expected_flow_tangent.omega,
        ),
        "tension_flux_diffusion": _relative_l2_error(
            tension_tangent.psi,
            expected_tension_tangent.psi,
        ),
        "tension_to_vorticity": _relative_l2_error(
            tension_tangent.omega,
            expected_tension_tangent.omega,
        ),
    }
    checks = {
        name: value <= max_relative_error for name, value in relative_errors.items()
    }
    diagnostics = {
        "schema": COSINE_EQUILIBRIUM_LINEARIZATION_SCHEMA,
        "shape": list(shape),
        "equilibrium": "psi0 = A cos(2π y / Ly)",
        "amplitude": amplitude,
        "resistivity": resistivity,
        "viscosity": viscosity,
        "wavenumbers": {
            "kx": float(kx),
            "ky": float(ky),
            "ky2": float(ky2),
            "tension_wavenumber_squared": float(tension_wavenumber_squared),
        },
        "relative_errors": relative_errors,
        "references": {
            "reduced_mhd_linearization": (
                "Nonzero-equilibrium JVP checks the ideal advection and "
                "magnetic-tension brackets used by tearing eigenmode operators."
            ),
            "tearing_context": (
                "The cosine current sheet is periodic and analytically tractable; "
                "it validates current-sheet coupling terms without claiming an "
                "FKR growth rate."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.cosine_equilibrium_linearization.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"max_relative_error": max_relative_error},
        "diagnostics": diagnostics,
    }
    return CosineEquilibriumLinearizationResult(
        flow_tangent=flow_tangent,
        expected_flow_tangent=expected_flow_tangent,
        tension_tangent=tension_tangent,
        expected_tension_tangent=expected_tension_tangent,
        relative_errors=relative_errors,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_linearized_rhs_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write linearized-RHS validation JSON, NPZ, figure, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_linearized_rhs_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "linearized_rhs.npz"
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
        schema=LINEARIZED_RHS_SCHEMA,
        jvp_psi=np.asarray(result.jvp.psi),
        jvp_omega=np.asarray(result.jvp.omega),
        finite_difference_psi=np.asarray(result.finite_difference.psi),
        finite_difference_omega=np.asarray(result.finite_difference.omega),
    )

    figure_path = plot_linearized_rhs_errors(
        tuple(result.relative_errors),
        tuple(result.relative_errors.values()),
        max_relative_error=float(result.validation["thresholds"]["max_relative_error"]),
        path=output_dir / "figures" / "linearized_rhs_errors.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "linearized_rhs_errors": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Matrix-free reduced-MHD linearized-RHS tangent validation.",
    )
    return manifest_path, result.validation


def write_reduced_mhd_linear_eigenmode_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write reduced-MHD linear eigenmode JSON, NPZ, figure, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_reduced_mhd_linear_eigenmode_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "reduced_mhd_linear_eigenmode.npz"
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
        schema=REDUCED_MHD_LINEAR_EIGENMODE_SCHEMA,
        psi_eigenfunction=result.psi_eigenfunction,
        omega_eigenfunction=result.omega_eigenfunction,
        operator_psi_action=result.operator_psi_action,
        operator_omega_action=result.operator_omega_action,
        expected_psi_eigenvalue=result.expected_eigenvalues["psi"],
        expected_omega_eigenvalue=result.expected_eigenvalues["omega"],
        measured_psi_eigenvalue=result.measured_eigenvalues["psi"],
        measured_omega_eigenvalue=result.measured_eigenvalues["omega"],
    )

    figure_path = plot_reduced_mhd_eigenmode_errors(
        ("psi eigenvalue", "omega eigenvalue", "psi residual", "omega residual"),
        (
            result.eigenvalue_abs_errors["psi"],
            result.eigenvalue_abs_errors["omega"],
            result.residual_norms["psi"],
            result.residual_norms["omega"],
        ),
        (
            result.validation["thresholds"]["max_eigenvalue_abs_error"],
            result.validation["thresholds"]["max_eigenvalue_abs_error"],
            result.validation["thresholds"]["max_residual_norm"],
            result.validation["thresholds"]["max_residual_norm"],
        ),
        path=output_dir / "figures" / "reduced_mhd_linear_eigenmode_errors.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "reduced_mhd_linear_eigenmode_errors": str(
                figure_path.relative_to(output_dir)
            ),
        },
        claim_level="validation",
        claim_scope="Reduced-MHD linear eigenmode residual validation.",
    )
    return manifest_path, result.validation


def write_cosine_equilibrium_linearization_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write cosine-equilibrium linearization JSON, NPZ, figure, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_cosine_equilibrium_linearization_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "cosine_equilibrium_linearization.npz"
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
        schema=COSINE_EQUILIBRIUM_LINEARIZATION_SCHEMA,
        flow_tangent_psi=np.asarray(result.flow_tangent.psi),
        flow_tangent_omega=np.asarray(result.flow_tangent.omega),
        expected_flow_tangent_psi=np.asarray(result.expected_flow_tangent.psi),
        expected_flow_tangent_omega=np.asarray(result.expected_flow_tangent.omega),
        tension_tangent_psi=np.asarray(result.tension_tangent.psi),
        tension_tangent_omega=np.asarray(result.tension_tangent.omega),
        expected_tension_tangent_psi=np.asarray(result.expected_tension_tangent.psi),
        expected_tension_tangent_omega=np.asarray(result.expected_tension_tangent.omega),
    )

    figure_path = plot_cosine_equilibrium_linearization_errors(
        tuple(result.relative_errors),
        tuple(result.relative_errors.values()),
        tuple(
            result.validation["thresholds"]["max_relative_error"]
            for _ in result.relative_errors
        ),
        path=output_dir / "figures" / "cosine_equilibrium_linearization_errors.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "cosine_equilibrium_linearization_errors": str(
                figure_path.relative_to(output_dir)
            ),
        },
        claim_level="validation",
        claim_scope="Cosine-equilibrium nonzero linearization validation.",
    )
    return manifest_path, result.validation
