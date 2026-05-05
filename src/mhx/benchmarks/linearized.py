"""Matrix-free reduced-MHD linearized-RHS validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import (
    finite_difference_linearized_reduced_mhd_rhs,
    linearized_reduced_mhd_operator,
    linearized_reduced_mhd_rhs,
)
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics import eigen_residual_norm, rayleigh_quotient
from mhx.physics import CosineTearingEquilibrium
from mhx.plotting import plot_linearized_rhs_errors, plot_reduced_mhd_eigenmode_errors
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    flatten_reduced_mhd_state,
)

LINEARIZED_RHS_SCHEMA = "mhx.validation.linearized_rhs.v1"
REDUCED_MHD_LINEAR_EIGENMODE_SCHEMA = "mhx.validation.reduced_mhd_linear_eigenmode.v1"


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
    )
    return manifest_path, result.validation


def _l2_norm(values) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.asarray(values) ** 2)))
