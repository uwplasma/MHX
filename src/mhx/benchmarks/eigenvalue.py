"""Analytic matrix-free diffusion eigenvalue validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics import (
    MatrixFreeOperator,
    eigen_residual_norm,
    power_iteration,
    rayleigh_quotient,
)
from mhx.numerics.spectral import laplacian
from mhx.plotting import plot_diffusion_eigenvalue_error, plot_power_iteration_history

DIFFUSION_EIGENVALUE_SCHEMA = "mhx.validation.diffusion_eigenvalue.v1"
POWER_ITERATION_SCHEMA = "mhx.validation.power_iteration.v1"


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
    )
    return manifest_path, result.validation
