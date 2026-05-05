"""Periodic current-sheet eigenvalue validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.equations.reduced_mhd import linearized_reduced_mhd_operator
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.physics import CosineTearingEquilibrium
from mhx.plotting import plot_periodic_current_sheet_spectrum
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    flatten_reduced_mhd_state,
)

PERIODIC_CURRENT_SHEET_EIGENVALUE_SCHEMA = (
    "mhx.validation.periodic_current_sheet_eigenvalue.v1"
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
