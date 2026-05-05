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
    linearized_reduced_mhd_rhs,
)
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.physics import CosineTearingEquilibrium
from mhx.plotting import plot_linearized_rhs_errors
from mhx.state import ReducedMHDParams, ReducedMHDState

LINEARIZED_RHS_SCHEMA = "mhx.validation.linearized_rhs.v1"


@dataclass(frozen=True)
class LinearizedRHSResult:
    """JVP/finite-difference consistency diagnostics for the reduced-MHD RHS."""

    jvp: ReducedMHDState
    finite_difference: ReducedMHDState
    absolute_errors: dict[str, float]
    relative_errors: dict[str, float]
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


def _l2_norm(values) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.asarray(values) ** 2)))
