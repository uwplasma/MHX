"""FKR constant-psi tearing regime-window validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mhx.benchmarks.theory import fkr_constant_psi_estimate
from mhx.io import write_manifest
from mhx.plotting import plot_fkr_validity_window

FKR_WINDOW_SCHEMA = "mhx.validation.fkr_window.v1"


@dataclass(frozen=True)
class FKRWindowResult:
    """Computed FKR regime-window arrays and pass/fail gates."""

    ka: np.ndarray
    gamma_tau_a: np.ndarray
    inner_width_a: np.ndarray
    delta_prime_a: np.ndarray
    constant_psi_product: np.ndarray
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_fkr_window_validation(
    *,
    lundquist: float = 1.0e6,
    ka: tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.7),
    max_inner_width_a: float = 0.05,
    max_constant_psi_product: float = 0.5,
) -> FKRWindowResult:
    r"""Validate that sampled modes lie in a conservative FKR regime window.

    The gate checks the constant-$\psi$ conditions used when applying the
    Furth-Killeen-Rosenbluth scaling estimate: positive Harris-sheet
    $\Delta'a$, a thin resistive layer $\delta/a$, and
    $\Delta'\delta \ll 1$. It is an analytic regime-selection benchmark, not an
    eigenvalue solve.
    """
    ka_values = np.asarray(ka, dtype=float)
    if ka_values.ndim != 1 or ka_values.shape[0] < 3:
        raise ValueError("at least three ka samples are required")
    if np.any(ka_values <= 0.0):
        raise ValueError("ka samples must be positive")
    if np.any(ka_values >= 1.0):
        raise ValueError("FKR constant-psi window requires ka < 1 for positive delta_prime")
    if np.any(np.diff(ka_values) <= 0.0):
        raise ValueError("ka samples must be strictly increasing")

    estimates = [fkr_constant_psi_estimate(lundquist=lundquist, ka=float(value)) for value in ka]
    gamma_tau_a = np.asarray([estimate.gamma_tau_a for estimate in estimates])
    inner_width_a = np.asarray([estimate.inner_width_a for estimate in estimates])
    delta_prime_a = np.asarray([estimate.delta_prime_a for estimate in estimates])
    constant_psi_product = delta_prime_a * inner_width_a

    checks = {
        "positive_delta_prime": bool(np.all(delta_prime_a > 0.0)),
        "thin_inner_layer": bool(np.all(inner_width_a <= max_inner_width_a)),
        "constant_psi_product_within_gate": bool(
            np.all(constant_psi_product <= max_constant_psi_product)
        ),
        "finite_positive_growth_estimates": bool(
            np.all(np.isfinite(gamma_tau_a)) and np.all(gamma_tau_a > 0.0)
        ),
    }
    diagnostics = {
        "schema": FKR_WINDOW_SCHEMA,
        "lundquist": lundquist,
        "ka": ka_values.tolist(),
        "gamma_tau_a": gamma_tau_a.tolist(),
        "inner_width_a": inner_width_a.tolist(),
        "delta_prime_a": delta_prime_a.tolist(),
        "constant_psi_product": constant_psi_product.tolist(),
        "references": {
            "fkr": "Furth, Killeen & Rosenbluth 1963 constant-psi tearing regime",
            "coppi_context": "Coppi large-Delta-prime regime is outside this gate",
        },
    }
    validation = {
        "schema": "mhx.validation.fkr_window.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_inner_width_a": max_inner_width_a,
            "max_constant_psi_product": max_constant_psi_product,
        },
        "diagnostics": diagnostics,
    }
    return FKRWindowResult(
        ka=ka_values,
        gamma_tau_a=gamma_tau_a,
        inner_width_a=inner_width_a,
        delta_prime_a=delta_prime_a,
        constant_psi_product=constant_psi_product,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_fkr_window_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write FKR regime-window JSON, NPZ, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_fkr_window_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "fkr_window.npz"
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
        schema=FKR_WINDOW_SCHEMA,
        ka=result.ka,
        gamma_tau_a=result.gamma_tau_a,
        inner_width_a=result.inner_width_a,
        delta_prime_a=result.delta_prime_a,
        constant_psi_product=result.constant_psi_product,
    )

    figure_path = plot_fkr_validity_window(
        result.ka,
        result.gamma_tau_a,
        result.constant_psi_product,
        max_constant_psi_product=float(result.validation["thresholds"]["max_constant_psi_product"]),
        path=output_dir / "figures" / "fkr_constant_psi_window.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "fkr_window": str(figure_path.relative_to(output_dir)),
        },
    )
    return manifest_path, result.validation
