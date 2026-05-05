"""FKR constant-psi tearing regime-window validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mhx.benchmarks.theory import fkr_constant_psi_estimate, harris_sheet_delta_prime
from mhx.io import write_manifest
from mhx.plotting import plot_fkr_validity_window, plot_harris_delta_prime

FKR_WINDOW_SCHEMA = "mhx.validation.fkr_window.v1"
HARRIS_DELTA_PRIME_SCHEMA = "mhx.validation.harris_delta_prime.v1"


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


@dataclass(frozen=True)
class HarrisDeltaPrimeResult:
    """Numerical Harris-sheet outer-region Delta-prime validation artifacts."""

    ka: np.ndarray
    numerical_delta_prime_a: np.ndarray
    analytic_delta_prime_a: np.ndarray
    relative_error: np.ndarray
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


def run_harris_delta_prime_validation(
    *,
    ka: tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.7),
    xmax_over_a: float = 18.0,
    steps: int = 4000,
    max_relative_error: float = 1.0e-8,
) -> HarrisDeltaPrimeResult:
    r"""Numerically recover the Harris-sheet outer-region ``Δ'a`` formula.

    For the Harris equilibrium ``B_y=B_0 tanh(x/a)``, the ideal outer tearing
    equation in dimensionless ``ξ=x/a`` is

    ``ψ'' - [(ka)^2 - 2 sech^2(ξ)] ψ = 0``.

    Decaying solutions on each side give the matching parameter

    ``Δ'a = 2 ψ'(0+)/ψ(0) = 2[(ka)^(-1)-ka]``.

    This gate integrates the outer ODE backward from a large positive
    ``xmax_over_a`` using RK4 and compares the recovered matching parameter
    against the analytic expression. It validates the outer-region tearing
    target used by FKR scaling; it is not yet the resistive inner-layer
    eigenvalue problem.
    """
    ka_values = np.asarray(ka, dtype=float)
    if ka_values.ndim != 1 or ka_values.shape[0] < 3:
        raise ValueError("at least three ka samples are required")
    if np.any(ka_values <= 0.0):
        raise ValueError("ka samples must be positive")
    if np.any(ka_values >= 1.0):
        raise ValueError("positive Harris Delta-prime requires ka < 1")
    if np.any(np.diff(ka_values) <= 0.0):
        raise ValueError("ka samples must be strictly increasing")
    if xmax_over_a <= 1.0:
        raise ValueError("xmax_over_a must be greater than 1")
    if steps < 100:
        raise ValueError("steps must be at least 100")

    numerical = np.asarray(
        [
            _integrate_harris_outer_delta_prime(
                float(value),
                xmax_over_a=xmax_over_a,
                steps=steps,
            )
            for value in ka_values
        ]
    )
    analytic = np.asarray([harris_sheet_delta_prime(float(value)) for value in ka_values])
    relative_error = np.abs(numerical - analytic) / np.maximum(np.abs(analytic), 1.0e-300)
    checks = {
        "finite_numerical_delta_prime": bool(np.all(np.isfinite(numerical))),
        "positive_delta_prime": bool(np.all(numerical > 0.0)),
        "matches_harris_outer_formula": bool(np.all(relative_error <= max_relative_error)),
        "strictly_decreasing_with_ka": bool(np.all(np.diff(numerical) < 0.0)),
    }
    diagnostics = {
        "schema": HARRIS_DELTA_PRIME_SCHEMA,
        "ka": ka_values.tolist(),
        "xmax_over_a": xmax_over_a,
        "steps": steps,
        "numerical_delta_prime_a": numerical.tolist(),
        "analytic_delta_prime_a": analytic.tolist(),
        "relative_error": relative_error.tolist(),
        "max_relative_error_observed": float(np.max(relative_error)),
        "references": {
            "harris_outer": (
                "Harris-sheet ideal outer equation gives Delta'a = "
                "2[(ka)^(-1)-ka] for the tearing parity solution."
            ),
            "fkr_context": (
                "Outer Delta-prime is the matching input for the FKR "
                "constant-psi tearing growth-rate estimate."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.harris_delta_prime.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_relative_error": max_relative_error,
            "xmax_over_a": xmax_over_a,
            "steps": steps,
        },
        "diagnostics": diagnostics,
    }
    return HarrisDeltaPrimeResult(
        ka=ka_values,
        numerical_delta_prime_a=numerical,
        analytic_delta_prime_a=analytic,
        relative_error=relative_error,
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


def write_harris_delta_prime_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write Harris outer Delta-prime validation JSON, NPZ, figure, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_harris_delta_prime_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "harris_delta_prime.npz"
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
        schema=HARRIS_DELTA_PRIME_SCHEMA,
        ka=result.ka,
        numerical_delta_prime_a=result.numerical_delta_prime_a,
        analytic_delta_prime_a=result.analytic_delta_prime_a,
        relative_error=result.relative_error,
    )

    figure_path = plot_harris_delta_prime(
        result.ka,
        result.numerical_delta_prime_a,
        result.analytic_delta_prime_a,
        result.relative_error,
        max_relative_error=float(result.validation["thresholds"]["max_relative_error"]),
        path=output_dir / "figures" / "harris_delta_prime.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "harris_delta_prime": str(figure_path.relative_to(output_dir)),
        },
    )
    return manifest_path, result.validation


def _integrate_harris_outer_delta_prime(
    ka: float,
    *,
    xmax_over_a: float,
    steps: int,
) -> float:
    step = -xmax_over_a / steps
    coordinate = xmax_over_a
    state = np.asarray([1.0, -ka], dtype=float)

    for _ in range(steps):
        k1 = _harris_outer_rhs(coordinate, state, ka)
        k2 = _harris_outer_rhs(coordinate + 0.5 * step, state + 0.5 * step * k1, ka)
        k3 = _harris_outer_rhs(coordinate + 0.5 * step, state + 0.5 * step * k2, ka)
        k4 = _harris_outer_rhs(coordinate + step, state + step * k3, ka)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        coordinate += step
    psi_at_origin, dpsi_at_origin = state
    return float(2.0 * dpsi_at_origin / psi_at_origin)


def _harris_outer_rhs(coordinate: float, state: np.ndarray, ka: float) -> np.ndarray:
    psi, dpsi = state
    sech_squared = 1.0 / np.cosh(coordinate) ** 2
    return np.asarray([dpsi, (ka**2 - 2.0 * sech_squared) * psi])
