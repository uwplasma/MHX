"""Literature-anchored reconnection scaling validation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mhx.benchmarks.theory import (
    fkr_constant_psi_estimate,
    ideal_tearing_aspect_ratio,
    loureiro_plasmoid_estimate,
)
from mhx.io import write_manifest
from mhx.plotting import (
    plot_fkr_scaling,
    plot_ideal_tearing_scaling,
    plot_plasmoid_scaling,
)

RECONNECTION_SCALING_SCHEMA = "mhx.validation.reconnection_scaling.v1"


@dataclass(frozen=True)
class ReconnectionScalingResult:
    """Computed scaling arrays and pass/fail gates for analytic reconnection theory."""

    lundquist: np.ndarray
    fkr_gamma: np.ndarray
    fkr_inner_width: np.ndarray
    plasmoid_gamma: np.ndarray
    plasmoid_mode: np.ndarray
    ideal_aspect_ratio: np.ndarray
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return least-squares slope of ``log(y)`` against ``log(x)``."""
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("loglog_slope requires positive x and y")
    return float(np.polyfit(np.log(x), np.log(y), deg=1)[0])


def run_reconnection_scaling_validation(
    *,
    lundquist: tuple[float, ...] = (1.0e4, 1.0e5, 1.0e6, 1.0e7),
    ka: float = 0.5,
    max_slope_error: float = 1.0e-12,
) -> ReconnectionScalingResult:
    """Validate analytic FKR, plasmoid, and ideal-tearing power-law exponents."""
    s_values = np.asarray(lundquist, dtype=float)
    if s_values.ndim != 1 or s_values.shape[0] < 3:
        raise ValueError("at least three Lundquist samples are required")
    if np.any(np.diff(s_values) <= 0.0):
        raise ValueError("Lundquist samples must be strictly increasing")

    fkr_estimates = [fkr_constant_psi_estimate(float(s_value), ka=ka) for s_value in s_values]
    plasmoid_estimates = [loureiro_plasmoid_estimate(float(s_value)) for s_value in s_values]
    fkr_gamma = np.asarray([item.gamma_tau_a for item in fkr_estimates])
    fkr_inner_width = np.asarray([item.inner_width_a for item in fkr_estimates])
    plasmoid_gamma = np.asarray([item.gamma_tau_a for item in plasmoid_estimates])
    plasmoid_mode = np.asarray([item.fastest_mode_k_l for item in plasmoid_estimates])
    ideal_aspect_ratio = np.asarray(
        [ideal_tearing_aspect_ratio(float(s_value)) for s_value in s_values]
    )

    slopes = {
        "fkr_gamma": loglog_slope(s_values, fkr_gamma),
        "fkr_inner_width": loglog_slope(s_values, fkr_inner_width),
        "plasmoid_gamma": loglog_slope(s_values, plasmoid_gamma),
        "plasmoid_mode": loglog_slope(s_values, plasmoid_mode),
        "ideal_aspect_ratio": loglog_slope(s_values, ideal_aspect_ratio),
    }
    expected = {
        "fkr_gamma": -3.0 / 5.0,
        "fkr_inner_width": -2.0 / 5.0,
        "plasmoid_gamma": 1.0 / 4.0,
        "plasmoid_mode": 3.0 / 8.0,
        "ideal_aspect_ratio": -1.0 / 3.0,
    }
    slope_errors = {name: abs(slopes[name] - expected[name]) for name in expected}
    checks = {
        f"{name}_slope_matches_literature": error <= max_slope_error
        for name, error in slope_errors.items()
    }
    checks["fkr_constant_psi_delta_prime_positive"] = all(
        item.delta_prime_a > 0.0 for item in fkr_estimates
    )
    diagnostics = {
        "schema": RECONNECTION_SCALING_SCHEMA,
        "lundquist": s_values.tolist(),
        "ka": ka,
        "slopes": slopes,
        "expected_slopes": expected,
        "slope_errors": slope_errors,
        "references": {
            "fkr": "Furth, Killeen & Rosenbluth 1963 constant-psi tearing scaling",
            "plasmoid": "Loureiro, Schekochihin & Cowley 2007 Sweet-Parker plasmoid scaling",
            "ideal_tearing": "Pucci-Velli ideal-tearing aspect-ratio scaling",
        },
    }
    validation = {
        "schema": "mhx.validation.reconnection_scaling.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"max_slope_error": max_slope_error},
        "diagnostics": diagnostics,
    }
    return ReconnectionScalingResult(
        lundquist=s_values,
        fkr_gamma=fkr_gamma,
        fkr_inner_width=fkr_inner_width,
        plasmoid_gamma=plasmoid_gamma,
        plasmoid_mode=plasmoid_mode,
        ideal_aspect_ratio=ideal_aspect_ratio,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_reconnection_scaling_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write analytic scaling validation JSON, NPZ, figures, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_reconnection_scaling_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "scaling_history.npz"
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
        schema=RECONNECTION_SCALING_SCHEMA,
        lundquist=result.lundquist,
        fkr_gamma=result.fkr_gamma,
        fkr_inner_width=result.fkr_inner_width,
        plasmoid_gamma=result.plasmoid_gamma,
        plasmoid_mode=result.plasmoid_mode,
        ideal_aspect_ratio=result.ideal_aspect_ratio,
    )

    figure_dir = output_dir / "figures"
    fkr_path = plot_fkr_scaling(
        result.lundquist,
        result.fkr_gamma,
        result.fkr_inner_width,
        path=figure_dir / "fkr_scaling.png",
    )
    plasmoid_path = plot_plasmoid_scaling(
        result.lundquist,
        result.plasmoid_gamma,
        result.plasmoid_mode,
        path=figure_dir / "plasmoid_scaling.png",
    )
    ideal_path = plot_ideal_tearing_scaling(
        result.lundquist,
        result.ideal_aspect_ratio,
        path=figure_dir / "ideal_tearing_scaling.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "fkr_scaling": str(fkr_path.relative_to(output_dir)),
            "plasmoid_scaling": str(plasmoid_path.relative_to(output_dir)),
            "ideal_tearing_scaling": str(ideal_path.relative_to(output_dir)),
        },
    )
    return manifest_path, result.validation
