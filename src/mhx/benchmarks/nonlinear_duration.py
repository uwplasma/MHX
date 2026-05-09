"""Nonlinear-run duration audit against literature time scales."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mhx.benchmarks.theory import loureiro_plasmoid_estimate
from mhx.io import write_manifest
from mhx.plotting import plot_nonlinear_duration_audit

NONLINEAR_DURATION_AUDIT_SCHEMA = "mhx.validation.nonlinear_duration_audit.v1"


@dataclass(frozen=True)
class NonlinearDurationAuditResult:
    """Reviewer-facing audit of current nonlinear runtime versus physics windows."""

    current_case_names: tuple[str, ...]
    current_end_times: np.ndarray
    target_names: tuple[str, ...]
    target_end_times: np.ndarray
    plasmoid_lundquist: np.ndarray
    plasmoid_efold_times: np.ndarray
    harris_growth_rate: float
    requested_linear_efolds: float
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_nonlinear_duration_audit(
    *,
    harris_growth_rate: float = 1.31e-2,
    requested_linear_efolds: float = 10.0,
    fast_smoke_t_end: float = 0.10,
    nonlinear_budget_t_end: float = 0.80,
    linear_timedomain_t_end: float = 80.0,
    plasmoid_lundquist: tuple[float, ...] = (1.0e4, 1.0e5, 1.0e6),
) -> NonlinearDurationAuditResult:
    r"""Audit whether current nonlinear FAST runs can support island/plasmoid claims.

    A growth-rate validation that aims to observe ``N`` e-foldings must satisfy
    ``t_end >= N/gamma``.  The current CI nonlinear runs are intentionally much
    shorter: they validate differentiability, energy signs, and artifact
    generation, not nonlinear Rutherford-island or plasmoid-chain physics.
    """
    _validate_duration_audit_inputs(
        harris_growth_rate=harris_growth_rate,
        requested_linear_efolds=requested_linear_efolds,
        fast_smoke_t_end=fast_smoke_t_end,
        nonlinear_budget_t_end=nonlinear_budget_t_end,
        linear_timedomain_t_end=linear_timedomain_t_end,
        plasmoid_lundquist=plasmoid_lundquist,
    )
    required_linear_window = requested_linear_efolds / harris_growth_rate
    target_names = (
        f"Harris k=0.5, S=1000: {requested_linear_efolds:g} e-folds",
        "Rutherford/island tracking campaign",
        "plasmoid-chain production campaign",
    )
    target_end_times = np.asarray(
        [
            required_linear_window,
            3.0 * required_linear_window,
            required_linear_window,
        ],
        dtype=np.float64,
    )
    current_case_names = (
        "FAST smoke trajectory",
        "nonlinear energy-budget gate",
        "linear Harris time-domain replay",
    )
    current_end_times = np.asarray(
        [fast_smoke_t_end, nonlinear_budget_t_end, linear_timedomain_t_end],
        dtype=np.float64,
    )
    plasmoid_s = np.asarray(plasmoid_lundquist, dtype=np.float64)
    plasmoid_efold_times = np.asarray(
        [1.0 / loureiro_plasmoid_estimate(float(value)).gamma_tau_a for value in plasmoid_s],
        dtype=np.float64,
    )
    max_current_nonlinear_time = float(max(fast_smoke_t_end, nonlinear_budget_t_end))
    nonlinear_fraction_of_target = max_current_nonlinear_time / required_linear_window
    linear_replay_fraction_of_target = linear_timedomain_t_end / required_linear_window
    checks = {
        "finite_times": bool(
            np.isfinite(current_end_times).all()
            and np.isfinite(target_end_times).all()
            and np.isfinite(plasmoid_efold_times).all()
        ),
        "current_nonlinear_runs_flagged_short": max_current_nonlinear_time
        < 0.05 * required_linear_window,
        "linear_replay_flagged_partial": linear_timedomain_t_end < required_linear_window,
        "production_targets_exceed_ci_gates": bool(
            np.min(target_end_times) > np.max(current_end_times[:2])
        ),
        "plasmoid_scaling_recorded": bool(np.all(np.diff(plasmoid_efold_times) < 0.0)),
    }
    diagnostics = {
        "schema": NONLINEAR_DURATION_AUDIT_SCHEMA,
        "harris_growth_rate": harris_growth_rate,
        "requested_linear_efolds": requested_linear_efolds,
        "required_linear_window": required_linear_window,
        "current_cases": [
            {"name": name, "t_end": float(t_end)}
            for name, t_end in zip(current_case_names, current_end_times, strict=True)
        ],
        "targets": [
            {"name": name, "t_end": float(t_end)}
            for name, t_end in zip(target_names, target_end_times, strict=True)
        ],
        "fractions": {
            "current_nonlinear_fraction_of_required_linear_window": (
                nonlinear_fraction_of_target
            ),
            "linear_replay_fraction_of_required_linear_window": (
                linear_replay_fraction_of_target
            ),
        },
        "plasmoid_estimates": [
            {
                "lundquist": float(value),
                "gamma_tau_a": float(loureiro_plasmoid_estimate(float(value)).gamma_tau_a),
                "one_efold_time_tau_a": float(time),
            }
            for value, time in zip(plasmoid_s, plasmoid_efold_times, strict=True)
        ],
        "interpretation": (
            "Current nonlinear FAST runs validate code paths and nonlinear energy "
            "balances only. They are too short to claim Rutherford island growth, "
            "Sweet-Parker reconnection rates, or plasmoid-chain nonlinear dynamics."
        ),
        "recommended_production_gates": (
            "Use adaptive stopping based on reconnected flux/island width; require "
            "at least 8-10 linear e-folds before nonlinear-island analysis; store "
            "time histories of reconnected flux, island width, current-sheet "
            "aspect ratio, E_rec proxy, and energy budget."
        ),
        "references": {
            "linear_growth_anchor": (
                "MacTaggart Harris-sheet benchmark gamma≈0.0131 at k=0.5, S=1000."
            ),
            "fkr": "Furth, Killeen & Rosenbluth, Phys. Fluids 6, 459 (1963).",
            "rutherford": "Rutherford, Phys. Fluids 16, 1903 (1973).",
            "plasmoid": (
                "Loureiro, Schekochihin & Cowley, Phys. Plasmas 14, 100703 (2007)."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.nonlinear_duration_audit.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_current_nonlinear_fraction_for_short_flag": 0.05,
            "requested_linear_efolds": requested_linear_efolds,
        },
        "diagnostics": diagnostics,
    }
    return NonlinearDurationAuditResult(
        current_case_names=current_case_names,
        current_end_times=current_end_times,
        target_names=target_names,
        target_end_times=target_end_times,
        plasmoid_lundquist=plasmoid_s,
        plasmoid_efold_times=plasmoid_efold_times,
        harris_growth_rate=harris_growth_rate,
        requested_linear_efolds=requested_linear_efolds,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_nonlinear_duration_audit(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write nonlinear-duration audit JSON, NPZ, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_nonlinear_duration_audit(**kwargs)
    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "nonlinear_duration_audit.npz"
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
        schema=NONLINEAR_DURATION_AUDIT_SCHEMA,
        current_case_names=np.asarray(result.current_case_names),
        current_end_times=result.current_end_times,
        target_names=np.asarray(result.target_names),
        target_end_times=result.target_end_times,
        plasmoid_lundquist=result.plasmoid_lundquist,
        plasmoid_efold_times=result.plasmoid_efold_times,
        harris_growth_rate=result.harris_growth_rate,
        requested_linear_efolds=result.requested_linear_efolds,
    )
    figure_path = plot_nonlinear_duration_audit(
        result.current_case_names,
        result.current_end_times,
        result.target_names,
        result.target_end_times,
        result.plasmoid_lundquist,
        result.plasmoid_efold_times,
        path=output_dir / "figures" / "nonlinear_duration_audit.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "nonlinear_duration_audit": str(figure_path.relative_to(output_dir)),
        },
    )
    return manifest_path, result.validation


def _validate_duration_audit_inputs(
    *,
    harris_growth_rate: float,
    requested_linear_efolds: float,
    fast_smoke_t_end: float,
    nonlinear_budget_t_end: float,
    linear_timedomain_t_end: float,
    plasmoid_lundquist: tuple[float, ...],
) -> None:
    if harris_growth_rate <= 0.0:
        raise ValueError("harris_growth_rate must be positive")
    if requested_linear_efolds <= 0.0:
        raise ValueError("requested_linear_efolds must be positive")
    if fast_smoke_t_end <= 0.0:
        raise ValueError("fast_smoke_t_end must be positive")
    if nonlinear_budget_t_end <= 0.0:
        raise ValueError("nonlinear_budget_t_end must be positive")
    if linear_timedomain_t_end <= 0.0:
        raise ValueError("linear_timedomain_t_end must be positive")
    if not plasmoid_lundquist or any(value <= 0.0 for value in plasmoid_lundquist):
        raise ValueError("plasmoid_lundquist values must be positive")
