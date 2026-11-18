#!/usr/bin/env python3
"""Write nonlinear-campaign evidence claim tables from local artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "mhx.docs.nonlinear_campaign_evidence.v1"
GATE_SUMMARY_SCHEMA = "mhx.docs.nonlinear_campaign_gate_summary.v1"
ARTIFACT_MANIFEST_SCHEMA = "mhx.artifacts.v1"

PROMOTION_GATE_ORDER = (
    "duration",
    "convergence",
    "seed_qi",
    "critical_points",
    "flux",
    "island_width",
    "movies",
    "manifest",
)
PROMOTION_GATE_LABELS = {
    "duration": "Duration",
    "convergence": "Convergence",
    "seed_qi": "Seed-QI",
    "critical_points": "X/O critical points",
    "flux": "Reconnected flux",
    "island_width": "Island width",
    "movies": "Fixed-scale movies",
    "manifest": "Manifests",
}
DEFAULT_MIN_RESPONSE_AMPLIFICATION = 1.05


@dataclass(frozen=True)
class EvidenceCase:
    label: str
    family: str
    run_dir: Path
    command: str
    metric_keys: tuple[str, ...]
    blocker: str
    promotion_path: Path | None = None


CASES = (
    EvidenceCase(
        label="2026-05-22 bounded double-Harris convergence",
        family="periodic double-Harris",
        run_dir=Path(
            "outputs/nonlinear_campaign_evidence_20260522/double_harris_convergence_n16_24_t8"
        ),
        command=(
            "python -m mhx.cli.main benchmark double-harris-convergence "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/"
            "double_harris_convergence_n16_24_t8 "
            "--resolutions 16,24 --dt-values 0.02,0.01 --reference-resolution 16 "
            "--reference-dt 0.01 --t-end 8 --save-interval 1 --fit-stop 4"
        ),
        metric_keys=(
            "case_count",
            "t_end",
            "resolution_growth_rate_spread",
            "timestep_growth_rate_spread",
            "resolution_max_growth_spread",
            "timestep_max_growth_spread",
        ),
        blocker=(
            "Small 16/24 grid and short t=8 validation sweep; production needs larger "
            "resolution, duration, seed, width/aspect, and Lundquist sweeps."
        ),
    ),
    EvidenceCase(
        label="2026-05-22 bounded double-Harris long replay",
        family="periodic double-Harris",
        run_dir=Path("outputs/nonlinear_campaign_evidence_20260522/double_harris_long_n48_t24"),
        command=(
            "python -m mhx.cli.main benchmark double-harris-long-run "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/double_harris_long_n48_t24 "
            "--nx 48 --ny 48 --t-end 24 --dt 0.02 --save-every 100 "
            "--fit-stop 8 --min-max-growth-factor 2 --no-movies"
        ),
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "fitted_early_growth_rate",
            "max_growth_factor",
            "reconnected_flux_amplification",
            "island_width_amplification",
            "relative_energy_increase",
        ),
        blocker=(
            "Positive response is validation evidence only; no production-duration "
            "convergence, seed-QI, aspect-ratio, or Lundquist sweep is attached."
        ),
    ),
    EvidenceCase(
        label="2026-05-22 bounded Rutherford FAST run",
        family="Rutherford executor",
        run_dir=Path("outputs/nonlinear_campaign_evidence_20260522/rutherford_fast_n24_t1"),
        command=(
            "python -m mhx.cli.main campaign rutherford-run-fast "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/rutherford_fast_n24_t1 "
            "--nx 24 --ny 24 --t-end 1.0 --dt 0.01 --save-every 5"
        ),
        metric_keys=(
            "shape",
            "t_end",
            "steps",
            "samples",
            "max_relative_energy_growth",
            "max_magnetic_divergence_linf",
        ),
        blocker=(
            "FAST schema/diagnostic run is far shorter than Rutherford-duration requirements "
            "and cannot be promoted to nonlinear physics."
        ),
    ),
    EvidenceCase(
        label="2026-05-22 bounded forced turbulent reconnection",
        family="forced turbulent reconnection",
        run_dir=Path(
            "outputs/nonlinear_campaign_evidence_20260522/forced_turbulent_reconnection_n24_t4"
        ),
        command=(
            "python -m mhx.cli.main benchmark forced-turbulent-reconnection "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/"
            "forced_turbulent_reconnection_n24_t4 "
            "--nx 24 --ny 24 --t-end 4 --dt 0.02 --save-every 10 --no-movies"
        ),
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "reconnection_proxy_change",
            "max_abs_reconnection_rate_proxy",
            "current_linf_growth",
            "max_relative_energy_growth",
            "max_magnetic_divergence_linf",
        ),
        blocker=(
            "2-D reduced-MHD proxy and single deterministic seed; no turbulent ensemble, "
            "3-D physics, inertial range, or LV99 scaling evidence."
        ),
        promotion_path=Path("readiness/promotion_readiness.json"),
    ),
    EvidenceCase(
        label="Archived GPU-assisted double-Harris convergence",
        family="periodic double-Harris",
        run_dir=Path(
            "outputs/campaigns/double_harris_convergence_gpu_n32_48_64_t16_20260519_173637"
        ),
        command="Archived command recorded in docs/project/long_run_evidence.md.",
        metric_keys=(
            "case_count",
            "t_end",
            "resolution_growth_rate_spread",
            "timestep_growth_rate_spread",
            "resolution_max_growth_spread",
            "timestep_max_growth_spread",
        ),
        blocker=(
            "Medium validation sweep is not a production-scale duration, seed, "
            "aspect-ratio, or Lundquist campaign."
        ),
    ),
    EvidenceCase(
        label="2026-05-22 GPU double-Harris validation promotion",
        family="periodic double-Harris",
        run_dir=Path(
            "outputs/campaigns/gpu_nonlinear_20260522_085049/double_harris_long_n128_t160"
        ),
        command=(
            "timeout 1800 python3 -m mhx.cli.main benchmark double-harris-long-run "
            "--nx 128 --ny 128 --width 0.36 --eta 0.0045 --nu 0.0045 "
            "--perturbation-amplitude 0.004 --dt 0.02 --t-end 160 "
            "--save-every 100 --movies; followed by matched n=64/96/128 "
            "convergence, width, eta, seed-QI, artifact-manifest, and "
            "double-harris-promotion-check gates."
        ),
        metric_keys=(
            "shape",
            "width",
            "resistivity",
            "viscosity",
            "t_end",
            "samples",
            "fitted_early_growth_rate",
            "max_growth_factor",
            "reconnected_flux_amplification",
            "island_width_amplification",
            "max_x_point_count",
            "max_o_point_count",
            "relative_energy_increase",
        ),
        blocker=(
            "Convergence-backed validation media passes duration, X/O, flux, width, "
            "movie, and manifest gates, but the attached promotion report declares "
            "claim_level_if_passed=validation; it is not a production Rutherford, "
            "Sweet-Parker, or plasmoid-chain claim."
        ),
        promotion_path=Path("promotion/promotion_readiness.json"),
    ),
    EvidenceCase(
        label="Archived GPU-assisted double-Harris response",
        family="periodic double-Harris",
        run_dir=Path("outputs/campaigns/growing_double_harris_gpu_96_t120_20260518_044120"),
        command="Archived command recorded in docs/project/long_run_evidence.md.",
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "fitted_early_growth_rate",
            "max_growth_factor",
            "reconnected_flux_amplification",
            "island_width_amplification",
            "max_x_point_count",
            "max_o_point_count",
            "relative_energy_increase",
        ),
        blocker=(
            "Convergence-backed validation media only; production claims still need larger "
            "seed, width/aspect, Lundquist, and duration sweeps."
        ),
        promotion_path=Path("promotion/promotion_readiness.json"),
    ),
    EvidenceCase(
        label="Archived current-schema Rutherford duration run",
        family="Rutherford executor",
        run_dir=Path("outputs/campaigns/rutherford_current_schema_96_dt005_20260517_161235"),
        command="Archived command recorded in docs/project/long_run_evidence.md.",
        metric_keys=(
            "shape",
            "end_step",
            "target_step",
            "history_samples",
            "max_relative_energy_growth",
            "final_magnetic_divergence_linf",
        ),
        blocker=(
            "Duration target completed, but promotion failed because reconnecting-flux "
            "and island-width amplification remained 1.00."
        ),
        promotion_path=Path("promotion/promotion_readiness.json"),
    ),
    EvidenceCase(
        label="Archived forced turbulent reconnection README media",
        family="forced turbulent reconnection",
        run_dir=Path("outputs/readme_media/forced_turbulent_reconnection_64_t80_wide"),
        command="Archived command recorded in docs/readme media QA artifacts.",
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "reconnection_proxy_change",
            "max_abs_reconnection_rate_proxy",
            "current_linf_growth",
            "max_relative_energy_growth",
            "max_magnetic_divergence_linf",
        ),
        blocker=(
            "Validation media only: single deterministic 2-D proxy run, no ensemble or "
            "3-D turbulent-reconnection scaling."
        ),
    ),
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compact_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [compact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_value(item) for key, item in value.items()}
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(converted):
        return None
    return converted


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _explicit_check(checks: dict[str, Any], key: str) -> bool | None:
    if key not in checks:
        return None
    return bool(checks[key])


def _check_or_fallback(checks: dict[str, Any], key: str, fallback: bool) -> bool:
    explicit = _explicit_check(checks, key)
    if explicit is not None:
        return explicit
    return bool(fallback)


def _file_is_present(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _artifact_manifest_files(artifact_manifest: dict[str, Any]) -> set[str]:
    files = artifact_manifest.get("files", [])
    if not isinstance(files, list):
        return set()
    return {
        str(record.get("path"))
        for record in files
        if isinstance(record, dict) and record.get("path") is not None
    }


def _select_history_path(root: Path) -> Path:
    for relative_path in (
        "production_history.npz",
        "periodic_double_harris_seeded_long_run.npz",
    ):
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return root / "production_history.npz"


def _array_amplification(values: np.ndarray) -> float | None:
    array = np.abs(np.asarray(values, dtype=np.float64))
    if array.size == 0 or not np.isfinite(array).all():
        return None
    initial = float(array[0])
    peak = float(np.max(array))
    denominator = max(initial, np.finfo(np.float64).tiny)
    return peak / denominator


def _load_history_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            stats: dict[str, Any] = {
                "present": True,
                "keys": sorted(keys),
                "schema": str(data["schema"]) if "schema" in keys else None,
                "sample_count": int(np.asarray(data["time"]).size) if "time" in keys else 0,
                "finite_arrays": all(
                    np.isfinite(np.asarray(data[key])).all()
                    for key in keys
                    if key != "schema" and np.asarray(data[key]).dtype.kind in {"f", "i", "u"}
                ),
                "has_reconnected_flux": "reconnected_flux" in keys,
                "has_island_width": "rutherford_island_width" in keys,
                "has_critical_point_counts": {"x_point_count", "o_point_count"}.issubset(keys),
            }
            if "time" in keys and np.asarray(data["time"]).size:
                stats["terminal_time"] = float(np.asarray(data["time"])[-1])
            if "step" in keys and np.asarray(data["step"]).size:
                stats["terminal_step"] = int(np.asarray(data["step"])[-1])
            if "x_point_count" in keys and np.asarray(data["x_point_count"]).size:
                stats["max_x_point_count"] = int(np.max(np.asarray(data["x_point_count"])))
            if "o_point_count" in keys and np.asarray(data["o_point_count"]).size:
                stats["max_o_point_count"] = int(np.max(np.asarray(data["o_point_count"])))
            if "reconnected_flux" in keys:
                stats["reconnected_flux_amplification"] = _array_amplification(
                    np.asarray(data["reconnected_flux"])
                )
            if "rutherford_island_width" in keys:
                stats["island_width_amplification"] = _array_amplification(
                    np.asarray(data["rutherford_island_width"])
                )
            return stats
    except (OSError, ValueError, KeyError):
        return {"present": True, "readable": False}


def _read_first_json(
    root: Path,
    relative_paths: tuple[str, ...],
) -> tuple[dict[str, Any], str | None]:
    for relative_path in relative_paths:
        path = root / relative_path
        if path.exists():
            return read_json(path), relative_path
    return {}, None


def _gate(checks: dict[str, bool], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    blockers = [key for key, value in checks.items() if not bool(value)]
    return {
        "passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "evidence": compact_value(evidence or {}),
    }


def _promotion_sources(root: Path) -> dict[str, Any]:
    promotion, promotion_path = _read_first_json(
        root,
        (
            "promotion/promotion_readiness.json",
            "readiness/promotion_readiness.json",
            "promotion_readiness.json",
        ),
    )
    promotion_validation, promotion_validation_path = _read_first_json(
        root,
        ("promotion/validation.json", "readiness/validation.json"),
    )
    diagnostics = promotion.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = promotion_validation.get("diagnostics", {})
    return {
        "promotion": promotion,
        "promotion_path": promotion_path,
        "promotion_validation": promotion_validation,
        "promotion_validation_path": promotion_validation_path,
        "promotion_diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
    }


def summarize_campaign_dir(
    campaign_dir: str | Path,
    *,
    require_movies: bool = True,
) -> dict[str, Any]:
    """Summarize production-promotion evidence for one campaign directory."""
    root = Path(campaign_dir)
    plan = read_json(root / "campaign_plan.json")
    diagnostics = read_json(root / "diagnostics.json")
    validation = read_json(root / "validation.json")
    manifest = read_json(root / "manifest.json")
    artifact_manifest = read_json(root / "artifact_manifest.json")
    artifact_files = _artifact_manifest_files(artifact_manifest)
    manifest_hashes = manifest.get("hashes", {})
    if not isinstance(manifest_hashes, dict):
        manifest_hashes = {}
    manifest_outputs = manifest.get("outputs", {})
    if not isinstance(manifest_outputs, dict):
        manifest_outputs = {}
    history_path = _select_history_path(root)
    history_relative_path = history_path.relative_to(root).as_posix()
    history_stats = _load_history_stats(history_path)
    promotion_sources = _promotion_sources(root)
    promotion = promotion_sources["promotion"]
    promotion_validation = promotion_sources["promotion_validation"]
    promotion_diagnostics = promotion_sources["promotion_diagnostics"]
    promotion_checks = promotion.get("checks", {})
    if not isinstance(promotion_checks, dict):
        promotion_checks = promotion_validation.get("checks", {})
    if not isinstance(promotion_checks, dict):
        promotion_checks = {}
    thresholds = _first_present(
        promotion.get("thresholds"),
        promotion_validation.get("thresholds"),
        promotion_diagnostics.get("thresholds"),
        {},
    )
    if not isinstance(thresholds, dict):
        thresholds = {}
    claim_level_if_gates_pass = str(
        _first_present(
            promotion.get("claim_level_if_passed"),
            promotion_diagnostics.get("claim_level_if_passed"),
            "production",
        )
    )
    production_gate_requested = claim_level_if_gates_pass == "production"

    duration_assessment = plan.get("duration_assessment", {})
    if not isinstance(duration_assessment, dict):
        duration_assessment = {}
    plan_validation_checks = validation.get("checks", {})
    if not isinstance(plan_validation_checks, dict):
        plan_validation_checks = {}
    target_step = _as_int(
        _first_present(
            promotion.get("target_step"),
            promotion_diagnostics.get("target_step"),
            diagnostics.get("target_step"),
            plan.get("estimated_steps"),
        )
    )
    terminal_step = _as_int(
        _first_present(
            promotion.get("terminal_step"),
            promotion_diagnostics.get("terminal_step"),
            diagnostics.get("end_step"),
            history_stats.get("terminal_step"),
        )
    )
    terminal_time = _as_float(
        _first_present(
            promotion.get("terminal_time"),
            promotion_diagnostics.get("terminal_time"),
            diagnostics.get("t_end"),
            history_stats.get("terminal_time"),
        )
    )
    min_t_end = _as_float(
        _first_present(
            thresholds.get("min_t_end"),
            thresholds.get("documented_readme_media_min_t_end"),
            duration_assessment.get("required_t_end"),
        )
    )
    duration_guard_allows_production = bool(
        duration_assessment.get("sufficient_for_production_claim")
        or duration_assessment.get("sufficient_for_nonlinear_claim")
        or plan_validation_checks.get("duration_guard_passed")
    )
    minimum_duration_reached = _explicit_check(promotion_checks, "minimum_duration_reached")
    if minimum_duration_reached is None:
        minimum_duration_reached = (
            terminal_time is not None and min_t_end is not None and terminal_time >= min_t_end
        ) or duration_guard_allows_production
    completed_target = _explicit_check(promotion_checks, "completed_target")
    if completed_target is None:
        completed_target = bool(diagnostics.get("completed_target")) or bool(
            minimum_duration_reached
        )
    duration_source_present = bool(plan) or terminal_time is not None or target_step is not None
    duration_checks = {
        "duration_source_present": duration_source_present,
        "minimum_duration_reached": bool(minimum_duration_reached),
        "completed_target_or_minimum_duration": bool(completed_target),
    }
    if target_step is not None or terminal_step is not None:
        duration_checks["terminal_step_reaches_target"] = _check_or_fallback(
            promotion_checks,
            "terminal_step_reaches_target",
            target_step is not None and terminal_step is not None and terminal_step >= target_step,
        )

    convergence_reports = _first_present(
        promotion.get("convergence_reports"),
        promotion_diagnostics.get("convergence_reports"),
        [],
    )
    if not isinstance(convergence_reports, list):
        convergence_reports = []
    min_convergence_dirs = int(thresholds.get("min_convergence_dirs", 2))
    convergence_checks = {
        "convergence_bundle_count": _check_or_fallback(
            promotion_checks,
            "convergence_bundle_count",
            len(convergence_reports) >= min_convergence_dirs,
        ),
        "convergence_bundles_passed": _check_or_fallback(
            promotion_checks,
            "convergence_bundles_passed",
            bool(convergence_reports)
            and len(convergence_reports) >= min_convergence_dirs
            and all(bool(report.get("passed")) for report in convergence_reports),
        ),
    }

    seed_qi_report = _first_present(
        promotion.get("seed_qi_report"),
        promotion_diagnostics.get("seed_qi_report"),
    )
    seed_qi_required = production_gate_requested or any(
        key.startswith("seed_qi") for key in promotion_checks
    )
    if seed_qi_required:
        seed_qi_checks = {
            "seed_qi_bundle_present": _check_or_fallback(
                promotion_checks,
                "seed_qi_bundle_present",
                isinstance(seed_qi_report, dict),
            ),
            "seed_qi_bundle_passed": _check_or_fallback(
                promotion_checks,
                "seed_qi_bundle_passed",
                isinstance(seed_qi_report, dict) and bool(seed_qi_report.get("passed")),
            ),
        }
    else:
        seed_qi_checks = {"seed_qi_not_required_for_declared_claim": True}

    max_x_point_count = _as_int(
        _first_present(
            promotion.get("max_x_point_count"),
            promotion_diagnostics.get("max_x_point_count"),
            history_stats.get("max_x_point_count"),
        )
    )
    max_o_point_count = _as_int(
        _first_present(
            promotion.get("max_o_point_count"),
            promotion_diagnostics.get("max_o_point_count"),
            history_stats.get("max_o_point_count"),
        )
    )
    critical_point_checks = {
        "critical_point_counts_present": _check_or_fallback(
            promotion_checks,
            "critical_point_counts_present",
            bool(history_stats.get("has_critical_point_counts")),
        ),
        "x_critical_points_detected": _check_or_fallback(
            promotion_checks,
            "x_critical_points_detected",
            max_x_point_count is not None and max_x_point_count > 0,
        ),
        "o_critical_points_detected": max_o_point_count is not None and max_o_point_count > 0,
    }

    min_flux_amplification = float(
        thresholds.get("min_reconnected_flux_amplification", DEFAULT_MIN_RESPONSE_AMPLIFICATION)
    )
    reconnected_flux_amplification = _as_float(
        _first_present(
            promotion.get("reconnected_flux_amplification"),
            promotion_diagnostics.get("reconnected_flux_amplification"),
            history_stats.get("reconnected_flux_amplification"),
        )
    )
    flux_checks = {
        "reconnected_flux_history_present": bool(history_stats.get("has_reconnected_flux"))
        or reconnected_flux_amplification is not None,
        "reconnected_flux_amplifies": _check_or_fallback(
            promotion_checks,
            "reconnected_flux_amplifies",
            reconnected_flux_amplification is not None
            and reconnected_flux_amplification >= min_flux_amplification,
        ),
    }

    min_island_width_amplification = float(
        thresholds.get("min_island_width_amplification", DEFAULT_MIN_RESPONSE_AMPLIFICATION)
    )
    island_width_amplification = _as_float(
        _first_present(
            promotion.get("island_width_amplification"),
            promotion_diagnostics.get("island_width_amplification"),
            history_stats.get("island_width_amplification"),
        )
    )
    island_width_checks = {
        "island_width_history_present": bool(history_stats.get("has_island_width"))
        or island_width_amplification is not None,
        "island_width_amplifies": _check_or_fallback(
            promotion_checks,
            "island_width_amplifies",
            island_width_amplification is not None
            and island_width_amplification >= min_island_width_amplification,
        ),
    }

    movie_paths = {
        "flux_movie": (
            root / "figures" / "fixed_scale_flux_movie.gif",
            root / "figures" / "periodic_double_harris_flux.gif",
        ),
        "current_density_movie": (
            root / "figures" / "fixed_scale_current_density_movie.gif",
            root / "figures" / "periodic_double_harris_current.gif",
        ),
    }
    selected_movie_paths = {
        name: next((path for path in paths if path.exists()), paths[0])
        for name, paths in movie_paths.items()
    }
    movie_checks = {
        "flux_movie_present": (not require_movies)
        or any(_file_is_present(path) for path in movie_paths["flux_movie"]),
        "current_density_movie_present": (not require_movies)
        or any(_file_is_present(path) for path in movie_paths["current_density_movie"]),
    }

    manifest_hash_names = set(manifest_hashes)
    manifest_hash_paths = {
        str(relative_path) for relative_path in manifest_outputs.values() if relative_path
    }
    manifest_hashes_manifest = bool(manifest_hashes)
    manifest_hashes_validation = (
        "validation" in manifest_hash_names
        or "validation.json" in manifest_hash_paths
        or "validation.json" in artifact_files
    )
    manifest_hashes_diagnostics = (
        "diagnostics" in manifest_hash_names
        or "diagnostics.json" in manifest_hash_paths
        or "diagnostics.json" in artifact_files
    )
    manifest_hashes_history = (
        "history" in manifest_hash_names
        or history_relative_path in manifest_hash_paths
        or history_relative_path in artifact_files
    )
    promotion_artifact_manifest = read_json(root / "promotion" / "artifact_manifest.json")
    promotion_artifact_files = _artifact_manifest_files(promotion_artifact_manifest)
    manifest_checks = {
        "run_manifest_present": bool(manifest),
        "execution_validation_present": bool(validation),
        "execution_validation_passed": bool(validation.get("passed")),
        "execution_diagnostics_present": bool(diagnostics),
        "run_manifest_hashes_outputs": manifest_hashes_manifest,
        "manifest_or_artifact_hashes_validation": manifest_hashes_validation,
        "manifest_or_artifact_hashes_diagnostics": manifest_hashes_diagnostics,
        "manifest_or_artifact_hashes_history": manifest_hashes_history,
        "artifact_manifest_schema_supported_when_present": (not artifact_manifest)
        or artifact_manifest.get("schema") == ARTIFACT_MANIFEST_SCHEMA,
        "promotion_validation_present": bool(promotion_sources["promotion_validation_path"]),
        "promotion_artifact_manifest_valid_when_present": (not promotion_artifact_manifest)
        or (
            promotion_artifact_manifest.get("schema") == ARTIFACT_MANIFEST_SCHEMA
            and {"manifest.json", "validation.json"}.issubset(promotion_artifact_files)
        ),
    }

    gates = {
        "duration": _gate(
            duration_checks,
            {
                "target_step": target_step,
                "terminal_step": terminal_step,
                "terminal_time": terminal_time,
                "min_t_end": min_t_end,
                "duration_assessment": duration_assessment,
            },
        ),
        "convergence": _gate(
            convergence_checks,
            {
                "min_convergence_dirs": min_convergence_dirs,
                "convergence_report_count": len(convergence_reports),
                "convergence_reports": convergence_reports,
            },
        ),
        "seed_qi": _gate(
            seed_qi_checks,
            {
                "seed_qi_required": seed_qi_required,
                "claim_level_if_gates_pass": claim_level_if_gates_pass,
                "seed_qi_report": seed_qi_report,
            },
        ),
        "critical_points": _gate(
            critical_point_checks,
            {
                "max_x_point_count": max_x_point_count,
                "max_o_point_count": max_o_point_count,
            },
        ),
        "flux": _gate(
            flux_checks,
            {
                "reconnected_flux_amplification": reconnected_flux_amplification,
                "min_reconnected_flux_amplification": min_flux_amplification,
            },
        ),
        "island_width": _gate(
            island_width_checks,
            {
                "island_width_amplification": island_width_amplification,
                "min_island_width_amplification": min_island_width_amplification,
            },
        ),
        "movies": _gate(
            movie_checks,
            {
                "require_movies": require_movies,
                "flux_movie": str(selected_movie_paths["flux_movie"]),
                "current_density_movie": str(selected_movie_paths["current_density_movie"]),
            },
        ),
        "manifest": _gate(
            manifest_checks,
            {
                "artifact_manifest_file_count": len(artifact_files),
                "promotion_artifact_manifest_file_count": len(promotion_artifact_files),
                "manifest_claim_level": manifest.get("claim_level"),
                "history_path": history_relative_path,
            },
        ),
    }
    failed_gates = [
        gate_name for gate_name in PROMOTION_GATE_ORDER if not gates[gate_name]["passed"]
    ]
    gate_ready = not failed_gates
    existing_claim_level = manifest.get("claim_level", "missing")
    production_claim_ready = gate_ready and claim_level_if_gates_pass == "production"
    return {
        "schema": GATE_SUMMARY_SCHEMA,
        "campaign_dir": str(root),
        "gate_ready": gate_ready,
        "production_claim_ready": production_claim_ready,
        "claim_level_if_gates_pass": claim_level_if_gates_pass,
        "promotable_claim_level": claim_level_if_gates_pass if gate_ready else "validation",
        "existing_claim_level": existing_claim_level,
        "existing_claim_level_consistent": existing_claim_level != "production"
        or production_claim_ready,
        "claim_boundary": _first_present(
            promotion.get("claim_boundary"),
            promotion_diagnostics.get("claim_boundary"),
            "A campaign is promotable only when duration, convergence, seed-QI, X/O "
            "critical-point, reconnecting-flux, island-width, fixed-scale movie, and "
            "manifest gates all pass.",
        ),
        "failed_gates": failed_gates,
        "gates": gates,
        "artifacts": {
            "campaign_plan": str(root / "campaign_plan.json"),
            "diagnostics": str(root / "diagnostics.json"),
            "validation": str(root / "validation.json"),
            "manifest": str(root / "manifest.json"),
            "artifact_manifest": str(root / "artifact_manifest.json"),
            "history": str(history_path),
            "promotion_readiness": promotion_sources["promotion_path"],
            "promotion_validation": promotion_sources["promotion_validation_path"],
        },
    }


def _format_failed_checks(gate: dict[str, Any]) -> str:
    blockers = gate.get("blockers", [])
    if not blockers:
        return "none"
    return ", ".join(f"`{blocker}`" for blocker in blockers)


def _format_gate_evidence(gate: dict[str, Any]) -> str:
    evidence = gate.get("evidence", {})
    if not isinstance(evidence, dict) or not evidence:
        return "n/a"
    interesting_items = []
    for key, value in evidence.items():
        if value in (None, {}, []):
            continue
        if key.endswith("reports"):
            interesting_items.append(f"`{key}`={len(value)}")
        elif key == "duration_assessment":
            if isinstance(value, dict):
                interesting_items.append(
                    f"`duration_assessment.t_end`={compact_value(value.get('t_end'))}"
                )
        elif isinstance(value, dict):
            if "passed" in value:
                interesting_items.append(f"`{key}.passed`={compact_value(value.get('passed'))}")
        else:
            interesting_items.append(f"`{key}`={compact_value(value)}")
    return ", ".join(interesting_items) if interesting_items else "n/a"


def write_campaign_gate_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Nonlinear campaign promotion gate summary",
        "",
        f"Campaign directory: `{summary['campaign_dir']}`",
        "",
        f"Gate ready: **{summary['gate_ready']}**",
        f"Production claim ready: **{summary['production_claim_ready']}**",
        f"Claim level if gates pass: `{summary['claim_level_if_gates_pass']}`",
        f"Promotable claim level: `{summary['promotable_claim_level']}`",
        f"Existing manifest claim level: `{summary['existing_claim_level']}`",
        "",
        summary["claim_boundary"],
        "",
        "| Gate | Status | Failed checks | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    gates = summary["gates"]
    for gate_name in PROMOTION_GATE_ORDER:
        gate = gates[gate_name]
        status = "pass" if gate["passed"] else "fail"
        failed_checks = _format_failed_checks(gate)
        evidence = _format_gate_evidence(gate)
        lines.append(
            f"| {PROMOTION_GATE_LABELS[gate_name]} | {status} | {failed_checks} | {evidence} |"
        )
    if summary["failed_gates"]:
        lines.extend(
            [
                "",
                "## Blockers",
                "",
                *[
                    f"- `{gate_name}`: {_format_failed_checks(gates[gate_name])}"
                    for gate_name in summary["failed_gates"]
                ],
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_campaign_gate_summary(
    campaign_dir: str | Path,
    *,
    output_json: str | Path,
    output_md: str | Path,
    require_movies: bool = True,
) -> dict[str, Any]:
    summary = summarize_campaign_dir(campaign_dir, require_movies=require_movies)
    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_campaign_gate_markdown(summary, Path(output_md))
    return summary


def collect_case(evidence_case: EvidenceCase) -> dict[str, Any]:
    manifest = read_json(evidence_case.run_dir / "manifest.json")
    diagnostics = read_json(evidence_case.run_dir / "diagnostics.json")
    validation = read_json(evidence_case.run_dir / "validation.json")
    promotion = (
        read_json(evidence_case.run_dir / evidence_case.promotion_path)
        if evidence_case.promotion_path is not None
        else {}
    )
    metrics = {
        key: compact_value(diagnostics[key])
        for key in evidence_case.metric_keys
        if key in diagnostics
    }
    for key in (
        "promotion_ready",
        "reconnected_flux_amplification",
        "island_width_amplification",
        "max_x_point_count",
        "max_o_point_count",
        "terminal_step",
        "target_step",
        "history_sample_count",
    ):
        if key in promotion and key not in metrics:
            metrics[key] = compact_value(promotion[key])
    return {
        "label": evidence_case.label,
        "family": evidence_case.family,
        "run_dir": str(evidence_case.run_dir),
        "present": (evidence_case.run_dir / "manifest.json").exists(),
        "command": evidence_case.command,
        "claim_level": manifest.get("claim_level", "missing"),
        "claim_scope": manifest.get("claim_scope", ""),
        "validation_passed": validation.get("passed"),
        "promotion_ready": promotion.get("promotion_ready", False),
        "production_claim_ready": False,
        "metrics": metrics,
        "checks": validation.get("checks", {}),
        "promotion_checks": promotion.get("checks", {}),
        "blocker": evidence_case.blocker,
        "artifacts": {
            "manifest": str(evidence_case.run_dir / "manifest.json"),
            "diagnostics": str(evidence_case.run_dir / "diagnostics.json"),
            "validation": str(evidence_case.run_dir / "validation.json"),
            "promotion_readiness": (
                str(evidence_case.run_dir / evidence_case.promotion_path)
                if evidence_case.promotion_path is not None
                else None
            ),
        },
    }


def format_metric_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "missing"
    parts = []
    for key, value in metrics.items():
        parts.append(f"`{key}`={value}")
    return ", ".join(parts)


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Nonlinear campaign evidence claims",
        "",
        "This generated table records the local nonlinear campaign evidence inspected for the",
        "double-Harris, Rutherford, and forced turbulent-reconnection lanes. It is deliberately",
        "conservative: every row remains below production physics claim level.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python tools/nonlinear_campaign_evidence.py \\",
        "  --output-json docs/project/nonlinear_campaign_evidence.json \\",
        "  --output-md docs/project/nonlinear_campaign_evidence.md",
        "```",
        "",
        "Summarize one candidate campaign bundle with:",
        "",
        "```bash",
        "python tools/nonlinear_campaign_evidence.py \\",
        "  --campaign-dir outputs/campaigns/rutherford_production \\",
        "  --output-json outputs/campaigns/rutherford_production/promotion_gate_summary.json \\",
        "  --output-md outputs/campaigns/rutherford_production/promotion_gate_summary.md \\",
        "  --require-production-ready",
        "```",
        "",
        "The per-campaign summary fails closed: `gate_ready` is true only if the duration,",
        "convergence, seed-QI, X/O critical-point, reconnecting-flux, island-width,",
        "fixed-scale movie, and manifest gates all pass. `production_claim_ready` also",
        'requires the upstream promotion report to allow `claim_level_if_passed = "production"`.',
        "The X/O gate requires both detected X points and detected O points rather than only",
        "the presence of count arrays.",
        "",
        "Seeded double-Harris long-run bundles are supported as validation-promotion inputs:",
        "the summarizer reads `periodic_double_harris_seeded_long_run.npz`,",
        "`figures/periodic_double_harris_flux.gif`, and",
        "`figures/periodic_double_harris_current.gif` when those files are present. These",
        "bundles can be `gate_ready` for their declared validation media claim while still",
        "leaving `production_claim_ready = false`.",
        "",
        "## Claim table",
        "",
        "| Lane | Artifact | Validation | Readiness gate | Key metrics | Production blocker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["cases"]:
        validation = "pass" if item["validation_passed"] else "fail/missing"
        promotion = "validation-ready" if item["promotion_ready"] else "not production-ready"
        metrics = format_metric_summary(item["metrics"])
        lines.append(
            (
                "| {family} | `{run_dir}` | {validation} | {promotion} | {metrics} | {blocker} |"
            ).format(
                family=item["family"],
                run_dir=item["run_dir"],
                validation=validation,
                promotion=promotion,
                metrics=metrics,
                blocker=item["blocker"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Passing double-Harris rows support validation-level nonlinear response and "
            "convergence scaffolding, not Rutherford/plasmoid production physics.",
            "- Passing Rutherford rows support executor/schema/duration mechanics unless "
            "the promotion report passes with positive response, convergence, seed-QI, "
            "geometry, and media gates.",
            "- Passing forced turbulent-reconnection rows support 2-D reduced-MHD proxy-media "
            "readiness only, not 3-D turbulent-reconnection or LV99 scaling claims.",
            "- Large binary outputs remain under `outputs/`, which is git-ignored; this page "
            "and the JSON summary are the small review artifacts.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    cases = [collect_case(evidence_case) for evidence_case in CASES]
    return {
        "schema": SCHEMA,
        "refresh_date": "2026-05-22",
        "claim_boundary": (
            "All listed nonlinear campaign artifacts are validation evidence; none support "
            "production Rutherford, plasmoid, Sweet-Parker, or 3-D turbulent-reconnection "
            "claims."
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output_json = "docs/project/nonlinear_campaign_evidence.json"
    default_output_md = "docs/project/nonlinear_campaign_evidence.md"
    parser.add_argument("--campaign-dir", type=Path, help="Summarize one campaign directory.")
    parser.add_argument("--output-json", default=default_output_json)
    parser.add_argument("--output-md", default=default_output_md)
    parser.add_argument(
        "--no-require-movies",
        dest="require_movies",
        action="store_false",
        help="Do not fail the gate summary when fixed-scale movies are missing.",
    )
    parser.add_argument(
        "--require-production-ready",
        action="store_true",
        help="Exit nonzero after writing the summary unless all promotion gates pass.",
    )
    args = parser.parse_args()

    if args.campaign_dir is not None:
        output_json = Path(args.output_json)
        output_md = Path(args.output_md)
        if args.output_json == default_output_json:
            output_json = args.campaign_dir / "promotion_gate_summary.json"
        if args.output_md == default_output_md:
            output_md = args.campaign_dir / "promotion_gate_summary.md"
        summary = write_campaign_gate_summary(
            args.campaign_dir,
            output_json=output_json,
            output_md=output_md,
            require_movies=args.require_movies,
        )
        print(f"wrote {output_json}")
        print(f"wrote {output_md}")
        if args.require_production_ready and not summary["production_claim_ready"]:
            raise SystemExit(1)
        return

    report = build_report()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, Path(args.output_md))
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
