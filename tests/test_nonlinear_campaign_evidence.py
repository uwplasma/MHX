from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_evidence_tool():
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "nonlinear_campaign_evidence.py"
    spec = importlib.util.spec_from_file_location("nonlinear_campaign_evidence_tool", tool_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_manifest(
    root: Path,
    *,
    history_name: str,
    flux_movie: str,
    current_movie: str,
) -> None:
    _write_json(
        root / "manifest.json",
        {
            "schema": "mhx.manifest.v1",
            "claim_level": "validation",
            "outputs": {
                "diagnostics": "diagnostics.json",
                "validation": "validation.json",
                "history": history_name,
                "flux_movie": flux_movie,
                "current_movie": current_movie,
            },
            "hashes": {
                "diagnostics": "hash",
                "validation": "hash",
                "history": "hash",
                "flux_movie": "hash",
                "current_movie": "hash",
            },
        },
    )


def _write_promotion_artifacts(root: Path, readiness: dict) -> None:
    _write_json(root / "promotion" / "promotion_readiness.json", readiness)
    _write_json(
        root / "promotion" / "validation.json",
        {
            "schema": readiness["validation_schema"],
            "passed": readiness["promotion_ready"],
            "checks": readiness["checks"],
            "diagnostics": readiness,
            "thresholds": readiness["thresholds"],
        },
    )
    _write_json(
        root / "promotion" / "artifact_manifest.json",
        {
            "schema": "mhx.artifacts.v1",
            "files": [
                {"path": "manifest.json"},
                {"path": "validation.json"},
                {"path": "promotion_readiness.json"},
            ],
        },
    )


def _write_rutherford_campaign(root: Path, *, max_o_point_count: int = 1) -> None:
    _write_json(
        root / "campaign_plan.json",
        {
            "estimated_steps": 2,
            "duration_assessment": {
                "t_end": 2.0,
                "required_t_end": 2.0,
                "sufficient_for_production_claim": True,
            },
        },
    )
    _write_json(root / "diagnostics.json", {"completed_target": True, "end_step": 2})
    _write_json(root / "validation.json", {"passed": True, "checks": {}})
    np.savez(
        root / "production_history.npz",
        schema="mhx.campaign.rutherford_history.v1",
        time=np.array([0.0, 1.0, 2.0]),
        step=np.array([0, 1, 2]),
        reconnected_flux=np.array([1.0, 1.1, 1.2]),
        rutherford_island_width=np.array([1.0, 1.1, 1.3]),
        x_point_count=np.array([0, 1, 1]),
        o_point_count=np.array([0, max_o_point_count, max_o_point_count]),
    )
    figure_dir = root / "figures"
    figure_dir.mkdir()
    (figure_dir / "fixed_scale_flux_movie.gif").write_bytes(b"gif")
    if max_o_point_count > 0:
        (figure_dir / "fixed_scale_current_density_movie.gif").write_bytes(b"gif")
    _write_run_manifest(
        root,
        history_name="production_history.npz",
        flux_movie="figures/fixed_scale_flux_movie.gif",
        current_movie="figures/fixed_scale_current_density_movie.gif",
    )
    _write_json(
        root / "artifact_manifest.json",
        {
            "schema": "mhx.artifacts.v1",
            "files": [
                {"path": "manifest.json"},
                {"path": "validation.json"},
                {"path": "diagnostics.json"},
                {"path": "production_history.npz"},
            ],
        },
    )
    readiness = {
        "schema": "mhx.campaign.rutherford_promotion.v1",
        "validation_schema": "mhx.campaign.rutherford_promotion.gates.v1",
        "promotion_ready": max_o_point_count > 0,
        "claim_level_if_passed": "production",
        "checks": {
            "completed_target": True,
            "terminal_step_reaches_target": True,
            "convergence_bundle_count": True,
            "convergence_bundles_passed": True,
            "seed_qi_bundle_present": True,
            "seed_qi_bundle_passed": True,
            "critical_point_counts_present": True,
            "x_critical_points_detected": True,
            "reconnected_flux_amplifies": True,
            "island_width_amplifies": True,
        },
        "thresholds": {
            "min_convergence_dirs": 2,
            "min_reconnected_flux_amplification": 1.05,
            "min_island_width_amplification": 1.05,
        },
        "target_step": 2,
        "terminal_step": 2,
        "convergence_reports": [{"passed": True}, {"passed": True}],
        "seed_qi_report": {"passed": True},
        "max_x_point_count": 1,
        "max_o_point_count": max_o_point_count,
        "reconnected_flux_amplification": 1.2,
        "island_width_amplification": 1.3,
    }
    _write_promotion_artifacts(root, readiness)


def _write_double_harris_campaign(root: Path) -> None:
    _write_json(root / "diagnostics.json", {"t_end": 120.0})
    _write_json(root / "validation.json", {"passed": True, "checks": {}})
    np.savez(
        root / "periodic_double_harris_seeded_long_run.npz",
        schema="mhx.validation.periodic_double_harris_seeded_long_run.v1",
        time=np.array([0.0, 60.0, 120.0]),
        reconnected_flux=np.array([0.5, 1.5, 3.0]),
        rutherford_island_width=np.array([0.25, 0.5, 1.0]),
        x_point_count=np.array([1, 2, 2]),
        o_point_count=np.array([1, 1, 2]),
    )
    figure_dir = root / "figures"
    figure_dir.mkdir()
    (figure_dir / "periodic_double_harris_flux.gif").write_bytes(b"gif")
    (figure_dir / "periodic_double_harris_current.gif").write_bytes(b"gif")
    _write_run_manifest(
        root,
        history_name="periodic_double_harris_seeded_long_run.npz",
        flux_movie="figures/periodic_double_harris_flux.gif",
        current_movie="figures/periodic_double_harris_current.gif",
    )
    readiness = {
        "schema": "mhx.validation.periodic_double_harris_promotion.v1",
        "validation_schema": "mhx.validation.periodic_double_harris_promotion.gates.v1",
        "promotion_ready": True,
        "claim_level_if_passed": "validation",
        "checks": {
            "minimum_duration_reached": True,
            "convergence_bundle_count": True,
            "convergence_bundles_passed": True,
            "critical_point_counts_present": True,
            "x_critical_points_detected": True,
            "o_critical_points_detected": True,
            "reconnected_flux_amplifies": True,
            "island_width_amplifies": True,
            "fixed_scale_movies_present": True,
        },
        "thresholds": {
            "min_t_end": 30.0,
            "min_convergence_dirs": 1,
            "min_reconnected_flux_amplification": 1.05,
            "min_island_width_amplification": 1.05,
        },
        "terminal_time": 120.0,
        "convergence_reports": [{"passed": True}],
        "max_x_point_count": 2,
        "max_o_point_count": 2,
        "reconnected_flux_amplification": 6.0,
        "island_width_amplification": 4.0,
    }
    _write_promotion_artifacts(root, readiness)


def test_rutherford_summary_requires_all_production_gates(tmp_path) -> None:
    tool = _load_evidence_tool()
    _write_rutherford_campaign(tmp_path)

    summary = tool.write_campaign_gate_summary(
        tmp_path,
        output_json=tmp_path / "gate_summary.json",
        output_md=tmp_path / "gate_summary.md",
    )

    assert summary["gate_ready"] is True
    assert summary["production_claim_ready"] is True
    assert summary["promotable_claim_level"] == "production"
    assert all(summary["gates"][gate_name]["passed"] for gate_name in tool.PROMOTION_GATE_ORDER)
    assert "Fixed-scale movies" in (tmp_path / "gate_summary.md").read_text()


def test_summary_blocks_missing_o_point_and_movie(tmp_path) -> None:
    tool = _load_evidence_tool()
    _write_rutherford_campaign(tmp_path, max_o_point_count=0)

    summary = tool.summarize_campaign_dir(tmp_path)

    assert summary["production_claim_ready"] is False
    assert summary["gates"]["critical_points"]["passed"] is False
    assert summary["gates"]["movies"]["passed"] is False
    assert "o_critical_points_detected" in summary["gates"]["critical_points"]["blockers"]
    assert "current_density_movie_present" in summary["gates"]["movies"]["blockers"]


def test_double_harris_summary_uses_seeded_long_run_artifacts(tmp_path) -> None:
    tool = _load_evidence_tool()
    _write_double_harris_campaign(tmp_path)

    summary = tool.write_campaign_gate_summary(
        tmp_path,
        output_json=tmp_path / "gate_summary.json",
        output_md=tmp_path / "gate_summary.md",
    )

    assert summary["gate_ready"] is True
    assert summary["production_claim_ready"] is False
    assert summary["promotable_claim_level"] == "validation"
    assert summary["gates"]["seed_qi"]["checks"]["seed_qi_not_required_for_declared_claim"]
    assert summary["gates"]["manifest"]["evidence"]["history_path"] == (
        "periodic_double_harris_seeded_long_run.npz"
    )
    movies = summary["gates"]["movies"]["evidence"]
    assert movies["flux_movie"].endswith("figures/periodic_double_harris_flux.gif")
    assert movies["current_density_movie"].endswith("figures/periodic_double_harris_current.gif")
