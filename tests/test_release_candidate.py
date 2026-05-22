from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mhx.benchmarks import (
    RELEASE_CANDIDATE_SCHEMA,
    run_release_candidate_assessment,
    write_release_candidate_report,
)
from mhx.cli.main import app

ROOT = Path(__file__).resolve().parents[1]


def _ready_payload() -> dict[str, object]:
    return {
        "schema": "mhx.readiness_report.v1",
        "public_release_ready": True,
        "publication_claim_ready": False,
    }


def test_release_candidate_static_repo_gate_passes() -> None:
    assessment = run_release_candidate_assessment(ROOT)

    assert assessment.diagnostics["schema"] == RELEASE_CANDIDATE_SCHEMA
    assert assessment.release_candidate_ready is True
    assert assessment.publication_claim_ready is False
    assert assessment.validation["passed"] is True
    assert assessment.diagnostics["checks"]["figure_manifest"] is True
    assert assessment.diagnostics["checks"]["metadata"] is True


def test_release_candidate_can_require_readiness_payload() -> None:
    ready = run_release_candidate_assessment(
        ROOT,
        readiness=_ready_payload(),
        require_readiness=True,
    )
    assert ready.release_candidate_ready is True

    blocked_payload = _ready_payload()
    blocked_payload["public_release_ready"] = False
    blocked = run_release_candidate_assessment(
        ROOT,
        readiness=blocked_payload,
        require_readiness=True,
    )

    assert blocked.release_candidate_ready is False
    assert blocked.validation["passed"] is False
    assert (
        blocked.diagnostics["check_groups"]["readiness"]["public_release_ready_when_supplied"]
        is False
    )


def test_write_release_candidate_report_and_cli(tmp_path) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(_ready_payload()), encoding="utf-8")

    diagnostics_path, validation = write_release_candidate_report(
        tmp_path / "release",
        repo_root=ROOT,
        readiness=readiness_path,
        require_readiness=True,
    )

    assert diagnostics_path == tmp_path / "release" / "release_candidate.json"
    assert validation["passed"] is True
    assert (tmp_path / "release" / "release_candidate.md").stat().st_size > 0
    assert (tmp_path / "release" / "manifest.json").exists()

    outdir = tmp_path / "cli-release"
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "release-candidate",
            "--repo-root",
            str(ROOT),
            "--outdir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "release_candidate.json").exists()
