from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    READINESS_REPORT_SCHEMA,
    REQUIRED_PUBLIC_RELEASE_CASES,
    run_readiness_assessment,
    write_readiness_report,
)
from mhx.cli.main import app


def _suite_payload() -> dict:
    return {
        "schema": "mhx.validation_suite.v1",
        "passed": True,
        "case_count": len(REQUIRED_PUBLIC_RELEASE_CASES),
        "cases": [
            {
                "name": name,
                "claim_level": "validation",
                "passed": True,
                "schema": f"schema.{name}",
                "checks": {"dummy": True},
            }
            for name in REQUIRED_PUBLIC_RELEASE_CASES
        ],
    }


def test_readiness_assessment_distinguishes_release_from_paper_claims() -> None:
    assessment = run_readiness_assessment(_suite_payload())

    assert assessment.diagnostics["schema"] == READINESS_REPORT_SCHEMA
    assert assessment.public_release_ready is True
    assert assessment.publication_claim_ready is False
    assert assessment.validation["passed"] is True
    assert assessment.validation["checks"]["publication_claims_blocked_until_production_campaigns"]


def test_readiness_assessment_catches_missing_and_failed_cases() -> None:
    payload = _suite_payload()
    payload["cases"] = payload["cases"][:-1]
    payload["case_count"] = len(payload["cases"])
    payload["cases"][0]["passed"] = False

    assessment = run_readiness_assessment(payload)

    assert assessment.public_release_ready is False
    assert assessment.validation["passed"] is False
    assert assessment.diagnostics["missing_required_cases"] == [REQUIRED_PUBLIC_RELEASE_CASES[-1]]
    assert assessment.diagnostics["failed_required_cases"] == [REQUIRED_PUBLIC_RELEASE_CASES[0]]


def test_write_readiness_report_and_cli(tmp_path) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "validation_suite.json").write_text(
        json.dumps(_suite_payload()),
        encoding="utf-8",
    )
    diagnostics_path, validation = write_readiness_report(tmp_path / "readiness", suite_dir)

    assert diagnostics_path == tmp_path / "readiness" / "readiness.json"
    assert validation["passed"] is True
    assert (tmp_path / "readiness" / "readiness.md").stat().st_size > 0
    assert (tmp_path / "readiness" / "figures" / "readiness_matrix.png").stat().st_size > 0
    assert (tmp_path / "readiness" / "manifest.json").exists()

    outdir = tmp_path / "cli-readiness"
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "readiness",
            "--suite",
            str(suite_dir),
            "--outdir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "readiness.json").exists()


def test_readiness_loader_rejects_missing_or_invalid_json(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        run_readiness_assessment(tmp_path / "missing")

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        run_readiness_assessment(invalid_path)
