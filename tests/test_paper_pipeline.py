from __future__ import annotations

import json

from typer.testing import CliRunner

from mhx.benchmarks import (
    PAPER_PIPELINE_GATES_SCHEMA,
    PAPER_PIPELINE_SCHEMA,
    write_paper_pipeline,
)
from mhx.cli.main import app


def test_paper_pipeline_subset_writes_manifest_and_checksums(tmp_path) -> None:
    result = write_paper_pipeline(
        tmp_path / "paper",
        case_names=("resistive_decay", "fkr_window"),
        require_release_ready=False,
    )

    assert result.validation["schema"] == PAPER_PIPELINE_GATES_SCHEMA
    assert result.validation["passed"] is True
    assert result.pipeline_path == tmp_path / "paper" / "paper_pipeline.json"
    assert result.pipeline_path.exists()
    assert result.manifest_path.exists()
    assert result.artifact_manifest_path.exists()
    diagnostics = json.loads(result.pipeline_path.read_text(encoding="utf-8"))
    assert diagnostics["schema"] == PAPER_PIPELINE_SCHEMA
    assert diagnostics["case_names"] == ["resistive_decay", "fkr_window"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["claim_level"] == "validation"
    assert manifest["outputs"]["validation_suite"] == "validation_suite/validation_suite.json"
    artifact_manifest = json.loads(result.artifact_manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {record["path"] for record in artifact_manifest["files"]}
    assert {
        "manifest.json",
        "paper_pipeline.json",
        "validation_suite/validation_suite.json",
        "validation_suite/manifest.json",
        "readiness/readiness.json",
    } <= artifact_paths
    assert artifact_manifest["claim_levels"]["manifest.json"] == "validation"
    assert (
        artifact_manifest["claim_levels"]["validation_suite/manifest.json"]
        == "validation"
    )


def test_paper_pipeline_cli_subset(tmp_path) -> None:
    outdir = tmp_path / "cli-paper"
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "paper-pipeline",
            "--outdir",
            str(outdir),
            "--cases",
            "resistive_decay,fkr_window",
            "--no-require-release-ready",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (outdir / "paper_pipeline.json").exists()
    assert (outdir / "artifact_manifest.json").exists()
