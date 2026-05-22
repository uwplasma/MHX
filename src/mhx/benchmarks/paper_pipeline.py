"""Reproducible paper/validation pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mhx.benchmarks.readiness import write_readiness_report
from mhx.benchmarks.suite import ValidationSuiteCase, validation_suite_cases, write_validation_suite
from mhx.io import write_artifact_manifest, write_manifest
from mhx.versioning import require_supported_api_version

PAPER_PIPELINE_SCHEMA = "mhx.paper_pipeline.v1"
PAPER_PIPELINE_GATES_SCHEMA = "mhx.paper_pipeline.gates.v1"


@dataclass(frozen=True)
class PaperPipelineResult:
    """Summary of one reproducible paper/validation pipeline bundle."""

    outdir: Path
    validation_suite_path: Path
    readiness_path: Path
    pipeline_path: Path
    manifest_path: Path
    artifact_manifest_path: Path
    validation: dict[str, Any]


def write_paper_pipeline(
    outdir: str | Path,
    *,
    case_names: tuple[str, ...] | None = None,
    require_release_ready: bool = True,
) -> PaperPipelineResult:
    """Generate validation figures, readiness reports, manifests, and checksums.

    With ``case_names=None`` this is the full deterministic FAST paper/release
    pipeline. Tests and local previews can pass a subset and set
    ``require_release_ready=False``; the resulting manifest is still useful, but
    it is explicitly marked as an incomplete validation bundle.
    """
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_version = require_supported_api_version(context="paper pipeline")
    selected_cases = _select_cases(case_names)

    suite_dir = output_dir / "validation_suite"
    validation_suite_path, suite = write_validation_suite(suite_dir, cases=selected_cases)
    readiness_dir = output_dir / "readiness"
    readiness_path, readiness_validation = write_readiness_report(readiness_dir, suite)

    suite_passed = bool(suite["passed"])
    readiness_passed = bool(readiness_validation["passed"])
    release_gate_ok = readiness_passed or not require_release_ready
    validation = {
        "schema": PAPER_PIPELINE_GATES_SCHEMA,
        "passed": suite_passed and release_gate_ok,
        "checks": {
            "validation_suite_passed": suite_passed,
            "readiness_report_generated": readiness_path.exists(),
            "release_ready_when_required": release_gate_ok,
            "case_subset_declared": case_names is None or len(case_names) > 0,
        },
        "diagnostics": {
            "schema": PAPER_PIPELINE_SCHEMA,
            "api_version": api_version,
            "case_count": suite["case_count"],
            "case_names": [case["name"] for case in suite["cases"]],
            "require_release_ready": require_release_ready,
            "public_release_ready": readiness_validation["diagnostics"]["public_release_ready"],
            "publication_claim_ready": readiness_validation["diagnostics"][
                "publication_claim_ready"
            ],
            "claim_level": "validation",
            "claim_scope": (
                "Deterministic FAST paper/validation bundle. Publication-level nonlinear "
                "claims require separately promoted production evidence."
            ),
        },
    }

    pipeline_path = output_dir / "paper_pipeline.json"
    validation_path = output_dir / "validation.json"
    markdown_path = output_dir / "paper_pipeline.md"
    manifest_path = output_dir / "manifest.json"
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    pipeline_path.write_text(
        json.dumps(validation["diagnostics"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_pipeline_markdown(validation), encoding="utf-8")
    write_manifest(
        manifest_path,
        config=validation["diagnostics"],
        outputs={
            "pipeline": pipeline_path.name,
            "validation": validation_path.name,
            "pipeline_markdown": markdown_path.name,
            "validation_suite": validation_suite_path.relative_to(output_dir).as_posix(),
            "readiness": readiness_path.relative_to(output_dir).as_posix(),
            "artifact_manifest": artifact_manifest_path.name,
        },
        claim_level="validation",
        claim_scope=str(validation["diagnostics"]["claim_scope"]),
    )
    write_artifact_manifest(output_dir, path=artifact_manifest_path)
    return PaperPipelineResult(
        outdir=output_dir,
        validation_suite_path=validation_suite_path,
        readiness_path=readiness_path,
        pipeline_path=pipeline_path,
        manifest_path=manifest_path,
        artifact_manifest_path=artifact_manifest_path,
        validation=validation,
    )


def _select_cases(case_names: tuple[str, ...] | None) -> tuple[ValidationSuiteCase, ...]:
    cases = validation_suite_cases()
    if case_names is None:
        return cases
    by_name = {case.name: case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        requested = ", ".join(missing)
        raise ValueError(f"unknown validation-suite case(s): {requested}; available: {available}")
    return tuple(by_name[name] for name in case_names)


def _pipeline_markdown(validation: dict[str, Any]) -> str:
    diagnostics = validation["diagnostics"]
    checks = validation["checks"]
    status = "passed" if validation["passed"] else "blocked"
    rows = "\n".join(
        f"| `{name}` | validation |" for name in diagnostics["case_names"]
    )
    check_rows = "\n".join(
        f"| `{name}` | `{value}` |" for name, value in checks.items()
    )
    return (
        "# MHX paper/validation pipeline\n\n"
        f"Pipeline status: **{status}**\n\n"
        "## Gates\n\n"
        "| Gate | Value |\n"
        "| --- | --- |\n"
        f"{check_rows}\n\n"
        "## Included validation cases\n\n"
        "| Case | Claim level |\n"
        "| --- | --- |\n"
        f"{rows}\n\n"
        "## Claim boundary\n\n"
        f"{diagnostics['claim_scope']}\n"
    )
