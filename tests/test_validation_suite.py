from __future__ import annotations

import json

from typer.testing import CliRunner

from mhx.benchmarks import (
    VALIDATION_SUITE_SCHEMA,
    validation_suite_cases,
    write_validation_suite,
)
from mhx.cli.main import app


def test_validation_suite_cases_are_unique() -> None:
    names = [case.name for case in validation_suite_cases()]
    assert len(names) == len(set(names))
    assert "linear_tearing_fast" in names
    assert "cosine_equilibrium_linearization" in names


def test_write_validation_suite_artifacts_and_cli(tmp_path) -> None:
    summary_path, summary = write_validation_suite(tmp_path / "suite")
    assert summary_path == tmp_path / "suite" / "validation_suite.json"
    assert summary["schema"] == VALIDATION_SUITE_SCHEMA
    assert summary["passed"] is True
    assert summary["jax_enable_x64"] is True
    assert summary["case_count"] == len(validation_suite_cases())
    assert (tmp_path / "suite" / "validation_suite.md").stat().st_size > 0
    assert (tmp_path / "suite" / "artifact_manifest.json").exists()
    assert (tmp_path / "suite" / "manifest.json").exists()
    assert (
        tmp_path
        / "suite"
        / "cosine_equilibrium_linearization"
        / "figures"
        / "cosine_equilibrium_linearization_errors.png"
    ).stat().st_size > 0
    persisted = json.loads((tmp_path / "suite" / "validation_suite.json").read_text())
    assert persisted["cases"][0]["passed"] is True

    outdir = tmp_path / "cli-suite"
    cli_result = CliRunner().invoke(app, ["validate", "all", "--outdir", str(outdir)])
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation_suite.json").exists()
