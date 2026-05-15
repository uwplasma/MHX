from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    DURATION_POLICY_SCHEMA,
    assess_duration,
    duration_policy_assessments,
    require_duration_for_claim,
    required_time_for_efolds,
    write_duration_policy,
)
from mhx.cli.main import app


def test_duration_policy_marks_short_runs_as_validation_only() -> None:
    assessments = duration_policy_assessments()
    by_name = {assessment.name: assessment for assessment in assessments}

    assert by_name["linear_tearing_fast"].sufficient_for_intended_scope is True
    assert by_name["linear_tearing_fast"].sufficient_for_nonlinear_claim is False
    assert by_name["nonlinear_energy_budget"].scope == "nonlinear_identity_gate"
    assert by_name["nonlinear_energy_budget"].sufficient_for_nonlinear_claim is False
    assert by_name["double_harris_ci_fast_movie"].scope == "smoke"
    assert by_name["double_harris_ci_fast_movie"].sufficient_for_nonlinear_claim is False
    assert by_name["double_harris_readme_release_media"].scope == "validation_media"
    assert by_name["double_harris_readme_release_media"].t_end > 30.0
    assert (
        by_name["double_harris_readme_release_media"].sufficient_for_nonlinear_claim
        is False
    )
    assert by_name["future_harris_linear_growth_campaign"].sufficient_for_production_claim is True
    assert by_name["future_harris_linear_growth_campaign"].sufficient_for_nonlinear_claim is False
    assert by_name["future_rutherford_island_campaign"].sufficient_for_nonlinear_claim is True
    assert by_name["future_plasmoid_linear_onset_campaign"].sufficient_for_nonlinear_claim is True


def test_duration_policy_raises_for_too_short_production_claims() -> None:
    required = required_time_for_efolds(1.31e-2, required_efolds=10.0)
    assert required == pytest.approx(10.0 / 1.31e-2)

    short = assess_duration(
        name="bad_production",
        purpose="nonlinear island claim",
        t_end=0.8,
    )
    assert short.sufficient_for_intended_scope is False
    assert "increase t_end" in short.action
    with pytest.raises(ValueError, match="too short"):
        require_duration_for_claim(
            name="bad_production",
            purpose="nonlinear island claim",
            t_end=0.8,
        )
    long_enough = require_duration_for_claim(
        name="good_production",
        purpose="nonlinear island claim",
        t_end=required,
    )
    assert long_enough.sufficient_for_production_claim is True
    assert long_enough.sufficient_for_nonlinear_claim is True


def test_duration_policy_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_duration_policy(tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert (tmp_path / "duration_policy.json").exists()
    assert (tmp_path / "duration_policy.md").exists()
    assert (tmp_path / "validation.json").exists()
    policy = json.loads((tmp_path / "duration_policy.json").read_text())
    assert policy["schema"] == DURATION_POLICY_SCHEMA
    assert "future_rutherford_island_campaign" in {
        item["name"] for item in policy["assessments"]
    }

    outdir = tmp_path / "cli"
    result = CliRunner().invoke(app, ["benchmark", "duration-policy", "--outdir", str(outdir)])
    assert result.exit_code == 0, result.stdout
    assert (outdir / "duration_policy.json").exists()
