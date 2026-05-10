from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA,
    build_rutherford_campaign_template,
    write_rutherford_campaign_template,
)
from mhx.cli.main import app


def test_rutherford_campaign_template_is_duration_guarded() -> None:
    result = build_rutherford_campaign_template(
        harris_growth_rate=1.31e-2,
        production_efolds=10.0,
        safety_factor=3.0,
        shape=(96, 96),
        dt=0.2,
        target_saved_frames=200,
    )

    assert result.diagnostics["schema"] == RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA
    assert result.validation["passed"] is True
    assert result.duration_assessment.sufficient_for_nonlinear_claim is True
    assert result.config.time.t1 >= 3.0 * 10.0 / 1.31e-2
    assert result.estimated_saved_frames >= 100
    assert result.config.physics.model == "reduced_mhd_nonlinear_tearing_campaign"


def test_rutherford_campaign_template_artifacts_and_cli(tmp_path) -> None:
    manifest_path, validation = write_rutherford_campaign_template(
        tmp_path,
        shape=(96, 96),
        dt=0.2,
        target_saved_frames=200,
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    manifest = json.loads(manifest_path.read_text())
    assert manifest["claim_level"] == "production_template"
    assert (tmp_path / "campaign.json").exists()
    assert (tmp_path / "campaign_config.toml").exists()
    assert (tmp_path / "duration_assessment.json").exists()

    outdir = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        [
            "campaign",
            "rutherford-template",
            "--outdir",
            str(outdir),
            "--nx",
            "96",
            "--ny",
            "96",
            "--dt",
            "0.2",
            "--target-saved-frames",
            "200",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "manifest.json").exists()


def test_rutherford_campaign_template_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        build_rutherford_campaign_template(dt=0.0)
    with pytest.raises(ValueError, match="target_saved_frames"):
        build_rutherford_campaign_template(target_saved_frames=4)


def test_campaign_config_is_not_silently_run_by_fast_runner(tmp_path) -> None:
    write_rutherford_campaign_template(
        tmp_path,
        shape=(96, 96),
        dt=0.2,
        target_saved_frames=200,
    )
    result = CliRunner().invoke(app, ["run", str(tmp_path / "campaign_config.toml")])
    assert result.exit_code != 0
    assert "FAST smoke runner" in result.output
