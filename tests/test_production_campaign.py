from __future__ import annotations

import json

import click
import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.campaigns import (
    PRODUCTION_RUTHERFORD_CHECKPOINT_INDEX_SCHEMA,
    PRODUCTION_RUTHERFORD_EXECUTION_SCHEMA,
    PRODUCTION_RUTHERFORD_HISTORY_SCHEMA,
    PRODUCTION_RUTHERFORD_PLAN_SCHEMA,
    PRODUCTION_RUTHERFORD_PROMOTION_SCHEMA,
    PRODUCTION_RUTHERFORD_RESUME_SCHEMA,
    PRODUCTION_RUTHERFORD_STATE_SCHEMA,
    WalltimePolicy,
    assess_rutherford_production_promotion,
    execute_rutherford_production_campaign,
    load_checkpoint_index,
    plan_rutherford_production_campaign,
    select_resume_checkpoint,
    write_checkpoint_metadata,
    write_rutherford_production_execution,
    write_rutherford_production_plan,
    write_rutherford_production_promotion_report,
    write_rutherford_resume_plan,
)
from mhx.cli.main import _exit_if_validation_failed, app


def test_production_campaign_plan_writes_runbook_and_checkpoint_contract(tmp_path) -> None:
    manifest_path, validation = write_rutherford_production_plan(
        tmp_path,
        shape=(96, 96),
        dt=0.5,
        target_saved_frames=120,
        walltime_policy=WalltimePolicy(
            max_walltime_hours=1.0,
            seconds_per_step_estimate=2.0,
            checkpoint_interval_minutes=10.0,
            preemption_margin_minutes=5.0,
        ),
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert validation["diagnostics"]["schema"] == PRODUCTION_RUTHERFORD_PLAN_SCHEMA
    assert validation["diagnostics"]["claim_level"] == "production_template"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["claim_level"] == "production_template"
    assert manifest["outputs"]["campaign_plan"] == "campaign_plan.json"
    assert manifest["outputs"]["checkpoint_index"] == "checkpoints/checkpoint_index.json"

    plan = json.loads((tmp_path / "campaign_plan.json").read_text())
    assert plan["estimated_steps"] > 1000
    assert plan["estimated_walltime_jobs"] >= 1
    assert plan["checkpoint_every_steps"] >= 1
    assert "dissipation_budget_residual" in plan["required_outputs"]["histories"]
    assert "current_sheet_aspect_ratio" in plan["required_outputs"]["histories"]

    checkpoint_index = json.loads(
        (tmp_path / "checkpoints" / "checkpoint_index.json").read_text()
    )
    assert checkpoint_index["schema"] == PRODUCTION_RUTHERFORD_CHECKPOINT_INDEX_SCHEMA
    assert checkpoint_index["checkpoints"] == []
    assert checkpoint_index["target_step"] == plan["estimated_steps"]

    runbook = (tmp_path / "runbook.md").read_text()
    assert "Duration gate" in runbook
    assert "mhx campaign rutherford-resume-plan" in runbook
    assert (tmp_path / "campaign_config.toml").exists()
    assert (tmp_path / "job_array.json").exists()


def test_production_plan_flags_below_floor_resolution_without_running() -> None:
    plan = plan_rutherford_production_campaign(
        shape=(64, 64),
        dt=0.5,
        target_saved_frames=120,
        min_production_resolution=96,
    )

    assert plan.validation["passed"] is False
    assert plan.validation["checks"]["production_resolution_floor"] is False
    assert plan.validation["checks"]["duration_guard_passed"] is True


def test_checkpoint_metadata_and_resume_plan_select_latest_valid_checkpoint(tmp_path) -> None:
    write_rutherford_production_plan(
        tmp_path,
        shape=(96, 96),
        dt=0.5,
        target_saved_frames=120,
        walltime_policy=WalltimePolicy(
            max_walltime_hours=1.0,
            seconds_per_step_estimate=2.0,
            checkpoint_interval_minutes=10.0,
            preemption_margin_minutes=5.0,
        ),
    )
    (tmp_path / "checkpoints" / "state_0000000100.npz").write_bytes(b"state-100")
    (tmp_path / "histories.npz").write_bytes(b"history")
    first_record = write_checkpoint_metadata(
        tmp_path,
        step=100,
        time=50.0,
        state_path="checkpoints/state_0000000100.npz",
        history_path="histories.npz",
        metrics={"total_energy": 1.0, "magnetic_divergence_linf": 0.0},
    )
    (tmp_path / "checkpoints" / "state_0000000200.npz").write_bytes(b"state-200")
    second_record = write_checkpoint_metadata(
        tmp_path,
        step=200,
        time=100.0,
        state_path="checkpoints/state_0000000200.npz",
        metrics={"total_energy": 0.9},
    )

    assert first_record.exists()
    assert second_record.exists()
    index = load_checkpoint_index(tmp_path)
    assert index["latest_checkpoint"] == "step_000000000200"
    assert [checkpoint["step"] for checkpoint in index["checkpoints"]] == [100, 200]

    resume = select_resume_checkpoint(tmp_path, target_step=250)
    assert resume.start_step == 200
    assert resume.target_step == 250
    assert resume.remaining_steps == 50
    assert resume.checkpoint is not None
    assert resume.command_contract["resume_from_checkpoint"] == (
        "checkpoints/step_000000000200.json"
    )

    resume_path, validation = write_rutherford_resume_plan(tmp_path, target_step=250)
    assert validation["passed"] is True
    saved_resume = json.loads(resume_path.read_text())
    assert saved_resume["schema"] == PRODUCTION_RUTHERFORD_RESUME_SCHEMA
    assert saved_resume["start_step"] == 200


def test_resume_plan_falls_back_to_initial_state_when_checkpoint_hash_changes(tmp_path) -> None:
    write_rutherford_production_plan(
        tmp_path,
        shape=(96, 96),
        dt=0.5,
        target_saved_frames=120,
    )
    state_path = tmp_path / "checkpoints" / "state_0000000100.npz"
    state_path.write_bytes(b"original-state")
    write_checkpoint_metadata(
        tmp_path,
        step=100,
        time=50.0,
        state_path="checkpoints/state_0000000100.npz",
    )
    state_path.write_bytes(b"tampered-state")

    resume = select_resume_checkpoint(tmp_path, target_step=200)
    assert resume.start_step == 0
    assert resume.checkpoint is None
    assert resume.validation["invalid_checkpoint_count"] == 1
    assert resume.validation["checks"]["can_resume_from_initial_state"] is True


def test_production_campaign_cli_plan_and_resume(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "campaign",
            "rutherford-plan-production",
            "--outdir",
            str(tmp_path),
            "--nx",
            "96",
            "--ny",
            "96",
            "--dt",
            "0.5",
            "--target-saved-frames",
            "120",
            "--max-walltime-hours",
            "1.0",
            "--seconds-per-step-estimate",
            "2.0",
            "--checkpoint-interval-minutes",
            "10.0",
            "--preemption-margin-minutes",
            "5.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifest.json").exists()

    resume = runner.invoke(
        app,
        ["campaign", "rutherford-resume-plan", str(tmp_path), "--target-step", "10"],
    )
    assert resume.exit_code == 0, resume.output
    assert (tmp_path / "resume_plan.json").exists()


def test_production_execution_runs_real_chunk_and_resumes(tmp_path) -> None:
    write_rutherford_production_plan(
        tmp_path,
        shape=(8, 8),
        dt=1.0e-2,
        target_saved_frames=10,
        min_production_resolution=8,
        walltime_policy=WalltimePolicy(
            max_walltime_hours=1.0,
            seconds_per_step_estimate=0.1,
            checkpoint_interval_minutes=1.0,
            preemption_margin_minutes=1.0,
        ),
    )

    result = execute_rutherford_production_campaign(
        tmp_path,
        max_steps=4,
        seed=0,
        write_movies=False,
    )

    assert result.validation["passed"] is True
    assert result.diagnostics["schema"] == PRODUCTION_RUTHERFORD_EXECUTION_SCHEMA
    assert result.start_step == 0
    assert result.end_step == 4
    assert (tmp_path / "production_history.npz").exists()
    assert (tmp_path / "figures" / "production_histories.png").stat().st_size > 0
    assert (tmp_path / "checkpoints" / "state_step_000000000004.npz").exists()
    assert (tmp_path / "resume_plan.json").exists()
    with np.load(tmp_path / "production_history.npz") as data:
        assert str(data["schema"]) == PRODUCTION_RUTHERFORD_HISTORY_SCHEMA
        assert data["time"].shape[0] >= 2
        assert np.isfinite(data["total_energy"]).all()
        assert np.isfinite(data["current_sheet_length"]).all()
        assert np.isfinite(data["current_sheet_thickness"]).all()
        assert np.isfinite(data["current_sheet_aspect_ratio"]).all()
        assert data["x_point_count"].dtype.kind in {"i", "u"}
        assert data["o_point_count"].dtype.kind in {"i", "u"}
    with np.load(tmp_path / "checkpoints" / "state_step_000000000004.npz") as data:
        assert str(data["schema"]) == PRODUCTION_RUTHERFORD_STATE_SCHEMA
        assert int(data["step"]) == 4
    assert (tmp_path / "figures" / "current_sheet_aspect_ratio.png").stat().st_size > 0

    second = execute_rutherford_production_campaign(
        tmp_path,
        max_steps=2,
        seed=0,
        write_movies=False,
    )
    assert second.start_step == 4
    assert second.end_step == 6
    with np.load(tmp_path / "production_history.npz") as data:
        assert int(data["step"][-1]) == 6


def test_production_execution_cli_writes_movies_for_tiny_chunk(tmp_path) -> None:
    write_rutherford_production_plan(
        tmp_path,
        shape=(8, 8),
        dt=1.0e-2,
        target_saved_frames=10,
        min_production_resolution=8,
    )
    result = CliRunner().invoke(
        app,
        [
            "campaign",
            "rutherford-execute",
            str(tmp_path),
            "--max-steps",
            "2",
            "--movies",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "figures" / "fixed_scale_flux_movie.gif").stat().st_size > 0
    assert (tmp_path / "figures" / "fixed_scale_current_density_movie.gif").stat().st_size > 0


def test_production_promotion_report_blocks_incomplete_bundle(tmp_path) -> None:
    write_rutherford_production_plan(
        tmp_path,
        shape=(8, 8),
        dt=1.0e-2,
        target_saved_frames=10,
        min_production_resolution=8,
    )
    execute_rutherford_production_campaign(
        tmp_path,
        max_steps=4,
        seed=0,
        write_movies=False,
    )

    assessment = assess_rutherford_production_promotion(
        tmp_path,
        require_movies=False,
        min_history_samples=2,
    )

    assert assessment.promotion_ready is False
    assert assessment.diagnostics["schema"] == PRODUCTION_RUTHERFORD_PROMOTION_SCHEMA
    assert assessment.validation["checks"]["completed_target"] is False
    assert assessment.validation["checks"]["convergence_bundles_passed"] is False
    assert assessment.validation["checks"]["seed_qi_bundle_present"] is False

    manifest_path, validation = write_rutherford_production_promotion_report(
        tmp_path,
        require_movies=False,
        min_history_samples=2,
    )
    assert validation["passed"] is False
    assert manifest_path == tmp_path / "promotion" / "manifest.json"
    assert (tmp_path / "promotion" / "promotion_readiness.json").exists()
    assert (tmp_path / "promotion" / "figures" / "promotion_matrix.png").stat().st_size > 0

    cli_result = CliRunner().invoke(
        app,
        [
            "campaign",
            "rutherford-promotion-check",
            str(tmp_path),
            "--no-require-movies",
            "--min-history-samples",
            "2",
        ],
    )
    assert cli_result.exit_code == 1
    assert "rutherford-promotion-check failed validation checks" in cli_result.output


def test_production_promotion_report_enables_explicit_production_claim(tmp_path) -> None:
    write_rutherford_production_plan(
        tmp_path,
        shape=(8, 8),
        dt=0.1,
        target_saved_frames=10,
        harris_growth_rate=10.0,
        production_efolds=0.1,
        safety_factor=1.0,
        min_production_resolution=8,
    )
    execution = execute_rutherford_production_campaign(
        tmp_path,
        seed=0,
        write_movies=False,
        max_relative_energy_growth=1.0,
    )
    assert execution.completed_target is True
    assert execution.diagnostics["claim_level"] == "validation"

    convergence_dirs = []
    for name in ("resolution_sweep", "time_step_sweep"):
        evidence_dir = tmp_path / "evidence" / name
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "validation.json").write_text(
            json.dumps({"schema": f"mhx.test.{name}.gates.v1", "passed": True}),
            encoding="utf-8",
        )
        (evidence_dir / "manifest.json").write_text(
            json.dumps({"claim_level": "validation"}),
            encoding="utf-8",
        )
        (evidence_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
        convergence_dirs.append(evidence_dir)
    seed_qi_dir = tmp_path / "evidence" / "seed_qi"
    seed_qi_dir.mkdir(parents=True)
    (seed_qi_dir / "validation.json").write_text(
        json.dumps({"schema": "mhx.test.seed_qi.gates.v1", "passed": True}),
        encoding="utf-8",
    )
    (seed_qi_dir / "manifest.json").write_text(
        json.dumps({"claim_level": "validation"}),
        encoding="utf-8",
    )

    manifest_path, validation = write_rutherford_production_promotion_report(
        tmp_path,
        convergence_dirs=tuple(convergence_dirs),
        seed_qi_dir=seed_qi_dir,
        require_movies=False,
        min_history_samples=2,
        max_energy_budget_residual=1.0,
    )

    assert validation["passed"] is True
    assert json.loads(manifest_path.read_text())["claim_level"] == "production"

    promoted = execute_rutherford_production_campaign(
        tmp_path,
        max_steps=0,
        seed=0,
        write_movies=False,
        allow_production_claim=True,
        max_relative_energy_growth=1.0,
    )
    assert promoted.validation["passed"] is True
    assert promoted.diagnostics["claim_level"] == "production"


def test_cli_validation_failure_reports_failed_checks(capsys) -> None:
    _exit_if_validation_failed({"passed": True}, context="unit")
    validation = {
        "passed": False,
        "checks": {"stable": False, "finite": True},
        "diagnostics": {
            "max_relative_energy_growth": 1.5,
            "final_magnetic_divergence_linf": 0.0,
            "steps_run": 4,
            "dt": 0.5,
            "shape": [96, 96],
        },
    }

    with pytest.raises(click.exceptions.Exit):
        _exit_if_validation_failed(validation, context="unit")

    captured = capsys.readouterr()
    assert "unit failed validation checks: stable" in captured.err
    assert "max_relative_energy_growth: 1.5" in captured.err
    assert "shape: [96, 96]" in captured.err


def test_production_campaign_rejects_invalid_controls(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_walltime_hours"):
        WalltimePolicy(max_walltime_hours=0.0).validated()
    with pytest.raises(ValueError, match="seconds_per_step"):
        WalltimePolicy(seconds_per_step_estimate=0.0).validated()
    with pytest.raises(ValueError, match="checkpoint_interval"):
        WalltimePolicy(checkpoint_interval_minutes=0.0).validated()
    with pytest.raises(ValueError, match="preemption_margin"):
        WalltimePolicy(preemption_margin_minutes=-1.0).validated()
    with pytest.raises(ValueError, match="usable walltime"):
        WalltimePolicy(max_walltime_hours=0.1, preemption_margin_minutes=6.0).validated()
    with pytest.raises(ValueError, match="min_steps_per_job"):
        WalltimePolicy(min_steps_per_job=0).validated()
    with pytest.raises(ValueError, match="step"):
        write_checkpoint_metadata(
            tmp_path,
            step=-1,
            time=0.0,
            state_path="missing.npz",
        )
    state_path = tmp_path / "state.npz"
    state_path.write_bytes(b"state")
    with pytest.raises(ValueError, match="time"):
        write_checkpoint_metadata(
            tmp_path,
            step=0,
            time=-1.0,
            state_path="state.npz",
        )
    with pytest.raises(ValueError, match="walltime_elapsed_seconds"):
        write_checkpoint_metadata(
            tmp_path,
            step=0,
            time=0.0,
            state_path="state.npz",
            walltime_elapsed_seconds=-1.0,
        )
    with pytest.raises(FileNotFoundError, match="checkpoint state"):
        write_checkpoint_metadata(
            tmp_path,
            step=0,
            time=0.0,
            state_path="missing.npz",
        )
    with pytest.raises(FileNotFoundError, match="campaign plan"):
        write_rutherford_production_execution(tmp_path / "missing", max_steps=1)
    write_rutherford_production_plan(
        tmp_path / "invalid-execution",
        shape=(8, 8),
        dt=1.0e-2,
        target_saved_frames=10,
        min_production_resolution=8,
    )
    with pytest.raises(ValueError, match="max_steps"):
        write_rutherford_production_execution(tmp_path / "invalid-execution", max_steps=-1)


def test_checkpoint_index_error_paths_and_absolute_artifacts(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint index"):
        load_checkpoint_index(tmp_path / "missing")

    created = load_checkpoint_index(tmp_path / "created", create_if_missing=True)
    assert created["target_step"] == 0
    index_path = tmp_path / "bad" / "checkpoints"
    index_path.mkdir(parents=True)
    (index_path / "checkpoint_index.json").write_text(
        json.dumps({"schema": "bad.schema", "checkpoints": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported checkpoint-index"):
        load_checkpoint_index(tmp_path / "bad")

    write_rutherford_production_plan(
        tmp_path / "absolute",
        shape=(96, 96),
        dt=0.5,
        target_saved_frames=120,
    )
    absolute_state = tmp_path / "external_state.npz"
    absolute_state.write_bytes(b"state")
    diagnostics_path = tmp_path / "absolute" / "diagnostics.json"
    diagnostics_path.write_text("{}", encoding="utf-8")
    record_path = write_checkpoint_metadata(
        tmp_path / "absolute",
        step=1,
        time=0.5,
        state_path=absolute_state,
        diagnostics_path=diagnostics_path,
        metrics={"bad": float("nan")},
    )
    record = json.loads(record_path.read_text())
    assert record["validation"]["passed"] is False
    assert record["artifacts"]["state"]["path"] == str(absolute_state)

    with pytest.raises(ValueError, match="target_step"):
        select_resume_checkpoint(tmp_path / "absolute", target_step=-1)
