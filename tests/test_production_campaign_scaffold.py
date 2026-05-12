from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mhx.campaigns import (
    PRODUCTION_RUTHERFORD_CHECKPOINT_INDEX_SCHEMA,
    PRODUCTION_RUTHERFORD_PLAN_SCHEMA,
    PRODUCTION_RUTHERFORD_RESUME_SCHEMA,
    WalltimePolicy,
    load_checkpoint_index,
    plan_rutherford_production_campaign,
    select_resume_checkpoint,
    write_checkpoint_metadata,
    write_rutherford_production_plan,
    write_rutherford_resume_plan,
)
from mhx.cli.main import app


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


def test_production_campaign_scaffold_rejects_invalid_controls(tmp_path) -> None:
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
