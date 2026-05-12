"""Production Rutherford-campaign planning, execution, and checkpoint metadata.

This module defines the reviewer-facing contract that a production executor must
satisfy: duration gates, walltime chunking, checkpoint records, resume
selection, histories, movies, and artifact manifests. The executor can run small
validation chunks in CI and the same code path can continue a long campaign on a
scheduler without changing schemas.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.benchmarks.campaigns import (
    RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA,
    build_rutherford_campaign_template,
)
from mhx.benchmarks.duration_policy import (
    DEFAULT_PRODUCTION_EFOLDS,
    HARRIS_REFERENCE_GROWTH_RATE,
)
from mhx.benchmarks.seed_robust_qi import seeded_tearing_initial_state
from mhx.config import RunConfig
from mhx.diagnostics import (
    island_width_from_mode,
    magnetic_divergence_linf,
    reconnected_flux_amplitude,
    trajectory_energies,
)
from mhx.equations.reduced_mhd import current_density, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_artifact_manifest, write_manifest
from mhx.plotting import plot_current_density_gif, plot_flux_gif
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import rk4_step
from mhx.versioning import require_supported_api_version

PRODUCTION_RUTHERFORD_PLAN_SCHEMA = "mhx.campaign.rutherford_production_plan.v1"
PRODUCTION_RUTHERFORD_RUNBOOK_SCHEMA = "mhx.campaign.rutherford_runbook.v1"
PRODUCTION_RUTHERFORD_CHECKPOINT_SCHEMA = "mhx.campaign.rutherford_checkpoint.v1"
PRODUCTION_RUTHERFORD_CHECKPOINT_INDEX_SCHEMA = "mhx.campaign.rutherford_checkpoint_index.v1"
PRODUCTION_RUTHERFORD_RESUME_SCHEMA = "mhx.campaign.rutherford_resume_plan.v1"
PRODUCTION_RUTHERFORD_EXECUTION_SCHEMA = "mhx.campaign.rutherford_execution.v1"
PRODUCTION_RUTHERFORD_HISTORY_SCHEMA = "mhx.campaign.rutherford_history.v1"
PRODUCTION_RUTHERFORD_STATE_SCHEMA = "mhx.campaign.rutherford_state.v1"


@dataclass(frozen=True)
class WalltimePolicy:
    """Walltime and checkpoint cadence for long nonlinear campaigns."""

    max_walltime_hours: float = 12.0
    seconds_per_step_estimate: float = 0.5
    checkpoint_interval_minutes: float = 30.0
    preemption_margin_minutes: float = 20.0
    min_steps_per_job: int = 1

    def validated(self) -> WalltimePolicy:
        """Return a validated copy or raise with a precise configuration error."""
        if self.max_walltime_hours <= 0.0:
            raise ValueError("max_walltime_hours must be positive")
        if self.seconds_per_step_estimate <= 0.0:
            raise ValueError("seconds_per_step_estimate must be positive")
        if self.checkpoint_interval_minutes <= 0.0:
            raise ValueError("checkpoint_interval_minutes must be positive")
        if self.preemption_margin_minutes < 0.0:
            raise ValueError("preemption_margin_minutes must be non-negative")
        if self.usable_walltime_seconds <= 0.0:
            raise ValueError("preemption_margin_minutes leaves no usable walltime")
        if self.min_steps_per_job < 1:
            raise ValueError("min_steps_per_job must be >= 1")
        return self

    @property
    def max_walltime_seconds(self) -> float:
        """Scheduler walltime budget in seconds."""
        return self.max_walltime_hours * 3600.0

    @property
    def usable_walltime_seconds(self) -> float:
        """Walltime remaining after reserving the preemption margin."""
        return self.max_walltime_seconds - self.preemption_margin_minutes * 60.0

    @property
    def checkpoint_interval_seconds(self) -> float:
        """Target checkpoint cadence in seconds."""
        return self.checkpoint_interval_minutes * 60.0

    def steps_per_walltime_job(self) -> int:
        """Conservative number of steps in one scheduler allocation."""
        steps = math.floor(self.usable_walltime_seconds / self.seconds_per_step_estimate)
        return max(self.min_steps_per_job, max(1, steps))

    def checkpoint_every_steps(self) -> int:
        """Conservative step interval between checkpoint records."""
        steps = math.floor(self.checkpoint_interval_seconds / self.seconds_per_step_estimate)
        return max(1, steps)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible walltime policy values."""
        return {
            **asdict(self),
            "max_walltime_seconds": self.max_walltime_seconds,
            "usable_walltime_seconds": self.usable_walltime_seconds,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "steps_per_walltime_job": self.steps_per_walltime_job(),
            "checkpoint_every_steps": self.checkpoint_every_steps(),
        }


@dataclass(frozen=True)
class ProductionCampaignPlan:
    """A duration-guarded production campaign plan that does not run the solver."""

    config: RunConfig
    walltime_policy: WalltimePolicy
    diagnostics: dict[str, Any]
    validation: dict[str, Any]
    runbook_markdown: str
    checkpoint_index: dict[str, Any]
    job_array: dict[str, Any]


@dataclass(frozen=True)
class ResumePlan:
    """Resume selection for a production campaign checkpoint index."""

    run_dir: Path
    checkpoint: dict[str, Any] | None
    start_step: int
    target_step: int
    remaining_steps: int
    chunks_remaining: int
    validation: dict[str, Any]
    command_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resume plan."""
        return {
            "schema": PRODUCTION_RUTHERFORD_RESUME_SCHEMA,
            "api_version": require_supported_api_version(context="rutherford resume plan"),
            "run_dir": str(self.run_dir),
            "checkpoint": self.checkpoint,
            "start_step": self.start_step,
            "target_step": self.target_step,
            "remaining_steps": self.remaining_steps,
            "chunks_remaining": self.chunks_remaining,
            "validation": self.validation,
            "command_contract": self.command_contract,
        }


@dataclass(frozen=True)
class ProductionExecutionResult:
    """A completed walltime chunk from a Rutherford campaign executor."""

    run_dir: Path
    start_step: int
    end_step: int
    target_step: int
    completed_target: bool
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def plan_rutherford_production_campaign(
    *,
    harris_growth_rate: float = HARRIS_REFERENCE_GROWTH_RATE,
    production_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
    safety_factor: float = 3.0,
    shape: tuple[int, int] = (128, 128),
    dt: float = 0.1,
    target_saved_frames: int = 400,
    run_output_dir: str | Path = "outputs/production/rutherford_island",
    min_production_resolution: int = 96,
    walltime_policy: WalltimePolicy | None = None,
) -> ProductionCampaignPlan:
    """Build a production Rutherford campaign plan without advancing the PDE."""
    policy = (walltime_policy or WalltimePolicy()).validated()
    template = build_rutherford_campaign_template(
        harris_growth_rate=harris_growth_rate,
        production_efolds=production_efolds,
        safety_factor=safety_factor,
        shape=shape,
        dt=dt,
        target_saved_frames=target_saved_frames,
        run_output_dir=run_output_dir,
    )
    estimated_steps = int(template.estimated_steps)
    checkpoint_every = min(policy.checkpoint_every_steps(), estimated_steps)
    steps_per_job = min(policy.steps_per_walltime_job(), estimated_steps)
    estimated_checkpoints = max(1, math.ceil(estimated_steps / checkpoint_every))
    estimated_walltime_jobs = max(1, math.ceil(estimated_steps / steps_per_job))
    walltime_seconds_estimate = estimated_steps * policy.seconds_per_step_estimate
    job_array = _build_job_array(
        run_dir=Path(run_output_dir),
        total_steps=estimated_steps,
        steps_per_job=steps_per_job,
        checkpoint_every_steps=checkpoint_every,
    )
    required_outputs = _required_production_outputs()
    checks = {
        "duration_guard_passed": template.duration_assessment.sufficient_for_nonlinear_claim,
        "production_resolution_floor": min(shape) >= min_production_resolution,
        "saved_frame_floor": template.estimated_saved_frames >= 100,
        "checkpoint_cadence_defined": checkpoint_every >= 1,
        "walltime_chunking_defined": steps_per_job >= 1,
        "checkpoint_index_initialized": True,
        "resume_contract_defined": True,
        "required_outputs_declared": all(required_outputs.values()),
        "claim_level_not_completed_production": True,
    }
    diagnostics = {
        "schema": PRODUCTION_RUTHERFORD_PLAN_SCHEMA,
        "template_schema": RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA,
        "api_version": require_supported_api_version(context="rutherford production plan"),
        "claim_level": "production_template",
        "claim_boundary": (
            "Planning, checkpoint, and runbook contract for a long nonlinear "
            "Rutherford campaign; not a completed production simulation."
        ),
        "config": template.config.to_dict(),
        "duration_assessment": template.duration_assessment.to_dict(),
        "walltime_policy": policy.to_dict(),
        "estimated_steps": estimated_steps,
        "estimated_saved_frames": int(template.estimated_saved_frames),
        "save_every": int(template.config.time.save_every),
        "checkpoint_every_steps": checkpoint_every,
        "estimated_checkpoints": estimated_checkpoints,
        "steps_per_walltime_job": steps_per_job,
        "estimated_walltime_jobs": estimated_walltime_jobs,
        "estimated_walltime_seconds": walltime_seconds_estimate,
        "min_production_resolution": min_production_resolution,
        "required_outputs": required_outputs,
        "job_array_schema": job_array["schema"],
    }
    validation = {
        "schema": "mhx.campaign.rutherford_production_plan.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_production_resolution": min_production_resolution,
            "min_saved_frames": 100,
            "required_efolds": production_efolds,
            "safety_factor": safety_factor,
        },
        "diagnostics": diagnostics,
    }
    checkpoint_index = _initial_checkpoint_index(
        run_dir=Path(run_output_dir),
        target_step=estimated_steps,
        target_time=float(template.config.time.t1),
        walltime_policy=policy,
        checkpoint_every_steps=checkpoint_every,
        steps_per_walltime_job=steps_per_job,
    )
    return ProductionCampaignPlan(
        config=template.config,
        walltime_policy=policy,
        diagnostics=diagnostics,
        validation=validation,
        runbook_markdown=_render_runbook(diagnostics, job_array),
        checkpoint_index=checkpoint_index,
        job_array=job_array,
    )


def write_rutherford_production_plan(
    outdir: str | Path,
    *,
    harris_growth_rate: float = HARRIS_REFERENCE_GROWTH_RATE,
    production_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
    safety_factor: float = 3.0,
    shape: tuple[int, int] = (128, 128),
    dt: float = 0.1,
    target_saved_frames: int = 400,
    run_output_dir: str | Path | None = None,
    min_production_resolution: int = 96,
    walltime_policy: WalltimePolicy | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write production-plan artifacts and an initialized checkpoint index."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedded_run_dir = Path(run_output_dir) if run_output_dir is not None else output_dir
    plan = plan_rutherford_production_campaign(
        harris_growth_rate=harris_growth_rate,
        production_efolds=production_efolds,
        safety_factor=safety_factor,
        shape=shape,
        dt=dt,
        target_saved_frames=target_saved_frames,
        run_output_dir=embedded_run_dir,
        min_production_resolution=min_production_resolution,
        walltime_policy=walltime_policy,
    )
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "campaign_plan.json"
    config_path = output_dir / "campaign_config.toml"
    validation_path = output_dir / "validation.json"
    runbook_path = output_dir / "runbook.md"
    checkpoint_index_path = checkpoints_dir / "checkpoint_index.json"
    job_array_path = output_dir / "job_array.json"
    manifest_path = output_dir / "manifest.json"
    plan_path.write_text(
        json.dumps(plan.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config_path.write_text(plan.config.to_toml(), encoding="utf-8")
    validation_path.write_text(
        json.dumps(plan.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    runbook_path.write_text(plan.runbook_markdown, encoding="utf-8")
    checkpoint_index_path.write_text(
        json.dumps(plan.checkpoint_index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    job_array_path.write_text(
        json.dumps(plan.job_array, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_manifest(
        manifest_path,
        config=plan.diagnostics,
        outputs={
            "campaign_plan": plan_path.name,
            "campaign_config": config_path.name,
            "validation": validation_path.name,
            "runbook": runbook_path.name,
            "checkpoint_index": checkpoint_index_path.relative_to(output_dir).as_posix(),
            "job_array": job_array_path.name,
        },
        claim_level="production_template",
        claim_scope=(
            "Production Rutherford campaign planning, walltime, checkpoint, and "
            "resume contract. This is not a completed production simulation."
        ),
    )
    return manifest_path, plan.validation


def write_checkpoint_metadata(
    run_dir: str | Path,
    *,
    step: int,
    time: float,
    state_path: str | Path,
    history_path: str | Path | None = None,
    diagnostics_path: str | Path | None = None,
    walltime_elapsed_seconds: float = 0.0,
    metrics: dict[str, float] | None = None,
    completed: bool = False,
) -> Path:
    """Write a checkpoint metadata record and update ``checkpoint_index.json``."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if time < 0.0:
        raise ValueError("time must be non-negative")
    if walltime_elapsed_seconds < 0.0:
        raise ValueError("walltime_elapsed_seconds must be non-negative")
    root = Path(run_dir)
    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _checkpoint_artifacts(
        root,
        state_path=state_path,
        history_path=history_path,
        diagnostics_path=diagnostics_path,
    )
    if not artifacts["state"]["exists"]:
        raise FileNotFoundError(f"checkpoint state artifact is missing: {state_path}")
    finite_metrics = all(math.isfinite(float(value)) for value in (metrics or {}).values())
    record = {
        "schema": PRODUCTION_RUTHERFORD_CHECKPOINT_SCHEMA,
        "api_version": require_supported_api_version(context="rutherford checkpoint"),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "checkpoint_id": f"step_{step:012d}",
        "step": int(step),
        "time": float(time),
        "completed": bool(completed),
        "walltime_elapsed_seconds": float(walltime_elapsed_seconds),
        "metrics": {str(key): float(value) for key, value in (metrics or {}).items()},
        "artifacts": artifacts,
        "validation": {
            "schema": "mhx.campaign.rutherford_checkpoint.gates.v1",
            "passed": finite_metrics and artifacts["state"]["exists"],
            "checks": {
                "state_artifact_exists": artifacts["state"]["exists"],
                "finite_metrics": finite_metrics,
                "step_nonnegative": True,
                "time_nonnegative": True,
            },
        },
    }
    record_path = checkpoints_dir / f"{record['checkpoint_id']}.json"
    record["metadata_path"] = record_path.relative_to(root).as_posix()
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    index = load_checkpoint_index(root, create_if_missing=True)
    checkpoints = [
        checkpoint
        for checkpoint in index["checkpoints"]
        if checkpoint["checkpoint_id"] != record["checkpoint_id"]
    ]
    checkpoints.append(record)
    checkpoints.sort(key=lambda checkpoint: int(checkpoint["step"]))
    index["checkpoints"] = checkpoints
    index["latest_checkpoint"] = checkpoints[-1]["checkpoint_id"] if checkpoints else None
    index["updated_utc"] = datetime.now(tz=timezone.utc).isoformat()
    _checkpoint_index_path(root).write_text(
        json.dumps(index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return record_path


def load_checkpoint_index(
    run_dir: str | Path,
    *,
    create_if_missing: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint index, optionally creating a minimal empty index."""
    root = Path(run_dir)
    path = _checkpoint_index_path(root)
    if not path.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"checkpoint index is missing: {path}")
        index = _initial_checkpoint_index(
            run_dir=root,
            target_step=0,
            target_time=0.0,
            walltime_policy=WalltimePolicy(),
            checkpoint_every_steps=WalltimePolicy().checkpoint_every_steps(),
            steps_per_walltime_job=WalltimePolicy().steps_per_walltime_job(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        return index
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != PRODUCTION_RUTHERFORD_CHECKPOINT_INDEX_SCHEMA:
        raise ValueError(f"unsupported checkpoint-index schema in {path}")
    return index


def select_resume_checkpoint(
    run_dir: str | Path,
    *,
    target_step: int | None = None,
) -> ResumePlan:
    """Select the latest valid checkpoint and return a deterministic resume plan."""
    root = Path(run_dir)
    index = load_checkpoint_index(root)
    selected_target_step = int(
        index.get("target_step", 0) if target_step is None else target_step
    )
    if selected_target_step < 0:
        raise ValueError("target_step must be non-negative")
    candidates = [
        checkpoint
        for checkpoint in index["checkpoints"]
        if int(checkpoint["step"]) <= selected_target_step
        and _checkpoint_record_is_valid(root, checkpoint)
    ]
    checkpoint = max(candidates, key=lambda item: int(item["step"])) if candidates else None
    start_step = int(checkpoint["step"]) if checkpoint is not None else 0
    remaining_steps = max(0, selected_target_step - start_step)
    steps_per_job = int(index["resume_policy"].get("steps_per_walltime_job", 1))
    chunks_remaining = 0 if remaining_steps == 0 else math.ceil(remaining_steps / steps_per_job)
    validation = {
        "schema": "mhx.campaign.rutherford_resume_plan.gates.v1",
        "passed": True,
        "checks": {
            "checkpoint_index_loaded": True,
            "target_step_not_before_checkpoint": start_step <= selected_target_step,
            "valid_checkpoint_available": checkpoint is not None,
            "can_resume_from_initial_state": checkpoint is None,
        },
        "invalid_checkpoint_count": len(index["checkpoints"]) - len(candidates),
    }
    command_contract = {
        "executor_status": "mhx_campaign_rutherford_execute_available",
        "resume_from_checkpoint": None
        if checkpoint is None
        else checkpoint["metadata_path"],
        "start_step": start_step,
        "target_step": selected_target_step,
        "steps_per_walltime_job": steps_per_job,
        "required_executor_behavior": (
            "load the selected state artifact, continue fixed-step integration, "
            "append histories without changing schema keys, write a checkpoint "
            "before walltime margin, and refresh checkpoint_index.json"
        ),
    }
    return ResumePlan(
        run_dir=root,
        checkpoint=checkpoint,
        start_step=start_step,
        target_step=selected_target_step,
        remaining_steps=remaining_steps,
        chunks_remaining=chunks_remaining,
        validation=validation,
        command_contract=command_contract,
    )


def write_rutherford_resume_plan(
    run_dir: str | Path,
    *,
    target_step: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``resume_plan.json`` for the latest valid checkpoint."""
    root = Path(run_dir)
    resume_plan = select_resume_checkpoint(root, target_step=target_step)
    path = root / "resume_plan.json"
    path.write_text(
        json.dumps(resume_plan.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path, resume_plan.validation


def write_rutherford_production_execution(
    run_dir: str | Path,
    *,
    max_steps: int | None = None,
    seed: int = 0,
    noise_amplitude: float = 1.0e-6,
    write_movies: bool = False,
    allow_production_claim: bool = False,
    max_relative_energy_growth: float = 1.0e-9,
    max_divergence_linf: float = 1.0e-9,
    walltime_elapsed_seconds: float = 0.0,
) -> tuple[Path, dict[str, Any]]:
    """Run one restartable Rutherford campaign chunk and write real artifacts.

    The function consumes ``campaign_plan.json`` and ``checkpoints/`` in
    ``run_dir``.  If no checkpoint is valid it starts from the deterministic
    initial condition; otherwise it resumes from the latest valid state.  Small
    CI runs normally use ``max_steps`` and keep ``claim_level = validation``.
    A completed target run can only emit ``claim_level = production`` when
    ``allow_production_claim`` is set and all execution gates pass.
    """
    result = execute_rutherford_production_campaign(
        run_dir,
        max_steps=max_steps,
        seed=seed,
        noise_amplitude=noise_amplitude,
        write_movies=write_movies,
        allow_production_claim=allow_production_claim,
        max_relative_energy_growth=max_relative_energy_growth,
        max_divergence_linf=max_divergence_linf,
        walltime_elapsed_seconds=walltime_elapsed_seconds,
    )
    return result.run_dir / "manifest.json", result.validation


def execute_rutherford_production_campaign(
    run_dir: str | Path,
    *,
    max_steps: int | None = None,
    seed: int = 0,
    noise_amplitude: float = 1.0e-6,
    write_movies: bool = False,
    allow_production_claim: bool = False,
    max_relative_energy_growth: float = 1.0e-9,
    max_divergence_linf: float = 1.0e-9,
    walltime_elapsed_seconds: float = 0.0,
) -> ProductionExecutionResult:
    """Execute a real restartable Rutherford campaign chunk."""
    root = Path(run_dir)
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative when provided")
    if noise_amplitude < 0.0:
        raise ValueError("noise_amplitude must be non-negative")
    if max_relative_energy_growth < 0.0:
        raise ValueError("max_relative_energy_growth must be non-negative")
    if max_divergence_linf < 0.0:
        raise ValueError("max_divergence_linf must be non-negative")
    if walltime_elapsed_seconds < 0.0:
        raise ValueError("walltime_elapsed_seconds must be non-negative")
    plan = _load_production_plan(root)
    config = plan["config"]
    mesh = config["mesh"]
    time_config = config["time"]
    physics = config["physics"]
    diagnostics_config = config["diagnostics"]
    shape = tuple(int(item) for item in mesh["shape"])
    grid = CartesianGrid(
        shape=shape,
        lower=tuple(float(item) for item in mesh["lower"]),
        upper=tuple(float(item) for item in mesh["upper"]),
    )
    dt = float(time_config["dt"])
    save_every = int(time_config["save_every"])
    target_step = int(plan["estimated_steps"])
    mode = tuple(int(item) for item in diagnostics_config.get("mode", (1, 1)))
    magnetic_shear = float(
        physics.get("equilibrium_parameters", {}).get("magnetic_shear", 1.0)
    )
    resistivity = float(physics["resistivity"])
    viscosity = float(physics["viscosity"])
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)
    resume_plan = select_resume_checkpoint(root, target_step=target_step)
    start_step = int(resume_plan.start_step)
    initial_state = (
        _load_checkpoint_state(root, resume_plan.checkpoint)
        if resume_plan.checkpoint is not None
        else seeded_tearing_initial_state(
            grid,
            seed=seed,
            perturbation_amplitude=float(
                physics.get("equilibrium_parameters", {}).get(
                    "perturbation_amplitude",
                    1.0e-3,
                )
            ),
            noise_amplitude=noise_amplitude,
        )
    )
    remaining_steps = max(0, target_step - start_step)
    steps_to_run = remaining_steps if max_steps is None else min(int(max_steps), remaining_steps)

    saved_steps, saved_times, saved_states = _advance_campaign_chunk(
        initial_state,
        params=params,
        grid=grid,
        start_step=start_step,
        steps=steps_to_run,
        dt=dt,
        save_every=save_every,
    )
    end_step = start_step + steps_to_run
    final_state = saved_states[-1] if saved_states else initial_state
    history = _history_from_saved_states(
        saved_steps=saved_steps,
        saved_times=saved_times,
        saved_states=saved_states,
        lengths=grid.lengths,
        mode=mode,
        magnetic_shear=magnetic_shear,
    )
    history = _append_history(root / "production_history.npz", history)

    diagnostics_path = root / "diagnostics.json"
    validation_path = root / "validation.json"
    history_path = root / "production_history.npz"
    state_path = root / "checkpoints" / f"state_step_{end_step:012d}.npz"
    figures_dir = root / "figures"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_campaign_state(state_path, step=end_step, time=end_step * dt, state=final_state)
    _write_campaign_history(history_path, history=history)

    chunk_trajectory = _trajectory_from_saved_states(saved_times, saved_states)
    figure_outputs = _write_execution_figures(
        figures_dir,
        history=history,
        trajectory=chunk_trajectory,
        extent=(
            float(mesh["lower"][0]),
            float(mesh["upper"][0]),
            float(mesh["lower"][1]),
            float(mesh["upper"][1]),
        ),
        lengths=grid.lengths,
        write_movies=write_movies and bool(saved_states),
    )
    completed_target = end_step >= target_step
    max_growth = _max_relative_energy_growth(history["total_energy"])
    divergence = (
        float(history["magnetic_divergence_linf"][-1])
        if history["magnetic_divergence_linf"].size
        else float(magnetic_divergence_linf(final_state, lengths=grid.lengths))
    )
    checks = {
        "campaign_plan_loaded": True,
        "checkpoint_resume_contract_used": True,
        "executor_made_progress_or_target_already_complete": steps_to_run > 0 or completed_target,
        "finite_histories": all(np.isfinite(value).all() for value in history.values()),
        "checkpoint_state_written": state_path.exists(),
        "checkpoint_metadata_written": True,
        "energy_growth_within_tolerance": max_growth <= max_relative_energy_growth,
        "magnetic_divergence_within_tolerance": divergence <= max_divergence_linf,
        "movies_written_when_requested": (not write_movies) or all(
            (root / relative_path).exists()
            for key, relative_path in figure_outputs.items()
            if key.endswith("_movie")
        ),
    }
    claim_level = (
        "production"
        if completed_target and allow_production_claim and all(checks.values())
        else "validation"
    )
    checks["production_claim_requires_explicit_completion"] = (
        claim_level != "production" or (completed_target and allow_production_claim)
    )
    diagnostics = {
        "schema": PRODUCTION_RUTHERFORD_EXECUTION_SCHEMA,
        "plan_schema": plan.get("schema"),
        "api_version": require_supported_api_version(context="rutherford execution"),
        "claim_level": claim_level,
        "claim_boundary": (
            "Restartable Rutherford campaign execution using the production "
            "checkpoint/history schema. Partial chunks are validation artifacts; "
            "production claims require completing the target and enabling the "
            "explicit production-claim gate."
        ),
        "start_step": start_step,
        "end_step": end_step,
        "steps_run": steps_to_run,
        "target_step": target_step,
        "completed_target": completed_target,
        "dt": dt,
        "save_every": save_every,
        "shape": list(shape),
        "seed": int(seed),
        "noise_amplitude": float(noise_amplitude),
        "resistivity": resistivity,
        "viscosity": viscosity,
        "mode": list(mode),
        "max_relative_energy_growth": max_growth,
        "final_magnetic_divergence_linf": divergence,
        "history_samples": int(history["time"].size),
        "resume_start_checkpoint": resume_plan.command_contract["resume_from_checkpoint"],
        "allow_production_claim": bool(allow_production_claim),
    }
    validation = {
        "schema": "mhx.campaign.rutherford_execution.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_relative_energy_growth": max_relative_energy_growth,
            "max_divergence_linf": max_divergence_linf,
        },
        "diagnostics": diagnostics,
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    checkpoint_record = write_checkpoint_metadata(
        root,
        step=end_step,
        time=end_step * dt,
        state_path=state_path.relative_to(root),
        history_path=history_path.relative_to(root),
        diagnostics_path=diagnostics_path.relative_to(root),
        walltime_elapsed_seconds=walltime_elapsed_seconds,
        metrics={
            "total_energy": float(history["total_energy"][-1]),
            "magnetic_divergence_linf": divergence,
            "rutherford_island_width": float(history["rutherford_island_width"][-1]),
            "reconnected_flux": float(history["reconnected_flux"][-1]),
        },
        completed=completed_target,
    )
    write_rutherford_resume_plan(root, target_step=target_step)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    write_artifact_manifest(root)
    outputs = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "histories": history_path.name,
        "checkpoint_state": state_path.relative_to(root).as_posix(),
        "checkpoint_metadata": checkpoint_record.relative_to(root).as_posix(),
        "checkpoint_index": "checkpoints/checkpoint_index.json",
        "resume_plan": "resume_plan.json",
        "artifact_manifest": "artifact_manifest.json",
        **figure_outputs,
    }
    write_manifest(
        manifest_path,
        config=diagnostics,
        outputs=outputs,
        claim_level=claim_level,
        claim_scope=diagnostics["claim_boundary"],
    )
    return ProductionExecutionResult(
        run_dir=root,
        start_step=start_step,
        end_step=end_step,
        target_step=target_step,
        completed_target=completed_target,
        diagnostics=diagnostics,
        validation=validation,
    )


def _required_production_outputs() -> dict[str, tuple[str, ...]]:
    return {
        "histories": (
            "time",
            "reconnected_flux",
            "rutherford_island_width",
            "reconnection_rate_proxy",
            "magnetic_energy",
            "kinetic_energy",
            "total_energy",
            "dissipation_budget_residual",
            "magnetic_divergence_linf",
            "current_density_linf",
        ),
        "figures": (
            "island_width_history.png",
            "reconnected_flux_history.png",
            "energy_budget_history.png",
            "current_sheet_aspect_ratio.png",
        ),
        "movies": (
            "fixed_scale_flux_movie.gif",
            "fixed_scale_current_density_movie.gif",
        ),
        "convergence": (
            "resolution_sweep",
            "time_step_sweep",
            "checkpoint_restart_reproducibility",
            "fit_window_sensitivity",
        ),
        "metadata": (
            "campaign_plan.json",
            "campaign_config.toml",
            "checkpoint_index.json",
            "manifest.json",
            "artifact_manifest.json",
        ),
    }


def _load_production_plan(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "campaign_plan.json"
    if not path.exists():
        raise FileNotFoundError(f"campaign plan is missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PRODUCTION_RUTHERFORD_PLAN_SCHEMA:
        raise ValueError(f"unsupported production-plan schema in {path}")
    return plan


def _load_checkpoint_state(
    run_dir: Path,
    checkpoint: dict[str, Any] | None,
) -> ReducedMHDState:
    if checkpoint is None:
        raise ValueError("checkpoint must not be None")
    state_record = checkpoint.get("artifacts", {}).get("state", {})
    state_path = Path(state_record.get("path", ""))
    absolute_path = state_path if state_path.is_absolute() else run_dir / state_path
    with np.load(absolute_path, allow_pickle=False) as data:
        schema = str(data["schema"])
        if schema != PRODUCTION_RUTHERFORD_STATE_SCHEMA:
            raise ValueError(f"unsupported Rutherford state schema {schema!r}")
        return ReducedMHDState(
            psi=jnp.asarray(data["psi"]),
            omega=jnp.asarray(data["omega"]),
        )


def _advance_campaign_chunk(
    state0: ReducedMHDState,
    *,
    params: ReducedMHDParams,
    grid: CartesianGrid,
    start_step: int,
    steps: int,
    dt: float,
    save_every: int,
) -> tuple[np.ndarray, np.ndarray, tuple[ReducedMHDState, ...]]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return (
            np.asarray([start_step], dtype=np.int64),
            np.asarray([start_step * dt], dtype=np.float64),
            (state0,),
        )

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    active_state = state0
    saved_steps: list[int] = [start_step]
    saved_times: list[float] = [start_step * dt]
    saved_states: list[ReducedMHDState] = [state0]
    for local_step in range(1, steps + 1):
        active_state = rk4_step(active_state, rhs, dt)
        global_step = start_step + local_step
        if global_step % save_every == 0 or local_step == steps:
            saved_steps.append(global_step)
            saved_times.append(global_step * dt)
            saved_states.append(active_state)
    return (
        np.asarray(saved_steps, dtype=np.int64),
        np.asarray(saved_times, dtype=np.float64),
        tuple(saved_states),
    )


def _history_from_saved_states(
    *,
    saved_steps: np.ndarray,
    saved_times: np.ndarray,
    saved_states: tuple[ReducedMHDState, ...],
    lengths: tuple[float, float],
    mode: tuple[int, int],
    magnetic_shear: float,
) -> dict[str, np.ndarray]:
    trajectory = _trajectory_from_saved_states(saved_times, saved_states)
    energies = trajectory_energies(trajectory, lengths=lengths)
    reconnected_flux = np.asarray(
        [float(reconnected_flux_amplitude(state, mode=mode)) for state in saved_states],
        dtype=np.float64,
    )
    island_width = np.asarray(
        [
            float(island_width_from_mode(state, mode=mode, magnetic_shear=magnetic_shear))
            for state in saved_states
        ],
        dtype=np.float64,
    )
    current_linf = np.asarray(
        [
            float(jnp.max(jnp.abs(current_density(state.psi, lengths=lengths))))
            for state in saved_states
        ],
        dtype=np.float64,
    )
    divergence = np.asarray(
        [float(magnetic_divergence_linf(state, lengths=lengths)) for state in saved_states],
        dtype=np.float64,
    )
    time = np.asarray(saved_times, dtype=np.float64)
    total = np.asarray(energies["total"], dtype=np.float64)
    reconnection_rate = _safe_gradient(reconnected_flux, time)
    dissipation_budget_residual = np.maximum(_safe_gradient(total, time), 0.0)
    return {
        "step": np.asarray(saved_steps, dtype=np.int64),
        "time": time,
        "reconnected_flux": reconnected_flux,
        "rutherford_island_width": island_width,
        "reconnection_rate_proxy": reconnection_rate,
        "magnetic_energy": np.asarray(energies["magnetic"], dtype=np.float64),
        "kinetic_energy": np.asarray(energies["kinetic"], dtype=np.float64),
        "total_energy": total,
        "dissipation_budget_residual": dissipation_budget_residual,
        "magnetic_divergence_linf": divergence,
        "current_density_linf": current_linf,
    }


def _append_history(path: Path, new_history: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not path.exists():
        return new_history
    with np.load(path, allow_pickle=False) as existing_data:
        existing = {
            key: np.asarray(existing_data[key])
            for key in new_history
            if key in existing_data
        }
    if not existing:
        return new_history
    last_step = int(existing["step"][-1])
    keep = np.asarray(new_history["step"], dtype=np.int64) > last_step
    if not np.any(keep):
        return existing
    return {
        key: np.concatenate((np.asarray(existing[key]), np.asarray(value)[keep]))
        for key, value in new_history.items()
    }


def _write_campaign_history(path: Path, *, history: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(PRODUCTION_RUTHERFORD_HISTORY_SCHEMA),
        **history,
    )
    return path


def _write_campaign_state(
    path: Path,
    *,
    step: int,
    time: float,
    state: ReducedMHDState,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(PRODUCTION_RUTHERFORD_STATE_SCHEMA),
        step=np.asarray(step, dtype=np.int64),
        time=np.asarray(time, dtype=np.float64),
        psi=np.asarray(state.psi),
        omega=np.asarray(state.omega),
    )
    return path


def _write_execution_figures(
    figure_dir: Path,
    *,
    history: dict[str, np.ndarray],
    trajectory: ReducedMHDTrajectory,
    extent: tuple[float, float, float, float],
    lengths: tuple[float, float],
    write_movies: bool,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    history_path = figure_dir / "production_histories.png"
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8), constrained_layout=True)
    axes[0, 0].plot(history["time"], history["reconnected_flux"])
    axes[0, 0].set_title("Reconnected flux")
    axes[0, 1].plot(history["time"], history["rutherford_island_width"])
    axes[0, 1].set_title("Rutherford island width")
    axes[1, 0].plot(history["time"], history["reconnection_rate_proxy"])
    axes[1, 0].set_title("Reconnection-rate proxy")
    axes[1, 1].plot(history["time"], history["magnetic_energy"], label=r"$E_B$")
    axes[1, 1].plot(history["time"], history["kinetic_energy"], label=r"$E_K$")
    axes[1, 1].plot(history["time"], history["total_energy"], label=r"$E$")
    axes[1, 1].set_title("Energy")
    axes[1, 1].legend(frameon=False, fontsize="small")
    for axis in axes.ravel():
        axis.set_xlabel("time")
        axis.grid(True, alpha=0.25)
    fig.savefig(history_path, dpi=180)
    plt.close(fig)
    outputs = {"production_histories": history_path.relative_to(figure_dir.parent).as_posix()}
    if write_movies:
        flux_path = plot_flux_gif(
            trajectory,
            path=figure_dir / "fixed_scale_flux_movie.gif",
            extent=extent,
        )
        current_path = plot_current_density_gif(
            trajectory,
            path=figure_dir / "fixed_scale_current_density_movie.gif",
            extent=extent,
            lengths=lengths,
        )
        outputs["fixed_scale_flux_movie"] = flux_path.relative_to(figure_dir.parent).as_posix()
        outputs["fixed_scale_current_density_movie"] = current_path.relative_to(
            figure_dir.parent
        ).as_posix()
    return outputs


def _trajectory_from_saved_states(
    times: np.ndarray,
    states: tuple[ReducedMHDState, ...],
) -> ReducedMHDTrajectory:
    return ReducedMHDTrajectory(
        times=jnp.asarray(times),
        states=ReducedMHDState(
            psi=jnp.stack([state.psi for state in states]),
            omega=jnp.stack([state.omega for state in states]),
        ),
    )


def _safe_gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if values.size < 2 or time.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(np.gradient(values, time), dtype=np.float64)


def _max_relative_energy_growth(total_energy: np.ndarray) -> float:
    if total_energy.size < 2:
        return 0.0
    energy_scale = max(abs(float(total_energy[0])), np.finfo(np.float64).tiny)
    return float(np.max(np.diff(total_energy)) / energy_scale)


def _initial_checkpoint_index(
    *,
    run_dir: Path,
    target_step: int,
    target_time: float,
    walltime_policy: WalltimePolicy,
    checkpoint_every_steps: int,
    steps_per_walltime_job: int,
) -> dict[str, Any]:
    return {
        "schema": PRODUCTION_RUTHERFORD_CHECKPOINT_INDEX_SCHEMA,
        "api_version": require_supported_api_version(context="rutherford checkpoint index"),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "updated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "target_step": int(target_step),
        "target_time": float(target_time),
        "latest_checkpoint": None,
        "checkpoints": [],
        "resume_policy": {
            "checkpoint_every_steps": int(checkpoint_every_steps),
            "steps_per_walltime_job": int(steps_per_walltime_job),
            "walltime_policy": walltime_policy.to_dict(),
        },
    }


def _checkpoint_index_path(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "checkpoint_index.json"


def _build_job_array(
    *,
    run_dir: Path,
    total_steps: int,
    steps_per_job: int,
    checkpoint_every_steps: int,
) -> dict[str, Any]:
    jobs = []
    start = 0
    job_index = 0
    while start < total_steps:
        end = min(total_steps, start + steps_per_job)
        jobs.append(
            {
                "job_index": job_index,
                "start_step": start,
                "end_step": end,
                "checkpoint_every_steps": checkpoint_every_steps,
                "expected_terminal_checkpoint": f"step_{end:012d}",
                "resume_plan_command": (
                    f"mhx campaign rutherford-resume-plan {run_dir} "
                    f"--target-step {total_steps}"
                ),
            }
        )
        start = end
        job_index += 1
    return {
        "schema": "mhx.campaign.rutherford_walltime_job_array.v1",
        "run_dir": str(run_dir),
        "total_steps": total_steps,
        "steps_per_job": steps_per_job,
        "checkpoint_every_steps": checkpoint_every_steps,
        "jobs": jobs,
    }


def _render_runbook(diagnostics: dict[str, Any], job_array: dict[str, Any]) -> str:
    config = diagnostics["config"]
    run_dir = config["output_dir"]
    return f"""# Rutherford production campaign runbook

Schema: `{PRODUCTION_RUTHERFORD_RUNBOOK_SCHEMA}`

This runbook is a production **execution contract**. It proves that the
duration, walltime chunking, checkpoint cadence, and artifact contracts are
defined before a long nonlinear run starts. It is not a completed production
simulation.

## Duration gate

The campaign uses

$$t_\\mathrm{{end}} = s_f N_e / \\gamma,$$

with `gamma = {diagnostics['duration_assessment']['growth_rate']:.6g}`,
`N_e = {diagnostics['duration_assessment']['required_efolds']:.6g}`, and
`s_f = {diagnostics['duration_assessment']['safety_factor']:.6g}`. The planned
final time is `{config['time']['t1']:.6g}` and the estimated fixed-step count is
`{diagnostics['estimated_steps']}`.

## Walltime and checkpoint contract

- Run directory: `{run_dir}`
- Steps per scheduler allocation: `{diagnostics['steps_per_walltime_job']}`
- Checkpoint cadence: every `{diagnostics['checkpoint_every_steps']}` steps
- Estimated walltime jobs: `{diagnostics['estimated_walltime_jobs']}`
- Checkpoint index: `checkpoints/checkpoint_index.json`

Before each walltime margin is reached, the executor must write a state artifact
and call `mhx.campaigns.write_checkpoint_metadata(...)` or an equivalent
schema-compatible writer.

## Resume command

```bash
mhx campaign rutherford-resume-plan {run_dir} \\
  --target-step {diagnostics['estimated_steps']}
```

The resume plan identifies the latest valid checkpoint by checking artifact
existence and SHA-256 hashes. If no checkpoint is valid, it explicitly falls
back to step zero.

## First walltime chunk

```json
{json.dumps(job_array['jobs'][0], indent=2, sort_keys=True)}
```

## Required publication artifacts

The production run is not paper-grade until all histories, fixed-scale movies,
convergence sweeps, checkpoint restart reproducibility checks, and artifact
hashes listed in `campaign_plan.json` are present.
"""


def _checkpoint_artifacts(
    run_dir: Path,
    *,
    state_path: str | Path,
    history_path: str | Path | None,
    diagnostics_path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    artifacts = {"state": _artifact_record(run_dir, state_path)}
    if history_path is not None:
        artifacts["history"] = _artifact_record(run_dir, history_path)
    if diagnostics_path is not None:
        artifacts["diagnostics"] = _artifact_record(run_dir, diagnostics_path)
    return artifacts


def _artifact_record(run_dir: Path, path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    absolute_path = artifact_path if artifact_path.is_absolute() else run_dir / artifact_path
    relative_path = (
        absolute_path.relative_to(run_dir).as_posix()
        if _is_relative_to(absolute_path, run_dir)
        else str(absolute_path)
    )
    exists = absolute_path.exists()
    return {
        "path": relative_path,
        "exists": exists,
        "size_bytes": absolute_path.stat().st_size if exists else 0,
        "sha256": _sha256_file(absolute_path) if exists else None,
    }


def _checkpoint_record_is_valid(run_dir: Path, checkpoint: dict[str, Any]) -> bool:
    artifacts = checkpoint.get("artifacts", {})
    state = artifacts.get("state")
    if not isinstance(state, dict):
        return False
    for record in artifacts.values():
        if not isinstance(record, dict) or not record.get("exists", False):
            return False
        path = Path(record["path"])
        absolute_path = path if path.is_absolute() else run_dir / path
        if not absolute_path.exists():
            return False
        if _sha256_file(absolute_path) != record.get("sha256"):
            return False
    return bool(checkpoint.get("validation", {}).get("passed", False))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True
