#!/usr/bin/env python3
"""Dry-run-first nonlinear production-campaign launcher.

The script writes an exact command manifest for a production nonlinear campaign
lane before anything expensive is run.  ``--execute`` is the only mode that
launches commands, and each launched command is bounded by a subprocess timeout.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "mhx.tools.nonlinear_production_campaign.v1"
GATES_SCHEMA = "mhx.tools.nonlinear_production_campaign.gates.v1"
DEFAULT_OUTDIR = Path("outputs/campaigns/nonlinear_production_campaign")
DEFAULT_HARRIS_GROWTH_RATE = 1.31e-2
DEFAULT_PRODUCTION_EFOLDS = 10.0
DEFAULT_SAFETY_FACTOR = 3.0


@dataclass(frozen=True)
class CampaignOptions:
    """Options that define the generated production-campaign lane."""

    outdir: Path = DEFAULT_OUTDIR
    python_executable: str = sys.executable
    harris_growth_rate: float = DEFAULT_HARRIS_GROWTH_RATE
    production_efolds: float = DEFAULT_PRODUCTION_EFOLDS
    safety_factor: float = DEFAULT_SAFETY_FACTOR
    nx: int = 256
    ny: int = 256
    dt: float = 2.0e-2
    target_saved_frames: int = 400
    save_every: int | None = None
    save_interval: float | None = None
    fit_stop: float | None = None
    width: float = 0.36
    widths: tuple[float, ...] = (0.32, 0.36, 0.40)
    eta: float = 4.5e-3
    nu: float | None = None
    etas: tuple[float, ...] = (6.0e-3, 4.5e-3, 3.0e-3)
    viscosities: tuple[float, ...] | None = None
    perturbation_amplitude: float = 4.0e-3
    mode_x: int = 2
    mode_y: int = 1
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    convergence_resolutions: tuple[int, ...] = (128, 192, 256)
    crosscheck_resolutions: tuple[int, ...] = (160, 224, 256)
    dt_values: tuple[float, ...] = (2.0e-2, 1.0e-2)
    crosscheck_dt_values: tuple[float, ...] = (2.0e-2, 1.0e-2)
    max_walltime_hours: float = 12.0
    seconds_per_step_estimate: float = 0.5
    checkpoint_interval_minutes: float = 30.0
    preemption_margin_minutes: float = 20.0
    min_production_resolution: int = 128
    timeout_seconds: int = 43_200
    gate_timeout_seconds: int = 1_800
    max_relative_energy_growth: float = 1.0e-6
    max_relative_energy_increase: float = 1.0e-8
    max_divergence_linf: float = 1.0e-8
    min_history_samples: int = 100
    min_reconnected_flux_amplification: float = 1.05
    min_island_width_amplification: float = 1.05
    noise_amplitude: float = 1.0e-6

    @property
    def t_end(self) -> float:
        """Duration required by the Harris e-fold policy."""
        return self.production_efolds * self.safety_factor / self.harris_growth_rate

    @property
    def viscosity(self) -> float:
        """Baseline viscosity, defaulting to magnetic Prandtl number one."""
        return self.eta if self.nu is None else self.nu

    @property
    def eta_viscosities(self) -> tuple[float, ...]:
        """Viscosity values paired with the resistivity/Lundquist sweep."""
        return self.etas if self.viscosities is None else self.viscosities

    @property
    def lundquist(self) -> float:
        """Nominal Lundquist-number proxy for the normalized double-Harris lane."""
        return 1.0 / self.eta

    @property
    def lundquist_values(self) -> tuple[float, ...]:
        """Lundquist-number proxies paired with ``etas``."""
        return tuple(1.0 / value for value in self.etas)

    @property
    def resolved_save_every(self) -> int:
        """Saved-step stride for the long double-Harris command."""
        if self.save_every is not None:
            return self.save_every
        total_steps = max(1, math.ceil(self.t_end / self.dt))
        return max(1, math.ceil(total_steps / self.target_saved_frames))

    @property
    def resolved_save_interval(self) -> float:
        """Physical save interval for convergence and parameter-sweep commands."""
        if self.save_interval is not None:
            return self.save_interval
        return self.resolved_save_every * self.dt

    @property
    def resolved_fit_stop(self) -> float:
        """Early growth-fit stop that remains inside the linear-growth window.

        The campaign duration may be many e-folding times long.  Fitting through
        the full nonlinear/saturated trajectory can make a physically growing
        run fail the early-growth gate.  The default therefore uses the larger
        of three saved samples or two declared Harris e-folding times, capped at
        the total duration.  Users can still override this with ``--fit-stop``.
        """
        if self.fit_stop is not None:
            return self.fit_stop
        minimum_sample_span = 3.0 * self.resolved_save_interval
        two_efold_span = 2.0 / self.harris_growth_rate
        return min(self.t_end, max(minimum_sample_span, two_efold_span))

    def validated(self) -> CampaignOptions:
        """Validate campaign options and return ``self`` for fluent construction."""
        if self.harris_growth_rate <= 0.0:
            raise ValueError("harris_growth_rate must be positive")
        if self.production_efolds <= 0.0:
            raise ValueError("production_efolds must be positive")
        if self.safety_factor <= 0.0:
            raise ValueError("safety_factor must be positive")
        if min(self.nx, self.ny) < self.min_production_resolution:
            raise ValueError("nx and ny must meet min_production_resolution")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.target_saved_frames < 3:
            raise ValueError("target_saved_frames must be at least three")
        if self.save_every is not None and self.save_every < 1:
            raise ValueError("save_every must be positive")
        if self.save_interval is not None and self.save_interval <= 0.0:
            raise ValueError("save_interval must be positive")
        if self.resolved_save_interval > self.t_end:
            raise ValueError("save_interval must not exceed the campaign duration")
        if self.fit_stop is not None and not 0.0 < self.fit_stop <= self.t_end:
            raise ValueError("fit_stop must be inside the campaign duration")
        if self.width <= 0.0 or any(value <= 0.0 for value in self.widths):
            raise ValueError("widths must be positive")
        if len(self.widths) < 2 or len(set(self.widths)) != len(self.widths):
            raise ValueError("widths must contain at least two unique entries")
        if self.eta <= 0.0 or any(value <= 0.0 for value in self.etas):
            raise ValueError("eta values must be positive")
        if self.viscosity <= 0.0 or any(value <= 0.0 for value in self.eta_viscosities):
            raise ValueError("viscosity values must be positive")
        if len(self.etas) < 2 or len(set(self.etas)) != len(self.etas):
            raise ValueError("etas must contain at least two unique entries")
        if len(self.eta_viscosities) != len(self.etas):
            raise ValueError("viscosities must match etas when provided")
        if self.perturbation_amplitude <= 0.0:
            raise ValueError("perturbation_amplitude must be positive")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain at least two unique entries")
        if len(self.convergence_resolutions) < 2 or len(self.crosscheck_resolutions) < 2:
            raise ValueError("convergence resolution lists must contain at least two entries")
        if min((*self.convergence_resolutions, *self.crosscheck_resolutions)) < 8:
            raise ValueError("convergence resolutions must be at least eight")
        if len(self.dt_values) < 2 or len(self.crosscheck_dt_values) < 2:
            raise ValueError("dt sweep lists must contain at least two entries")
        if any(value <= 0.0 for value in (*self.dt_values, *self.crosscheck_dt_values)):
            raise ValueError("dt sweep values must be positive")
        if self.timeout_seconds <= 0 or self.gate_timeout_seconds <= 0:
            raise ValueError("timeout controls must be positive")
        if self.max_walltime_hours <= 0.0:
            raise ValueError("max_walltime_hours must be positive")
        if self.seconds_per_step_estimate <= 0.0:
            raise ValueError("seconds_per_step_estimate must be positive")
        if self.checkpoint_interval_minutes <= 0.0:
            raise ValueError("checkpoint_interval_minutes must be positive")
        if self.preemption_margin_minutes < 0.0:
            raise ValueError("preemption_margin_minutes must be non-negative")
        if self.min_history_samples < 1:
            raise ValueError("min_history_samples must be positive")
        if self.min_reconnected_flux_amplification < 1.0:
            raise ValueError("min_reconnected_flux_amplification must be at least one")
        if self.min_island_width_amplification < 1.0:
            raise ValueError("min_island_width_amplification must be at least one")
        return self


@dataclass(frozen=True)
class CommandSpec:
    """One exact command in the generated campaign lane."""

    command_id: str
    stage: str
    family: str
    gate: str
    description: str
    argv: tuple[str, ...]
    timeout_seconds: int
    expensive: bool
    expected_outputs: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible command record."""
        return {
            "id": self.command_id,
            "stage": self.stage,
            "family": self.family,
            "gate": self.gate,
            "description": self.description,
            "command": list(self.argv),
            "shell": shlex.join(self.argv),
            "timeout_seconds": self.timeout_seconds,
            "expensive": self.expensive,
            "expected_outputs": list(self.expected_outputs),
            "claim_boundary": self.claim_boundary,
        }


def build_manifest(options: CampaignOptions, *, mode: str = "dry_run") -> dict[str, Any]:
    """Build the exact command manifest without running commands."""
    validated_options = options.validated()
    if mode not in {"dry_run", "execute"}:
        raise ValueError("mode must be dry_run or execute")
    commands = build_command_specs(validated_options)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "mode": mode,
        "claim_level": "validation",
        "claim_boundary": (
            "Automation manifest and command lane only. Generated commands remain "
            "validation artifacts unless the Rutherford target completes, promotion "
            "readiness passes with convergence/seed/movie/response evidence, and the "
            "explicit final production-claim command succeeds."
        ),
        "production_claim_policy": {
            "default_claim_level": "validation",
            "double_harris_boundary": (
                "The double-Harris promotion checker can promote media to "
                "convergence-backed validation only; it does not authorize a "
                "Rutherford, Sweet-Parker, or plasmoid-chain production claim."
            ),
            "rutherford_boundary": (
                "Rutherford output can become production only after duration "
                "completion, passing promotion gates, and an explicit "
                "--allow-production-claim finalize command."
            ),
            "production_finalize_command_id": "rutherford_finalize_production_claim",
        },
        "campaign": _campaign_metadata(validated_options),
        "required_gates": _required_gates(),
        "commands": [command.to_dict() for command in commands],
        "execution": {
            "requested": mode == "execute",
            "status": "not_started" if mode == "execute" else "skipped_dry_run",
            "results": [],
        },
        "validation": _manifest_validation(validated_options),
    }
    return manifest


def build_command_specs(options: CampaignOptions) -> tuple[CommandSpec, ...]:
    """Return exact commands for the nonlinear production campaign lane."""
    root = options.outdir
    rutherford_dir = root / "rutherford"
    double_harris_dir = root / "double_harris"
    long_run_dir = double_harris_dir / "long_run"
    convergence_dir = double_harris_dir / "convergence_main"
    convergence_crosscheck_dir = double_harris_dir / "convergence_crosscheck"
    width_sweep_dir = double_harris_dir / "width_sweep"
    eta_sweep_dir = double_harris_dir / "eta_lundquist_sweep"
    seed_qi_dir = root / "seed_qi"

    return (
        _rutherford_plan_command(options, rutherford_dir),
        _double_harris_long_run_command(options, long_run_dir),
        _double_harris_convergence_command(
            options,
            convergence_dir,
            command_id="double_harris_convergence_main",
            description="Production-resolution/time-step convergence bundle.",
            resolutions=options.convergence_resolutions,
            dt_values=options.dt_values,
        ),
        _double_harris_convergence_command(
            options,
            convergence_crosscheck_dir,
            command_id="double_harris_convergence_crosscheck",
            description="Independent convergence cross-check bundle.",
            resolutions=options.crosscheck_resolutions,
            dt_values=options.crosscheck_dt_values,
        ),
        _double_harris_parameter_sweep_command(
            options,
            width_sweep_dir,
            command_id="double_harris_width_sweep",
            sweep_axis="width",
            description="Sheet-width robustness sweep paired with aspect-ratio promotion gates.",
        ),
        _double_harris_parameter_sweep_command(
            options,
            eta_sweep_dir,
            command_id="double_harris_eta_lundquist_sweep",
            sweep_axis="resistivity",
            description="Eta/Lundquist-number robustness sweep with Pm-matched viscosity.",
        ),
        _seed_qi_command(options, seed_qi_dir),
        _rutherford_execute_command(options, rutherford_dir),
        _double_harris_promotion_command(
            options,
            long_run_dir=long_run_dir,
            convergence_dirs=(convergence_dir, convergence_crosscheck_dir),
        ),
        _rutherford_promotion_command(
            options,
            rutherford_dir=rutherford_dir,
            convergence_dirs=(convergence_dir, convergence_crosscheck_dir),
            seed_qi_dir=seed_qi_dir,
        ),
        _rutherford_finalize_command(options, rutherford_dir),
    )


def write_manifest_outputs(manifest: dict[str, Any], outdir: Path) -> tuple[Path, Path]:
    """Write the JSON manifest and executable shell command list."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "production_campaign_manifest.json"
    commands_path = output_dir / "run_commands.sh"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commands_path.write_text(_render_shell_script(manifest), encoding="utf-8")
    commands_path.chmod(commands_path.stat().st_mode | 0o111)
    return manifest_path, commands_path


def execute_manifest(manifest: dict[str, Any], *, manifest_path: Path | None = None) -> int:
    """Execute manifest commands sequentially with per-command timeouts."""
    manifest["execution"]["requested"] = True
    manifest["execution"]["status"] = "running"
    manifest["execution"]["results"] = []
    _write_manifest_if_requested(manifest, manifest_path)
    for command in manifest["commands"]:
        started = datetime.now(tz=timezone.utc)
        result: dict[str, Any] = {
            "id": command["id"],
            "started_utc": started.isoformat(),
            "timeout_seconds": int(command["timeout_seconds"]),
        }
        try:
            completed = subprocess.run(
                command["command"],
                check=False,
                timeout=int(command["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired:
            result.update(
                {
                    "status": "timeout",
                    "returncode": 124,
                    "finished_utc": datetime.now(tz=timezone.utc).isoformat(),
                }
            )
            manifest["execution"]["results"].append(result)
            manifest["execution"]["status"] = "failed"
            _write_manifest_if_requested(manifest, manifest_path)
            return 124
        result.update(
            {
                "status": "passed" if completed.returncode == 0 else "failed",
                "returncode": int(completed.returncode),
                "finished_utc": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        manifest["execution"]["results"].append(result)
        if completed.returncode != 0:
            manifest["execution"]["status"] = "failed"
            _write_manifest_if_requested(manifest, manifest_path)
            return int(completed.returncode)
        _write_manifest_if_requested(manifest, manifest_path)
    manifest["execution"]["status"] = "passed"
    _write_manifest_if_requested(manifest, manifest_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    mode = "execute" if args.execute else "dry_run"
    options = _options_from_args(args)
    manifest = build_manifest(options, mode=mode)
    manifest_path, commands_path = write_manifest_outputs(manifest, options.outdir)
    print(f"wrote {manifest_path}")
    print(f"wrote {commands_path}")
    if not args.execute:
        print("dry run only; pass --execute to launch commands with subprocess timeouts")
        return 0
    return execute_manifest(manifest, manifest_path=manifest_path)


def _campaign_metadata(options: CampaignOptions) -> dict[str, Any]:
    return {
        "root": str(options.outdir),
        "python_executable": options.python_executable,
        "duration": {
            "harris_growth_rate": options.harris_growth_rate,
            "production_efolds": options.production_efolds,
            "safety_factor": options.safety_factor,
            "t_end": options.t_end,
            "duration_formula": "production_efolds * safety_factor / harris_growth_rate",
        },
        "grid": {
            "nx": options.nx,
            "ny": options.ny,
            "min_production_resolution": options.min_production_resolution,
        },
        "time": {
            "dt": options.dt,
            "target_saved_frames": options.target_saved_frames,
            "save_every": options.resolved_save_every,
            "save_interval": options.resolved_save_interval,
            "fit_start": 0.0,
            "fit_stop": options.resolved_fit_stop,
        },
        "physics": {
            "width": options.width,
            "widths": list(options.widths),
            "eta": options.eta,
            "nu": options.viscosity,
            "lundquist_proxy": options.lundquist,
            "etas": list(options.etas),
            "eta_lundquist_proxy": list(options.lundquist_values),
            "viscosities": list(options.eta_viscosities),
            "perturbation_amplitude": options.perturbation_amplitude,
            "mode": [options.mode_x, options.mode_y],
        },
        "seed_qi": {
            "seeds": list(options.seeds),
            "noise_amplitude": options.noise_amplitude,
        },
        "timeouts": {
            "expensive_command_seconds": options.timeout_seconds,
            "gate_command_seconds": options.gate_timeout_seconds,
        },
    }


def _manifest_validation(options: CampaignOptions) -> dict[str, Any]:
    checks = {
        "dry_run_first_contract": True,
        "duration_policy_positive": options.t_end > 0.0,
        "production_resolution_floor": min(options.nx, options.ny)
        >= options.min_production_resolution,
        "convergence_has_two_bundles": True,
        "seed_qi_has_multiple_seeds": len(options.seeds) >= 2,
        "width_sweep_declared": len(options.widths) >= 2,
        "eta_lundquist_sweep_declared": len(options.etas) >= 2,
        "movie_commands_enabled": True,
        "promotion_commands_declared": True,
        "production_claim_requires_finalize_command": True,
    }
    return {
        "schema": GATES_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "claim_level_unless_passed": "validation",
    }


def _required_gates() -> list[dict[str, Any]]:
    return [
        {
            "gate": "duration",
            "command_ids": ["rutherford_plan", "double_harris_long_run"],
            "required": True,
            "evidence": (
                "Rutherford duration plan and double-Harris run reach the same "
                "policy time."
            ),
        },
        {
            "gate": "convergence",
            "command_ids": [
                "double_harris_convergence_main",
                "double_harris_convergence_crosscheck",
            ],
            "required": True,
            "evidence": "Two convergence bundles are supplied to promotion checks.",
        },
        {
            "gate": "seed_qi",
            "command_ids": ["seed_qi"],
            "required": True,
            "evidence": "Seed-robust QI bundle is supplied to Rutherford promotion.",
        },
        {
            "gate": "width_aspect",
            "command_ids": ["double_harris_width_sweep", "rutherford_promotion_check"],
            "required": True,
            "evidence": "Sheet-width sweep plus Rutherford current-sheet aspect-ratio histories.",
        },
        {
            "gate": "eta_lundquist",
            "command_ids": ["double_harris_eta_lundquist_sweep"],
            "required": True,
            "evidence": "Eta sweep records normalized Lundquist-number proxies.",
        },
        {
            "gate": "movie",
            "command_ids": ["double_harris_long_run", "rutherford_execute"],
            "required": True,
            "evidence": "Fixed-scale flux/current movies are required by promotion checks.",
        },
        {
            "gate": "promotion",
            "command_ids": [
                "double_harris_promotion_check",
                "rutherford_promotion_check",
                "rutherford_finalize_production_claim",
            ],
            "required": True,
            "evidence": "Production claim remains blocked until promotion and finalize pass.",
        },
    ]


def _rutherford_plan_command(options: CampaignOptions, outdir: Path) -> CommandSpec:
    argv = _mhx(
        options,
        "campaign",
        "rutherford-plan-production",
        "--outdir",
        outdir,
        "--harris-growth-rate",
        options.harris_growth_rate,
        "--production-efolds",
        options.production_efolds,
        "--safety-factor",
        options.safety_factor,
        "--nx",
        options.nx,
        "--ny",
        options.ny,
        "--dt",
        options.dt,
        "--target-saved-frames",
        options.target_saved_frames,
        "--max-walltime-hours",
        options.max_walltime_hours,
        "--seconds-per-step-estimate",
        options.seconds_per_step_estimate,
        "--checkpoint-interval-minutes",
        options.checkpoint_interval_minutes,
        "--preemption-margin-minutes",
        options.preemption_margin_minutes,
        "--min-production-resolution",
        options.min_production_resolution,
    )
    return CommandSpec(
        command_id="rutherford_plan",
        stage="plan",
        family="Rutherford executor",
        gate="duration",
        description="Write duration-guarded Rutherford production plan and checkpoint contract.",
        argv=argv,
        timeout_seconds=options.gate_timeout_seconds,
        expensive=False,
        expected_outputs=(
            str(outdir / "campaign_plan.json"),
            str(outdir / "runbook.md"),
            str(outdir / "manifest.json"),
        ),
        claim_boundary="Planning artifact only; not a completed simulation.",
    )


def _double_harris_long_run_command(options: CampaignOptions, outdir: Path) -> CommandSpec:
    argv = _mhx(
        options,
        "benchmark",
        "double-harris-long-run",
        "--outdir",
        outdir,
        "--nx",
        options.nx,
        "--ny",
        options.ny,
        "--width",
        options.width,
        "--eta",
        options.eta,
        "--nu",
        options.viscosity,
        "--perturbation-amplitude",
        options.perturbation_amplitude,
        "--mode-x",
        options.mode_x,
        "--mode-y",
        options.mode_y,
        "--dt",
        options.dt,
        "--t-end",
        options.t_end,
        "--save-every",
        options.resolved_save_every,
        "--fit-start",
        0.0,
        "--fit-stop",
        options.resolved_fit_stop,
        "--min-reconnected-flux-amplification",
        options.min_reconnected_flux_amplification,
        "--min-island-width-amplification",
        options.min_island_width_amplification,
        "--movies",
    )
    return CommandSpec(
        command_id="double_harris_long_run",
        stage="execute",
        family="periodic double-Harris",
        gate="duration,movie",
        description="Run the production-duration seeded double-Harris movie bundle.",
        argv=argv,
        timeout_seconds=options.timeout_seconds,
        expensive=True,
        expected_outputs=(
            str(outdir / "periodic_double_harris_seeded_long_run.npz"),
            str(outdir / "figures" / "periodic_double_harris_flux.gif"),
            str(outdir / "figures" / "periodic_double_harris_current.gif"),
            str(outdir / "manifest.json"),
        ),
        claim_boundary=(
            "Double-Harris output is validation evidence until convergence, seed, "
            "width/aspect, eta/Lundquist, movie, and promotion gates pass."
        ),
    )


def _double_harris_convergence_command(
    options: CampaignOptions,
    outdir: Path,
    *,
    command_id: str,
    description: str,
    resolutions: tuple[int, ...],
    dt_values: tuple[float, ...],
) -> CommandSpec:
    argv = _mhx(
        options,
        "benchmark",
        "double-harris-convergence",
        "--outdir",
        outdir,
        "--resolutions",
        _csv(resolutions),
        "--dt-values",
        _csv(dt_values),
        "--reference-resolution",
        options.nx,
        "--reference-dt",
        options.dt,
        "--width",
        options.width,
        "--eta",
        options.eta,
        "--nu",
        options.viscosity,
        "--perturbation-amplitude",
        options.perturbation_amplitude,
        "--mode-x",
        options.mode_x,
        "--mode-y",
        options.mode_y,
        "--t-end",
        options.t_end,
        "--save-interval",
        options.resolved_save_interval,
        "--fit-start",
        0.0,
        "--fit-stop",
        options.resolved_fit_stop,
    )
    return CommandSpec(
        command_id=command_id,
        stage="convergence",
        family="periodic double-Harris",
        gate="convergence",
        description=description,
        argv=argv,
        timeout_seconds=options.timeout_seconds,
        expensive=True,
        expected_outputs=(
            str(outdir / "periodic_double_harris_convergence.npz"),
            str(outdir / "validation.json"),
            str(outdir / "artifact_manifest.json"),
        ),
        claim_boundary="Convergence evidence only; not a standalone production claim.",
    )


def _double_harris_parameter_sweep_command(
    options: CampaignOptions,
    outdir: Path,
    *,
    command_id: str,
    sweep_axis: str,
    description: str,
) -> CommandSpec:
    axis_arguments: tuple[Any, ...]
    gate = "width_aspect" if sweep_axis == "width" else "eta_lundquist"
    if sweep_axis == "width":
        axis_arguments = ("--widths", _csv(options.widths))
    else:
        axis_arguments = (
            "--etas",
            _csv(options.etas),
            "--viscosities",
            _csv(options.eta_viscosities),
        )
    argv = _mhx(
        options,
        "benchmark",
        "double-harris-parameter-sweep",
        "--outdir",
        outdir,
        "--sweep-axis",
        sweep_axis,
        *axis_arguments,
        "--nx",
        options.nx,
        "--ny",
        options.ny,
        "--width",
        options.width,
        "--eta",
        options.eta,
        "--nu",
        options.viscosity,
        "--perturbation-amplitude",
        options.perturbation_amplitude,
        "--mode-x",
        options.mode_x,
        "--mode-y",
        options.mode_y,
        "--dt",
        options.dt,
        "--t-end",
        options.t_end,
        "--save-interval",
        options.resolved_save_interval,
        "--fit-start",
        0.0,
        "--fit-stop",
        options.resolved_fit_stop,
    )
    return CommandSpec(
        command_id=command_id,
        stage="sweep",
        family="periodic double-Harris",
        gate=gate,
        description=description,
        argv=argv,
        timeout_seconds=options.timeout_seconds,
        expensive=True,
        expected_outputs=(
            str(outdir / "periodic_double_harris_parameter_sweep.npz"),
            str(outdir / "validation.json"),
            str(outdir / "artifact_manifest.json"),
        ),
        claim_boundary="Parameter robustness evidence only; not a standalone production claim.",
    )


def _seed_qi_command(options: CampaignOptions, outdir: Path) -> CommandSpec:
    argv = _mhx(
        options,
        "benchmark",
        "seed-robust-qi",
        "--outdir",
        outdir,
        "--seeds",
        _csv(options.seeds),
        "--nx",
        options.nx,
        "--ny",
        options.ny,
        "--t-end",
        options.t_end,
        "--dt",
        options.dt,
        "--save-every",
        options.resolved_save_every,
        "--eta",
        options.eta,
        "--nu",
        options.viscosity,
        "--noise-amplitude",
        options.noise_amplitude,
    )
    return CommandSpec(
        command_id="seed_qi",
        stage="seed_qi",
        family="quality indicators",
        gate="seed_qi",
        description="Run seed-robust quality indicators used by the promotion lane.",
        argv=argv,
        timeout_seconds=options.timeout_seconds,
        expensive=True,
        expected_outputs=(
            str(outdir / "seed_robust_qi.npz"),
            str(outdir / "validation.json"),
            str(outdir / "manifest.json"),
        ),
        claim_boundary="Seed robustness evidence only; not a production physics claim.",
    )


def _rutherford_execute_command(options: CampaignOptions, run_dir: Path) -> CommandSpec:
    argv = _mhx(
        options,
        "campaign",
        "rutherford-execute",
        run_dir,
        "--seed",
        options.seeds[0],
        "--noise-amplitude",
        options.noise_amplitude,
        "--movies",
        "--max-relative-energy-growth",
        options.max_relative_energy_growth,
        "--max-divergence-linf",
        options.max_divergence_linf,
    )
    return CommandSpec(
        command_id="rutherford_execute",
        stage="execute",
        family="Rutherford executor",
        gate="duration,movie",
        description="Run the full Rutherford target duration with fixed-scale movies.",
        argv=argv,
        timeout_seconds=options.timeout_seconds,
        expensive=True,
        expected_outputs=(
            str(run_dir / "production_history.npz"),
            str(run_dir / "figures" / "fixed_scale_flux_movie.gif"),
            str(run_dir / "figures" / "fixed_scale_current_density_movie.gif"),
            str(run_dir / "manifest.json"),
        ),
        claim_boundary=(
            "Execution remains validation unless target completion, promotion report, "
            "and explicit production-claim finalize gates pass."
        ),
    )


def _double_harris_promotion_command(
    options: CampaignOptions,
    *,
    long_run_dir: Path,
    convergence_dirs: tuple[Path, ...],
) -> CommandSpec:
    convergence_args = _repeated_path_args("--convergence-dir", convergence_dirs)
    argv = _mhx(
        options,
        "benchmark",
        "double-harris-promotion-check",
        long_run_dir,
        *convergence_args,
        "--min-convergence-dirs",
        len(convergence_dirs),
        "--min-history-samples",
        options.min_history_samples,
        "--min-t-end",
        options.t_end,
        "--min-reconnected-flux-amplification",
        options.min_reconnected_flux_amplification,
        "--min-island-width-amplification",
        options.min_island_width_amplification,
        "--max-relative-energy-increase",
        options.max_relative_energy_increase,
    )
    return CommandSpec(
        command_id="double_harris_promotion_check",
        stage="promotion",
        family="periodic double-Harris",
        gate="promotion",
        description="Verify double-Harris media/convergence promotion boundaries.",
        argv=argv,
        timeout_seconds=options.gate_timeout_seconds,
        expensive=False,
        expected_outputs=(
            str(long_run_dir / "promotion" / "promotion_readiness.json"),
            str(long_run_dir / "promotion" / "manifest.json"),
        ),
        claim_boundary="Passing promotes only to convergence-backed validation media.",
    )


def _rutherford_promotion_command(
    options: CampaignOptions,
    *,
    rutherford_dir: Path,
    convergence_dirs: tuple[Path, ...],
    seed_qi_dir: Path,
) -> CommandSpec:
    convergence_args = _repeated_path_args("--convergence-dir", convergence_dirs)
    argv = _mhx(
        options,
        "campaign",
        "rutherford-promotion-check",
        rutherford_dir,
        *convergence_args,
        "--seed-qi-dir",
        seed_qi_dir,
        "--min-convergence-dirs",
        len(convergence_dirs),
        "--min-history-samples",
        options.min_history_samples,
        "--max-energy-budget-residual",
        options.max_relative_energy_growth,
        "--max-divergence-linf",
        options.max_divergence_linf,
        "--min-reconnected-flux-amplification",
        options.min_reconnected_flux_amplification,
        "--min-island-width-amplification",
        options.min_island_width_amplification,
    )
    return CommandSpec(
        command_id="rutherford_promotion_check",
        stage="promotion",
        family="Rutherford executor",
        gate="promotion",
        description="Verify Rutherford production-readiness gates.",
        argv=argv,
        timeout_seconds=options.gate_timeout_seconds,
        expensive=False,
        expected_outputs=(
            str(rutherford_dir / "promotion" / "promotion_readiness.json"),
            str(rutherford_dir / "promotion" / "manifest.json"),
        ),
        claim_boundary="Failing checks keep the Rutherford bundle at validation level.",
    )


def _rutherford_finalize_command(options: CampaignOptions, run_dir: Path) -> CommandSpec:
    argv = _mhx(
        options,
        "campaign",
        "rutherford-execute",
        run_dir,
        "--max-steps",
        0,
        "--seed",
        options.seeds[0],
        "--noise-amplitude",
        options.noise_amplitude,
        "--allow-production-claim",
        "--max-relative-energy-growth",
        options.max_relative_energy_growth,
        "--max-divergence-linf",
        options.max_divergence_linf,
    )
    return CommandSpec(
        command_id="rutherford_finalize_production_claim",
        stage="promotion",
        family="Rutherford executor",
        gate="promotion",
        description="Refresh Rutherford manifest with production claim only if gates passed.",
        argv=argv,
        timeout_seconds=options.gate_timeout_seconds,
        expensive=False,
        expected_outputs=(str(run_dir / "manifest.json"), str(run_dir / "validation.json")),
        claim_boundary="Can emit production only after the promotion report already passes.",
    )


def _mhx(options: CampaignOptions, *parts: Any) -> tuple[str, ...]:
    return (
        options.python_executable,
        "-m",
        "mhx.cli.main",
        *(str(part) if isinstance(part, Path) else _format_value(part) for part in parts),
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _csv(values: tuple[int | float, ...]) -> str:
    return ",".join(_format_value(value) for value in values)


def _repeated_path_args(flag: str, values: tuple[Path, ...]) -> tuple[str, ...]:
    arguments: list[str] = []
    for path in values:
        arguments.extend((flag, str(path)))
    return tuple(arguments)


def _render_shell_script(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by tools/run_nonlinear_production_campaign.py.",
        "# Prefer the Python launcher with --execute for portable timeout enforcement.",
        "run_step() {",
        "  local timeout_seconds=\"$1\"",
        "  shift",
        "  if command -v timeout >/dev/null 2>&1; then",
        "    timeout \"${timeout_seconds}\" \"$@\"",
        "  else",
        "    \"$@\"",
        "  fi",
        "}",
        "",
    ]
    for command in manifest["commands"]:
        lines.extend(
            [
                f"echo '==> {command['id']}'",
                f"run_step {command['timeout_seconds']} {shlex.join(command['command'])}",
                "",
            ]
        )
    return "\n".join(lines)


def _write_manifest_if_requested(manifest: dict[str, Any], manifest_path: Path | None) -> None:
    if manifest_path is None:
        return
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write or execute a dry-run-first nonlinear production-campaign command manifest."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Write commands only; default.")
    parser.add_argument("--execute", action="store_true", help="Run generated commands.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--harris-growth-rate", type=float, default=DEFAULT_HARRIS_GROWTH_RATE)
    parser.add_argument("--production-efolds", type=float, default=DEFAULT_PRODUCTION_EFOLDS)
    parser.add_argument("--safety-factor", type=float, default=DEFAULT_SAFETY_FACTOR)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--dt", type=float, default=2.0e-2)
    parser.add_argument("--target-saved-frames", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--save-interval", type=float, default=None)
    parser.add_argument("--fit-stop", type=float, default=None)
    parser.add_argument("--width", type=float, default=0.36)
    parser.add_argument("--widths", default="0.32,0.36,0.40")
    parser.add_argument("--eta", type=float, default=4.5e-3)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--etas", default="0.006,0.0045,0.003")
    parser.add_argument("--viscosities", default=None)
    parser.add_argument("--perturbation-amplitude", type=float, default=4.0e-3)
    parser.add_argument("--mode-x", type=int, default=2)
    parser.add_argument("--mode-y", type=int, default=1)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--convergence-resolutions", default="128,192,256")
    parser.add_argument("--crosscheck-resolutions", default="160,224,256")
    parser.add_argument("--dt-values", default="0.02,0.01")
    parser.add_argument("--crosscheck-dt-values", default="0.02,0.01")
    parser.add_argument("--max-walltime-hours", type=float, default=12.0)
    parser.add_argument("--seconds-per-step-estimate", type=float, default=0.5)
    parser.add_argument("--checkpoint-interval-minutes", type=float, default=30.0)
    parser.add_argument("--preemption-margin-minutes", type=float, default=20.0)
    parser.add_argument("--min-production-resolution", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--gate-timeout-seconds", type=int, default=1_800)
    parser.add_argument("--max-relative-energy-growth", type=float, default=1.0e-6)
    parser.add_argument("--max-relative-energy-increase", type=float, default=1.0e-8)
    parser.add_argument("--max-divergence-linf", type=float, default=1.0e-8)
    parser.add_argument("--min-history-samples", type=int, default=100)
    parser.add_argument("--min-reconnected-flux-amplification", type=float, default=1.05)
    parser.add_argument("--min-island-width-amplification", type=float, default=1.05)
    parser.add_argument("--noise-amplitude", type=float, default=1.0e-6)
    args = parser.parse_args(argv)
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    return args


def _options_from_args(args: argparse.Namespace) -> CampaignOptions:
    viscosities = None if args.viscosities is None else _parse_float_csv(args.viscosities)
    return CampaignOptions(
        outdir=args.outdir,
        python_executable=args.python_executable,
        harris_growth_rate=args.harris_growth_rate,
        production_efolds=args.production_efolds,
        safety_factor=args.safety_factor,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        target_saved_frames=args.target_saved_frames,
        save_every=args.save_every,
        save_interval=args.save_interval,
        fit_stop=args.fit_stop,
        width=args.width,
        widths=_parse_float_csv(args.widths),
        eta=args.eta,
        nu=args.nu,
        etas=_parse_float_csv(args.etas),
        viscosities=viscosities,
        perturbation_amplitude=args.perturbation_amplitude,
        mode_x=args.mode_x,
        mode_y=args.mode_y,
        seeds=_parse_int_csv(args.seeds),
        convergence_resolutions=_parse_int_csv(args.convergence_resolutions),
        crosscheck_resolutions=_parse_int_csv(args.crosscheck_resolutions),
        dt_values=_parse_float_csv(args.dt_values),
        crosscheck_dt_values=_parse_float_csv(args.crosscheck_dt_values),
        max_walltime_hours=args.max_walltime_hours,
        seconds_per_step_estimate=args.seconds_per_step_estimate,
        checkpoint_interval_minutes=args.checkpoint_interval_minutes,
        preemption_margin_minutes=args.preemption_margin_minutes,
        min_production_resolution=args.min_production_resolution,
        timeout_seconds=args.timeout_seconds,
        gate_timeout_seconds=args.gate_timeout_seconds,
        max_relative_energy_growth=args.max_relative_energy_growth,
        max_relative_energy_increase=args.max_relative_energy_increase,
        max_divergence_linf=args.max_divergence_linf,
        min_history_samples=args.min_history_samples,
        min_reconnected_flux_amplification=args.min_reconnected_flux_amplification,
        min_island_width_amplification=args.min_island_width_amplification,
        noise_amplitude=args.noise_amplitude,
    )


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
