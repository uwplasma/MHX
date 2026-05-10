"""Production-campaign templates with explicit duration guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mhx.benchmarks.duration_policy import (
    DEFAULT_PRODUCTION_EFOLDS,
    HARRIS_REFERENCE_GROWTH_RATE,
    DurationAssessment,
    require_duration_for_claim,
    required_time_for_efolds,
)
from mhx.config import (
    DiagnosticsConfig,
    MeshConfig,
    NumericsConfig,
    PhysicsConfig,
    RunConfig,
    TimeConfig,
)
from mhx.io import write_manifest

RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA = "mhx.campaign.rutherford_template.v1"


@dataclass(frozen=True)
class RutherfordCampaignTemplate:
    """Duration-guarded nonlinear island campaign configuration bundle."""

    config: RunConfig
    duration_assessment: DurationAssessment
    estimated_steps: int
    estimated_saved_frames: int
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def build_rutherford_campaign_template(
    *,
    harris_growth_rate: float = HARRIS_REFERENCE_GROWTH_RATE,
    production_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
    safety_factor: float = 3.0,
    shape: tuple[int, int] = (128, 128),
    dt: float = 0.1,
    target_saved_frames: int = 400,
    run_output_dir: str | Path = "outputs/production/rutherford_island",
) -> RutherfordCampaignTemplate:
    """Build a long nonlinear-island campaign template without running it."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if target_saved_frames < 10:
        raise ValueError("target_saved_frames must be >= 10")
    t_end = required_time_for_efolds(
        harris_growth_rate,
        required_efolds=production_efolds,
        safety_factor=safety_factor,
    )
    duration_assessment = require_duration_for_claim(
        name="rutherford_campaign_template",
        purpose="nonlinear Rutherford island-width growth campaign",
        t_end=t_end,
        growth_rate=harris_growth_rate,
        required_efolds=production_efolds,
        safety_factor=safety_factor,
    )
    estimated_steps = max(1, round(t_end / dt))
    save_every = max(1, round(estimated_steps / target_saved_frames))
    estimated_saved_frames = max(1, estimated_steps // save_every)
    linear_fit_end = required_time_for_efolds(
        harris_growth_rate,
        required_efolds=production_efolds,
    )
    config = RunConfig(
        name="rutherford_island_production_template",
        output_dir=Path(run_output_dir),
        mesh=MeshConfig(
            shape=shape,
            lower=(0.0, 0.0),
            upper=(6.283185307179586, 6.283185307179586),
        ),
        time=TimeConfig(t0=0.0, t1=t_end, dt=dt, save_every=save_every),
        physics=PhysicsConfig(
            model="reduced_mhd_nonlinear_tearing_campaign",
            equilibrium="cosine_tearing",
            equilibrium_parameters={"perturbation_amplitude": 1.0e-3},
            resistivity=1.0e-3,
            viscosity=1.0e-3,
        ),
        numerics=NumericsConfig(method="spectral", enable_x64=True, enable_jit=True),
        diagnostics=DiagnosticsConfig(
            quantities=("energy", "mode_growth", "divergence_error"),
            mode=(1, 1),
            fit_time_window=(0.0, linear_fit_end),
        ),
    )
    checks = {
        "duration_guard_passed": duration_assessment.sufficient_for_nonlinear_claim,
        "not_fast_resolution": min(shape) >= 64,
        "long_time_window": t_end >= linear_fit_end * safety_factor,
        "sufficient_saved_frames": estimated_saved_frames >= 100,
        "fixed_scale_movie_requirements_recorded": True,
        "nonlinear_diagnostics_requirements_recorded": True,
    }
    diagnostics = {
        "schema": RUTHERFORD_CAMPAIGN_TEMPLATE_SCHEMA,
        "config": config.to_dict(),
        "duration_assessment": duration_assessment.to_dict(),
        "estimated_steps": estimated_steps,
        "estimated_saved_frames": estimated_saved_frames,
        "target_saved_frames": target_saved_frames,
        "claim_level": "production_template",
        "claim_boundary": (
            "This artifact is a duration-guarded production template, not a "
            "completed nonlinear reconnection result."
        ),
        "required_runtime_outputs": {
            "histories": (
                "reconnected_flux",
                "rutherford_island_width",
                "current_sheet_aspect_ratio",
                "reconnection_rate_proxy",
                "magnetic_energy",
                "kinetic_energy",
                "dissipation_budget_residual",
            ),
            "figures": (
                "fixed_scale_flux_movie.gif",
                "fixed_scale_current_movie.gif",
                "island_width_history.png",
                "reconnected_flux_history.png",
                "energy_budget_history.png",
            ),
            "convergence": (
                "resolution_sweep",
                "time_step_sweep",
                "fit_window_sensitivity",
            ),
        },
        "references": {
            "linear_tearing": "Furth, Killeen & Rosenbluth, Phys. Fluids 6, 459 (1963).",
            "nonlinear_island": "Rutherford, Phys. Fluids 16, 1903 (1973).",
        },
    }
    validation = {
        "schema": "mhx.campaign.rutherford_template.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": diagnostics,
    }
    return RutherfordCampaignTemplate(
        config=config,
        duration_assessment=duration_assessment,
        estimated_steps=estimated_steps,
        estimated_saved_frames=estimated_saved_frames,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_rutherford_campaign_template(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write a reviewer-facing long nonlinear Rutherford-campaign template."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_rutherford_campaign_template(**kwargs)

    campaign_path = output_dir / "campaign.json"
    validation_path = output_dir / "validation.json"
    config_path = output_dir / "campaign_config.toml"
    assessment_path = output_dir / "duration_assessment.json"
    manifest_path = output_dir / "manifest.json"

    campaign_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config_path.write_text(result.config.to_toml(), encoding="utf-8")
    assessment_path.write_text(
        json.dumps(result.duration_assessment.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "campaign": campaign_path.name,
            "validation": validation_path.name,
            "config": config_path.name,
            "duration_assessment": assessment_path.name,
        },
        claim_level="production_template",
        claim_scope=(
            "Duration-guarded Rutherford island campaign template; not a completed "
            "production simulation."
        ),
    )
    return manifest_path, result.validation
