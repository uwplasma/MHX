"""Simulation-duration policy helpers for literature-aligned reconnection claims."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mhx.benchmarks.theory import loureiro_plasmoid_estimate
from mhx.io import write_manifest

DURATION_POLICY_SCHEMA = "mhx.duration_policy.v1"
HARRIS_REFERENCE_GROWTH_RATE = 1.31e-2
DEFAULT_PRODUCTION_EFOLDS = 10.0

_VALIDATION_ONLY_SCOPES = frozenset(
    {
        "smoke",
        "linear_operator_replay",
        "nonlinear_identity_gate",
        "differentiability_gate",
    }
)


@dataclass(frozen=True)
class DurationAssessment:
    """Assessment of whether a run is long enough for its declared purpose."""

    name: str
    purpose: str
    scope: str
    t_end: float
    growth_rate: float
    required_efolds: float
    safety_factor: float
    required_t_end: float
    observed_efolds: float
    sufficient_for_intended_scope: bool
    sufficient_for_production_claim: bool
    sufficient_for_nonlinear_claim: bool
    action: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible values."""
        return asdict(self)


def required_time_for_efolds(
    growth_rate: float,
    *,
    required_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
    safety_factor: float = 1.0,
) -> float:
    """Return ``safety_factor * required_efolds / growth_rate``."""
    _validate_duration_policy_inputs(
        t_end=1.0,
        growth_rate=growth_rate,
        required_efolds=required_efolds,
        safety_factor=safety_factor,
    )
    return safety_factor * required_efolds / growth_rate


def assess_duration(
    *,
    name: str,
    purpose: str,
    t_end: float,
    growth_rate: float = HARRIS_REFERENCE_GROWTH_RATE,
    required_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
    safety_factor: float = 1.0,
    scope: str = "nonlinear_production",
) -> DurationAssessment:
    """Assess a simulation duration against an e-fold requirement.

    Validation-only scopes are allowed to be shorter than production nonlinear
    targets, but the returned assessment explicitly marks them as not sufficient
    for nonlinear-island or plasmoid claims.
    """
    if not name:
        raise ValueError("name must be non-empty")
    if not purpose:
        raise ValueError("purpose must be non-empty")
    if not scope:
        raise ValueError("scope must be non-empty")
    _validate_duration_policy_inputs(
        t_end=t_end,
        growth_rate=growth_rate,
        required_efolds=required_efolds,
        safety_factor=safety_factor,
    )
    required_t_end = required_time_for_efolds(
        growth_rate,
        required_efolds=required_efolds,
        safety_factor=safety_factor,
    )
    observed_efolds = t_end * growth_rate
    sufficient_for_claim = t_end >= required_t_end
    if scope in _VALIDATION_ONLY_SCOPES:
        sufficient_for_intended_scope = True
        action = (
            "allowed for validation/smoke only; do not use this run as a "
            "nonlinear reconnection claim"
            if not sufficient_for_claim
            else "long enough for the e-fold target, but scope remains validation-only"
        )
    else:
        sufficient_for_intended_scope = sufficient_for_claim
        action = (
            "long enough for declared production e-fold target"
            if sufficient_for_claim
            else f"increase t_end to at least {required_t_end:.6g}"
        )
    return DurationAssessment(
        name=name,
        purpose=purpose,
        scope=scope,
        t_end=t_end,
        growth_rate=growth_rate,
        required_efolds=required_efolds,
        safety_factor=safety_factor,
        required_t_end=required_t_end,
        observed_efolds=observed_efolds,
        sufficient_for_intended_scope=sufficient_for_intended_scope,
        sufficient_for_production_claim=(
            sufficient_for_claim and scope not in _VALIDATION_ONLY_SCOPES
        ),
        sufficient_for_nonlinear_claim=(sufficient_for_claim and scope == "nonlinear_production"),
        action=action,
    )


def require_duration_for_claim(
    *,
    name: str,
    purpose: str,
    t_end: float,
    growth_rate: float = HARRIS_REFERENCE_GROWTH_RATE,
    required_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
    safety_factor: float = 1.0,
) -> DurationAssessment:
    """Return an assessment or raise if a production claim would be too short."""
    assessment = assess_duration(
        name=name,
        purpose=purpose,
        t_end=t_end,
        growth_rate=growth_rate,
        required_efolds=required_efolds,
        safety_factor=safety_factor,
        scope="nonlinear_production",
    )
    if not assessment.sufficient_for_nonlinear_claim:
        raise ValueError(
            f"{name} is too short for a nonlinear production claim: "
            f"t_end={t_end:g}, required_t_end={assessment.required_t_end:g}"
        )
    return assessment


def duration_policy_assessments(
    *,
    harris_growth_rate: float = HARRIS_REFERENCE_GROWTH_RATE,
    production_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
) -> tuple[DurationAssessment, ...]:
    """Return current and future duration assessments used by docs and CI."""
    harris_required = required_time_for_efolds(
        harris_growth_rate,
        required_efolds=production_efolds,
    )
    plasmoid_s = 1.0e6
    plasmoid_growth = loureiro_plasmoid_estimate(plasmoid_s).gamma_tau_a
    plasmoid_required = required_time_for_efolds(
        plasmoid_growth,
        required_efolds=production_efolds,
    )
    return (
        assess_duration(
            name="linear_tearing_fast",
            purpose="configuration, IO, plotting, and smoke integration",
            t_end=0.10,
            growth_rate=harris_growth_rate,
            required_efolds=production_efolds,
            scope="smoke",
        ),
        assess_duration(
            name="linear_tearing_timedomain",
            purpose="linear operator replay and growth-fit plumbing",
            t_end=80.0,
            growth_rate=harris_growth_rate,
            required_efolds=1.0,
            scope="linear_operator_replay",
        ),
        assess_duration(
            name="nonlinear_energy_budget",
            purpose="nonlinear bracket cancellation and dissipation identity",
            t_end=0.80,
            growth_rate=harris_growth_rate,
            required_efolds=production_efolds,
            scope="nonlinear_identity_gate",
        ),
        assess_duration(
            name="future_harris_linear_growth_campaign",
            purpose="production linear-growth observation with ten e-folds",
            t_end=harris_required,
            growth_rate=harris_growth_rate,
            required_efolds=production_efolds,
            scope="linear_production",
        ),
        assess_duration(
            name="future_rutherford_island_campaign",
            purpose="nonlinear island tracking after a resolved linear phase",
            t_end=3.0 * harris_required,
            growth_rate=harris_growth_rate,
            required_efolds=production_efolds,
            safety_factor=3.0,
            scope="nonlinear_production",
        ),
        assess_duration(
            name="future_plasmoid_linear_onset_campaign",
            purpose="Sweet-Parker plasmoid onset at S=1e6, time window only",
            t_end=plasmoid_required,
            growth_rate=plasmoid_growth,
            required_efolds=production_efolds,
            scope="nonlinear_production",
        ),
    )


def write_duration_policy(
    outdir: str | Path,
    *,
    harris_growth_rate: float = HARRIS_REFERENCE_GROWTH_RATE,
    production_efolds: float = DEFAULT_PRODUCTION_EFOLDS,
) -> tuple[Path, dict[str, Any]]:
    """Write duration-policy JSON, Markdown, validation, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assessments = duration_policy_assessments(
        harris_growth_rate=harris_growth_rate,
        production_efolds=production_efolds,
    )
    policy = {
        "schema": DURATION_POLICY_SCHEMA,
        "harris_growth_rate": harris_growth_rate,
        "production_efolds": production_efolds,
        "assessments": [assessment.to_dict() for assessment in assessments],
        "policy": {
            "validation_only_scopes": sorted(_VALIDATION_ONLY_SCOPES),
            "production_rule": "t_end >= safety_factor * required_efolds / growth_rate",
            "claim_boundary": (
                "Validation-only runs may pass their engineering gates while "
                "remaining explicitly unusable as nonlinear reconnection claims."
            ),
        },
    }
    checks = {
        "all_intended_scopes_have_valid_duration": all(
            assessment.sufficient_for_intended_scope for assessment in assessments
        ),
        "current_short_runs_are_validation_only": all(
            assessment.scope in _VALIDATION_ONLY_SCOPES
            for assessment in assessments
            if (
                not assessment.name.startswith("future_")
                and not assessment.sufficient_for_production_claim
            )
        ),
        "future_production_templates_are_long_enough": all(
            assessment.sufficient_for_production_claim
            for assessment in assessments
            if assessment.name.startswith("future_")
        ),
        "future_nonlinear_templates_are_long_enough": all(
            assessment.sufficient_for_nonlinear_claim
            for assessment in assessments
            if assessment.name.startswith("future_") and assessment.scope == "nonlinear_production"
        ),
    }
    validation = {
        "schema": "mhx.duration_policy.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": policy,
    }
    policy_path = output_dir / "duration_policy.json"
    markdown_path = output_dir / "duration_policy.md"
    validation_path = output_dir / "validation.json"
    manifest_path = output_dir / "manifest.json"
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_duration_policy_markdown(policy), encoding="utf-8")
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_manifest(
        manifest_path,
        config=policy,
        outputs={
            "duration_policy": policy_path.name,
            "duration_policy_markdown": markdown_path.name,
            "validation": validation_path.name,
        },
    )
    return manifest_path, validation


def _duration_policy_markdown(policy: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {name} | `{scope}` | {t_end:.6g} | {observed_efolds:.3g} | "
        "{required_t_end:.6g} | {intended} | {production} | {nonlinear} | {action} |".format(
            name=item["name"],
            scope=item["scope"],
            t_end=item["t_end"],
            observed_efolds=item["observed_efolds"],
            required_t_end=item["required_t_end"],
            intended="yes" if item["sufficient_for_intended_scope"] else "no",
            production="yes" if item["sufficient_for_production_claim"] else "no",
            nonlinear="yes" if item["sufficient_for_nonlinear_claim"] else "no",
            action=item["action"],
        )
        for item in policy["assessments"]
    )
    return (
        "# MHX simulation-duration policy\n\n"
        "Production reconnection claims must satisfy "
        "`t_end >= safety_factor * required_efolds / growth_rate`. "
        "Short FAST runs are allowed only when their scope is explicitly "
        "validation-only.\n\n"
        "| Run | Scope | t_end | observed e-folds | required t_end | intended ok | "
        "production ok | nonlinear ok | action |\n"
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |\n"
        f"{rows}\n"
    )


def _validate_duration_policy_inputs(
    *,
    t_end: float,
    growth_rate: float,
    required_efolds: float,
    safety_factor: float,
) -> None:
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if growth_rate <= 0.0:
        raise ValueError("growth_rate must be positive")
    if required_efolds <= 0.0:
        raise ValueError("required_efolds must be positive")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive")
