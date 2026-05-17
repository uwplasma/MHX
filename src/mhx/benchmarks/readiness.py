"""Reviewer-readiness assessment for validation and publication artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mhx.io import write_manifest

READINESS_REPORT_SCHEMA = "mhx.readiness_report.v1"

REQUIRED_PUBLIC_RELEASE_CASES = (
    "linear_tearing_fast",
    "resistive_decay",
    "reconnection_scaling",
    "fkr_window",
    "fkr_growth_rate",
    "harris_delta_prime",
    "linear_tearing_eigenvalue",
    "linear_tearing_dispersion",
    "linear_tearing_layer",
    "linear_tearing_timedomain",
    "linearized_rhs",
    "reduced_mhd_eigenmode",
    "cosine_equilibrium_linearization",
    "periodic_current_sheet_eigenvalue",
    "periodic_current_sheet_timedomain",
    "periodic_current_sheet_nonlinear_bridge",
    "periodic_double_harris_nonlinear_growth",
    "periodic_double_harris_convergence",
    "nonlinear_energy_budget",
    "orszag_tang_vortex",
    "decaying_mhd_turbulence",
    "forced_turbulent_reconnection",
    "nonlinear_duration_audit",
    "seed_robust_qi",
    "seed_robust_qi_sweep",
    "neural_ode_reproducibility",
    "neural_ode_latent_fit",
    "rutherford_production_execution",
    "duration_policy",
    "diffusion_eigenvalue",
    "power_iteration",
    "arnoldi",
)

PRODUCTION_PUBLICATION_GAPS = (
    "long Rutherford/island campaign with checkpoint/resume and duration gates",
    "long plasmoid-chain campaign with Lundquist/aspect-ratio convergence",
    "medium/production resolution sweeps with documented tolerance budgets",
    "neural-ODE train/validation/test experiment linked to production-quality data",
)


@dataclass(frozen=True)
class ReadinessAssessment:
    """Machine-readable release and paper-readiness assessment."""

    public_release_ready: bool
    publication_claim_ready: bool
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def run_readiness_assessment(
    validation_suite: str | Path | dict[str, Any],
    *,
    required_cases: tuple[str, ...] = REQUIRED_PUBLIC_RELEASE_CASES,
) -> ReadinessAssessment:
    """Assess whether current artifacts support public release and paper claims."""
    suite = _load_validation_suite(validation_suite)
    case_map = {str(case["name"]): case for case in suite.get("cases", [])}
    missing_cases = [name for name in required_cases if name not in case_map]
    failed_cases = [
        name
        for name, case in sorted(case_map.items())
        if name in required_cases and not bool(case.get("passed", False))
    ]
    claim_levels = sorted(
        {str(case.get("claim_level", "unspecified")) for case in case_map.values()}
    )
    validation_suite_passed = bool(suite.get("passed", False))
    public_release_ready = validation_suite_passed and not missing_cases and not failed_cases
    publication_claim_ready = False
    checks = {
        "validation_suite_passed": validation_suite_passed,
        "all_required_public_release_cases_present": not missing_cases,
        "all_required_public_release_cases_passed": not failed_cases,
        "claim_levels_are_explicit": all(
            level in {"smoke", "validation"} for level in claim_levels
        ),
        "production_gaps_are_declared": bool(PRODUCTION_PUBLICATION_GAPS),
        "publication_claims_blocked_until_production_campaigns": not publication_claim_ready,
    }
    diagnostics = {
        "schema": READINESS_REPORT_SCHEMA,
        "suite_schema": suite.get("schema"),
        "suite_case_count": int(suite.get("case_count", len(case_map))),
        "required_public_release_cases": list(required_cases),
        "present_required_cases": [name for name in required_cases if name in case_map],
        "missing_required_cases": missing_cases,
        "failed_required_cases": failed_cases,
        "claim_levels": claim_levels,
        "public_release_ready": public_release_ready,
        "publication_claim_ready": publication_claim_ready,
        "production_publication_gaps": list(PRODUCTION_PUBLICATION_GAPS),
        "interpretation": (
            "Public-release readiness requires all FAST validation gates to be "
            "present and passing. Publication-level nonlinear reconnection claims "
            "remain blocked until long production campaigns and convergence "
            "evidence are attached."
        ),
    }
    validation = {
        "schema": "mhx.readiness_report.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": diagnostics,
    }
    return ReadinessAssessment(
        public_release_ready=public_release_ready,
        publication_claim_ready=publication_claim_ready,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_readiness_report(
    outdir: str | Path,
    validation_suite: str | Path | dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Write readiness JSON, Markdown, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_readiness_assessment(validation_suite)
    diagnostics_path = output_dir / "readiness.json"
    validation_path = output_dir / "validation.json"
    markdown_path = output_dir / "readiness.md"
    figure_path = output_dir / "figures" / "readiness_matrix.png"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(_readiness_markdown(result.diagnostics), encoding="utf-8")
    _write_readiness_figure(result.diagnostics, figure_path)
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "readiness": diagnostics_path.name,
            "validation": validation_path.name,
            "readiness_markdown": markdown_path.name,
            "readiness_matrix": str(figure_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Reviewer-facing readiness assessment of validation artifacts.",
    )
    return diagnostics_path, result.validation


def _load_validation_suite(validation_suite: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(validation_suite, dict):
        return validation_suite
    path = Path(validation_suite)
    if path.is_dir():
        path = path / "validation_suite.json"
    if not path.exists():
        raise FileNotFoundError(f"validation suite not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("validation suite JSON must contain an object")
    return data


def _readiness_markdown(diagnostics: dict[str, Any]) -> str:
    status = "yes" if diagnostics["public_release_ready"] else "no"
    paper_status = "yes" if diagnostics["publication_claim_ready"] else "no"
    missing = diagnostics["missing_required_cases"] or ["none"]
    failed = diagnostics["failed_required_cases"] or ["none"]
    gaps = "\n".join(f"- {item}" for item in diagnostics["production_publication_gaps"])
    return (
        "# MHX readiness report\n\n"
        f"- Public release ready: **{status}**\n"
        f"- Publication nonlinear-claim ready: **{paper_status}**\n"
        f"- Validation-suite schema: `{diagnostics['suite_schema']}`\n"
        f"- Suite case count: `{diagnostics['suite_case_count']}`\n\n"
        "## Missing required cases\n\n"
        + "\n".join(f"- `{item}`" for item in missing)
        + "\n\n## Failed required cases\n\n"
        + "\n".join(f"- `{item}`" for item in failed)
        + "\n\n## Production publication gaps\n\n"
        + gaps
        + "\n"
    )


def _write_readiness_figure(diagnostics: dict[str, Any], path: Path) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ("release gates", "paper nonlinear claims")
    values = np.asarray(
        [
            1.0 if diagnostics["public_release_ready"] else 0.0,
            1.0 if diagnostics["publication_claim_ready"] else 0.0,
        ]
    )
    colors = ["#2ca25f" if value else "#de2d26" for value in values]
    fig, ax = plt.subplots(figsize=(6.0, 2.6), constrained_layout=True)
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("ready gate")
    ax.set_title("MHX reviewer-readiness matrix")
    for row, value in enumerate(values):
        ax.text(0.03, row, "ready" if value else "blocked", va="center", color="white")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
