#!/usr/bin/env python3
"""Write nonlinear-campaign evidence claim tables from local artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "mhx.docs.nonlinear_campaign_evidence.v1"


@dataclass(frozen=True)
class EvidenceCase:
    label: str
    family: str
    run_dir: Path
    command: str
    metric_keys: tuple[str, ...]
    blocker: str
    promotion_path: Path | None = None


CASES = (
    EvidenceCase(
        label="2026-05-22 bounded double-Harris convergence",
        family="periodic double-Harris",
        run_dir=Path(
            "outputs/nonlinear_campaign_evidence_20260522/double_harris_convergence_n16_24_t8"
        ),
        command=(
            "python -m mhx.cli.main benchmark double-harris-convergence "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/"
            "double_harris_convergence_n16_24_t8 "
            "--resolutions 16,24 --dt-values 0.02,0.01 --reference-resolution 16 "
            "--reference-dt 0.01 --t-end 8 --save-interval 1 --fit-stop 4"
        ),
        metric_keys=(
            "case_count",
            "t_end",
            "resolution_growth_rate_spread",
            "timestep_growth_rate_spread",
            "resolution_max_growth_spread",
            "timestep_max_growth_spread",
        ),
        blocker=(
            "Small 16/24 grid and short t=8 validation sweep; production needs larger "
            "resolution, duration, seed, width/aspect, and Lundquist sweeps."
        ),
    ),
    EvidenceCase(
        label="2026-05-22 bounded double-Harris long replay",
        family="periodic double-Harris",
        run_dir=Path("outputs/nonlinear_campaign_evidence_20260522/double_harris_long_n48_t24"),
        command=(
            "python -m mhx.cli.main benchmark double-harris-long-run "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/double_harris_long_n48_t24 "
            "--nx 48 --ny 48 --t-end 24 --dt 0.02 --save-every 100 "
            "--fit-stop 8 --min-max-growth-factor 2 --no-movies"
        ),
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "fitted_early_growth_rate",
            "max_growth_factor",
            "reconnected_flux_amplification",
            "island_width_amplification",
            "relative_energy_increase",
        ),
        blocker=(
            "Positive response is validation evidence only; no production-duration "
            "convergence, seed-QI, aspect-ratio, or Lundquist sweep is attached."
        ),
    ),
    EvidenceCase(
        label="2026-05-22 bounded Rutherford FAST run",
        family="Rutherford executor",
        run_dir=Path("outputs/nonlinear_campaign_evidence_20260522/rutherford_fast_n24_t1"),
        command=(
            "python -m mhx.cli.main campaign rutherford-run-fast "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/rutherford_fast_n24_t1 "
            "--nx 24 --ny 24 --t-end 1.0 --dt 0.01 --save-every 5"
        ),
        metric_keys=(
            "shape",
            "t_end",
            "steps",
            "samples",
            "max_relative_energy_growth",
            "max_magnetic_divergence_linf",
        ),
        blocker=(
            "FAST schema/diagnostic run is far shorter than Rutherford-duration requirements "
            "and cannot be promoted to nonlinear physics."
        ),
    ),
    EvidenceCase(
        label="2026-05-22 bounded forced turbulent reconnection",
        family="forced turbulent reconnection",
        run_dir=Path(
            "outputs/nonlinear_campaign_evidence_20260522/forced_turbulent_reconnection_n24_t4"
        ),
        command=(
            "python -m mhx.cli.main benchmark forced-turbulent-reconnection "
            "--outdir outputs/nonlinear_campaign_evidence_20260522/"
            "forced_turbulent_reconnection_n24_t4 "
            "--nx 24 --ny 24 --t-end 4 --dt 0.02 --save-every 10 --no-movies"
        ),
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "reconnection_proxy_change",
            "max_abs_reconnection_rate_proxy",
            "current_linf_growth",
            "max_relative_energy_growth",
            "max_magnetic_divergence_linf",
        ),
        blocker=(
            "2-D reduced-MHD proxy and single deterministic seed; no turbulent ensemble, "
            "3-D physics, inertial range, or LV99 scaling evidence."
        ),
        promotion_path=Path("readiness/promotion_readiness.json"),
    ),
    EvidenceCase(
        label="Archived GPU-assisted double-Harris convergence",
        family="periodic double-Harris",
        run_dir=Path(
            "outputs/campaigns/double_harris_convergence_gpu_n32_48_64_t16_20260519_173637"
        ),
        command="Archived command recorded in docs/long_run_evidence.md.",
        metric_keys=(
            "case_count",
            "t_end",
            "resolution_growth_rate_spread",
            "timestep_growth_rate_spread",
            "resolution_max_growth_spread",
            "timestep_max_growth_spread",
        ),
        blocker=(
            "Medium validation sweep is not a production-scale duration, seed, "
            "aspect-ratio, or Lundquist campaign."
        ),
    ),
    EvidenceCase(
        label="Archived GPU-assisted double-Harris response",
        family="periodic double-Harris",
        run_dir=Path("outputs/campaigns/growing_double_harris_gpu_96_t120_20260518_044120"),
        command="Archived command recorded in docs/long_run_evidence.md.",
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "fitted_early_growth_rate",
            "max_growth_factor",
            "reconnected_flux_amplification",
            "island_width_amplification",
            "max_x_point_count",
            "max_o_point_count",
            "relative_energy_increase",
        ),
        blocker=(
            "Convergence-backed validation media only; production claims still need larger "
            "seed, width/aspect, Lundquist, and duration sweeps."
        ),
        promotion_path=Path("promotion/promotion_readiness.json"),
    ),
    EvidenceCase(
        label="Archived current-schema Rutherford duration run",
        family="Rutherford executor",
        run_dir=Path("outputs/campaigns/rutherford_current_schema_96_dt005_20260517_161235"),
        command="Archived command recorded in docs/long_run_evidence.md.",
        metric_keys=(
            "shape",
            "end_step",
            "target_step",
            "history_samples",
            "max_relative_energy_growth",
            "final_magnetic_divergence_linf",
        ),
        blocker=(
            "Duration target completed, but promotion failed because reconnecting-flux "
            "and island-width amplification remained 1.00."
        ),
        promotion_path=Path("promotion/promotion_readiness.json"),
    ),
    EvidenceCase(
        label="Archived forced turbulent reconnection README media",
        family="forced turbulent reconnection",
        run_dir=Path("outputs/readme_media/forced_turbulent_reconnection_64_t80_wide"),
        command="Archived command recorded in docs/readme media QA artifacts.",
        metric_keys=(
            "shape",
            "t_end",
            "samples",
            "reconnection_proxy_change",
            "max_abs_reconnection_rate_proxy",
            "current_linf_growth",
            "max_relative_energy_growth",
            "max_magnetic_divergence_linf",
        ),
        blocker=(
            "Validation media only: single deterministic 2-D proxy run, no ensemble or "
            "3-D turbulent-reconnection scaling."
        ),
    ),
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compact_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [compact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_value(item) for key, item in value.items()}
    return value


def collect_case(evidence_case: EvidenceCase) -> dict[str, Any]:
    manifest = read_json(evidence_case.run_dir / "manifest.json")
    diagnostics = read_json(evidence_case.run_dir / "diagnostics.json")
    validation = read_json(evidence_case.run_dir / "validation.json")
    promotion = (
        read_json(evidence_case.run_dir / evidence_case.promotion_path)
        if evidence_case.promotion_path is not None
        else {}
    )
    metrics = {
        key: compact_value(diagnostics[key])
        for key in evidence_case.metric_keys
        if key in diagnostics
    }
    for key in (
        "promotion_ready",
        "reconnected_flux_amplification",
        "island_width_amplification",
        "max_x_point_count",
        "max_o_point_count",
        "terminal_step",
        "target_step",
        "history_sample_count",
    ):
        if key in promotion and key not in metrics:
            metrics[key] = compact_value(promotion[key])
    return {
        "label": evidence_case.label,
        "family": evidence_case.family,
        "run_dir": str(evidence_case.run_dir),
        "present": (evidence_case.run_dir / "manifest.json").exists(),
        "command": evidence_case.command,
        "claim_level": manifest.get("claim_level", "missing"),
        "claim_scope": manifest.get("claim_scope", ""),
        "validation_passed": validation.get("passed"),
        "promotion_ready": promotion.get("promotion_ready", False),
        "production_claim_ready": False,
        "metrics": metrics,
        "checks": validation.get("checks", {}),
        "promotion_checks": promotion.get("checks", {}),
        "blocker": evidence_case.blocker,
        "artifacts": {
            "manifest": str(evidence_case.run_dir / "manifest.json"),
            "diagnostics": str(evidence_case.run_dir / "diagnostics.json"),
            "validation": str(evidence_case.run_dir / "validation.json"),
            "promotion_readiness": (
                str(evidence_case.run_dir / evidence_case.promotion_path)
                if evidence_case.promotion_path is not None
                else None
            ),
        },
    }


def format_metric_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "missing"
    parts = []
    for key, value in metrics.items():
        parts.append(f"`{key}`={value}")
    return ", ".join(parts)


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Nonlinear campaign evidence claims",
        "",
        "This generated table records the local nonlinear campaign evidence inspected for the",
        "double-Harris, Rutherford, and forced turbulent-reconnection lanes. It is deliberately",
        "conservative: every row remains below production physics claim level.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python tools/nonlinear_campaign_evidence.py \\",
        "  --output-json docs/nonlinear_campaign_evidence.json \\",
        "  --output-md docs/nonlinear_campaign_evidence.md",
        "```",
        "",
        "## Claim table",
        "",
        "| Lane | Artifact | Validation | Readiness gate | Key metrics | Production blocker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["cases"]:
        validation = "pass" if item["validation_passed"] else "fail/missing"
        promotion = "validation-ready" if item["promotion_ready"] else "not production-ready"
        metrics = format_metric_summary(item["metrics"])
        lines.append(
            (
                "| {family} | `{run_dir}` | {validation} | {promotion} | {metrics} | {blocker} |"
            ).format(
                family=item["family"],
                run_dir=item["run_dir"],
                validation=validation,
                promotion=promotion,
                metrics=metrics,
                blocker=item["blocker"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Passing double-Harris rows support validation-level nonlinear response and "
            "convergence scaffolding, not Rutherford/plasmoid production physics.",
            "- Passing Rutherford rows support executor/schema/duration mechanics unless "
            "the promotion report passes with positive response, convergence, seed-QI, "
            "geometry, and media gates.",
            "- Passing forced turbulent-reconnection rows support 2-D reduced-MHD proxy-media "
            "readiness only, not 3-D turbulent-reconnection or LV99 scaling claims.",
            "- Large binary outputs remain under `outputs/`, which is git-ignored; this page "
            "and the JSON summary are the small review artifacts.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    cases = [collect_case(evidence_case) for evidence_case in CASES]
    return {
        "schema": SCHEMA,
        "refresh_date": "2026-05-22",
        "claim_boundary": (
            "All listed nonlinear campaign artifacts are validation evidence; none support "
            "production Rutherford, plasmoid, Sweet-Parker, or 3-D turbulent-reconnection "
            "claims."
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="docs/nonlinear_campaign_evidence.json")
    parser.add_argument("--output-md", default="docs/nonlinear_campaign_evidence.md")
    args = parser.parse_args()

    report = build_report()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, Path(args.output_md))
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
