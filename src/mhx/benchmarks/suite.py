"""Reviewer-facing validation-suite runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mhx.benchmarks.current_sheet import (
    write_periodic_current_sheet_eigenvalue_validation,
)
from mhx.benchmarks.decay import write_resistive_decay_validation
from mhx.benchmarks.eigenvalue import (
    write_arnoldi_validation,
    write_diffusion_eigenvalue_validation,
    write_power_iteration_validation,
)
from mhx.benchmarks.fkr import (
    write_fkr_growth_rate_validation,
    write_fkr_window_validation,
    write_harris_delta_prime_validation,
)
from mhx.benchmarks.linearized import (
    write_cosine_equilibrium_linearization_validation,
    write_linearized_rhs_validation,
    write_reduced_mhd_linear_eigenmode_validation,
)
from mhx.benchmarks.report import validate_run
from mhx.benchmarks.scaling import write_reconnection_scaling_validation
from mhx.benchmarks.tearing import run_linear_tearing_smoke
from mhx.benchmarks.tearing_eigen import (
    write_linear_tearing_dispersion_validation,
    write_linear_tearing_eigenvalue_validation,
    write_linear_tearing_layer_validation,
    write_linear_tearing_timedomain_validation,
)
from mhx.config import load_config
from mhx.io import (
    write_artifact_manifest,
    write_manifest,
    write_reduced_mhd_trajectory_npz,
)
from mhx.runtime import configure_jax
from mhx.versioning import VALIDATION_SUITE_SCHEMA, require_supported_api_version


@dataclass(frozen=True)
class ValidationSuiteCase:
    """One executable validation-suite case."""

    name: str
    command: str
    runner: Callable[[Path], tuple[Path, dict[str, Any]]]


def validation_suite_cases() -> tuple[ValidationSuiteCase, ...]:
    """Return the deterministic validation cases executed by ``mhx validate all``."""
    return (
        ValidationSuiteCase(
            name="linear_tearing_fast",
            command="mhx benchmark run --config examples/linear_tearing.toml",
            runner=_write_linear_tearing_fast_validation,
        ),
        ValidationSuiteCase(
            name="resistive_decay",
            command="mhx benchmark decay",
            runner=write_resistive_decay_validation,
        ),
        ValidationSuiteCase(
            name="reconnection_scaling",
            command="mhx benchmark scaling",
            runner=write_reconnection_scaling_validation,
        ),
        ValidationSuiteCase(
            name="fkr_window",
            command="mhx benchmark fkr-window",
            runner=write_fkr_window_validation,
        ),
        ValidationSuiteCase(
            name="fkr_growth_rate",
            command="mhx benchmark fkr-growth",
            runner=write_fkr_growth_rate_validation,
        ),
        ValidationSuiteCase(
            name="harris_delta_prime",
            command="mhx benchmark harris-delta-prime",
            runner=write_harris_delta_prime_validation,
        ),
        ValidationSuiteCase(
            name="linear_tearing_eigenvalue",
            command="mhx benchmark linear-tearing-eigenvalue",
            runner=write_linear_tearing_eigenvalue_validation,
        ),
        ValidationSuiteCase(
            name="linear_tearing_dispersion",
            command="mhx benchmark linear-tearing-dispersion",
            runner=write_linear_tearing_dispersion_validation,
        ),
        ValidationSuiteCase(
            name="linear_tearing_layer",
            command="mhx benchmark linear-tearing-layer",
            runner=write_linear_tearing_layer_validation,
        ),
        ValidationSuiteCase(
            name="linear_tearing_timedomain",
            command="mhx benchmark linear-tearing-timedomain",
            runner=write_linear_tearing_timedomain_validation,
        ),
        ValidationSuiteCase(
            name="linearized_rhs",
            command="mhx benchmark linearized-rhs",
            runner=write_linearized_rhs_validation,
        ),
        ValidationSuiteCase(
            name="reduced_mhd_eigenmode",
            command="mhx benchmark reduced-mhd-eigenmode",
            runner=write_reduced_mhd_linear_eigenmode_validation,
        ),
        ValidationSuiteCase(
            name="cosine_equilibrium_linearization",
            command="mhx benchmark cosine-equilibrium-linearization",
            runner=write_cosine_equilibrium_linearization_validation,
        ),
        ValidationSuiteCase(
            name="periodic_current_sheet_eigenvalue",
            command="mhx benchmark current-sheet-eigenvalue",
            runner=write_periodic_current_sheet_eigenvalue_validation,
        ),
        ValidationSuiteCase(
            name="diffusion_eigenvalue",
            command="mhx benchmark diffusion-eigenvalue",
            runner=write_diffusion_eigenvalue_validation,
        ),
        ValidationSuiteCase(
            name="power_iteration",
            command="mhx benchmark power-iteration",
            runner=write_power_iteration_validation,
        ),
        ValidationSuiteCase(
            name="arnoldi",
            command="mhx benchmark arnoldi",
            runner=write_arnoldi_validation,
        ),
    )


def write_validation_suite(
    outdir: str | Path,
    *,
    cases: tuple[ValidationSuiteCase, ...] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run all selected FAST validation gates and write a suite summary."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_version = require_supported_api_version(context="validation suite")
    jax_enable_x64 = configure_jax(enable_x64=True)
    selected_cases = cases or validation_suite_cases()
    case_results = []
    for case in selected_cases:
        case_dir = output_dir / case.name
        manifest_path, validation = case.runner(case_dir)
        case_results.append(
            {
                "name": case.name,
                "command": f"{case.command} --outdir {case_dir.as_posix()}",
                "output_dir": case_dir.relative_to(output_dir).as_posix(),
                "manifest": manifest_path.relative_to(output_dir).as_posix(),
                "validation": (case_dir / "validation.json").relative_to(output_dir).as_posix(),
                "schema": validation["schema"],
                "passed": bool(validation["passed"]),
                "checks": validation["checks"],
            }
        )
    summary = {
        "schema": VALIDATION_SUITE_SCHEMA,
        "api_version": api_version,
        "passed": all(item["passed"] for item in case_results),
        "case_count": len(case_results),
        "jax_enable_x64": jax_enable_x64,
        "cases": case_results,
    }
    summary_path = output_dir / "validation_suite.json"
    markdown_path = output_dir / "validation_suite.md"
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    manifest_path = output_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_suite_markdown(summary), encoding="utf-8")
    write_artifact_manifest(output_dir, path=artifact_manifest_path)
    write_manifest(
        manifest_path,
        config={"schema": VALIDATION_SUITE_SCHEMA},
        outputs={
            "summary": summary_path.name,
            "summary_markdown": markdown_path.name,
            "artifact_manifest": artifact_manifest_path.name,
        },
    )
    return summary_path, summary


def _write_linear_tearing_fast_validation(outdir: Path) -> tuple[Path, dict[str, Any]]:
    cfg = load_config("examples/linear_tearing.toml").with_output_dir(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trajectory, diagnostics = run_linear_tearing_smoke(cfg)
    config_path = outdir / "config_effective.json"
    diagnostics_path = outdir / "diagnostics.json"
    trajectory_path = outdir / "trajectory.npz"
    manifest_path = outdir / "manifest.json"
    config_path.write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnostics["grid_shape"] = list(cfg.mesh.shape)
    diagnostics["mesh_lower"] = list(cfg.mesh.lower)
    diagnostics["mesh_upper"] = list(cfg.mesh.upper)
    diagnostics["quantities"] = list(cfg.diagnostics.quantities)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_reduced_mhd_trajectory_npz(
        trajectory_path,
        trajectory=trajectory,
        config=cfg.to_dict(),
        diagnostics=diagnostics,
    )
    write_manifest(
        manifest_path,
        config=cfg.to_dict(),
        outputs={
            "config": config_path.name,
            "diagnostics": diagnostics_path.name,
            "trajectory": trajectory_path.name,
        },
    )
    _, validation = validate_run(outdir)
    return manifest_path, validation


def _suite_markdown(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        "| `{name}` | `{schema}` | `{passed}` | `{manifest}` |".format(**case)
        for case in summary["cases"]
    )
    status = "passed" if summary["passed"] else "failed"
    return (
        "# MHX validation suite\n\n"
        f"Suite status: **{status}**\n\n"
        "| Case | Validation schema | Passed | Manifest |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n"
    )
