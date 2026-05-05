"""Run-directory benchmark report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_run(
    run_dir: str | Path,
    *,
    max_relative_energy_growth: float = 1.0e-10,
    require_finite_gamma: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Validate a run directory against lightweight regression checks."""
    directory = Path(run_dir)
    diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    initial_energy = float(diagnostics["initial_total_energy"])
    final_energy = float(diagnostics["final_total_energy"])
    relative_growth = (final_energy - initial_energy) / max(abs(initial_energy), 1.0e-300)
    checks = {
        "finite_positive_initial_energy": initial_energy > 0.0,
        "finite_positive_final_energy": final_energy > 0.0,
        "energy_growth_within_tolerance": relative_growth <= max_relative_energy_growth,
        "finite_gamma_fit": (
            _is_finite_number(diagnostics["gamma_fit"]) if require_finite_gamma else True
        ),
    }
    result = {
        "schema": "mhx.validation.v1",
        "run_dir": str(directory),
        "passed": all(checks.values()),
        "checks": checks,
        "relative_energy_growth": relative_growth,
        "max_relative_energy_growth": max_relative_energy_growth,
    }
    output_path = directory / "validation.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return output_path, result


def write_run_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown summaries for a completed run directory."""
    directory = Path(run_dir)
    diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    additional_scalar_diagnostics = _additional_scalar_diagnostics(diagnostics)
    report = {
        "schema": "mhx.benchmark_report.v1",
        "run_dir": str(directory),
        "diagnostics": diagnostics,
        "additional_scalar_diagnostics": additional_scalar_diagnostics,
        "manifest_schema": manifest["schema"],
        "manifest_hashes": manifest["hashes"],
    }
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _report_markdown(report: dict[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    rows = [
        ("final time", diagnostics["final_time"]),
        ("initial total energy", diagnostics["initial_total_energy"]),
        ("final total energy", diagnostics["final_total_energy"]),
        ("diagnostic mode", diagnostics["diagnostic_mode"]),
        ("initial mode amplitude", diagnostics["initial_mode_amplitude"]),
        ("final mode amplitude", diagnostics["final_mode_amplitude"]),
        ("gamma fit", diagnostics["gamma_fit"]),
    ]
    table = "\n".join(f"| {name} | `{value}` |" for name, value in rows)
    additional_scalars = report["additional_scalar_diagnostics"]
    additional_section = ""
    if additional_scalars:
        additional_rows = "\n".join(
            f"| `{name}` | `{value}` |" for name, value in sorted(additional_scalars.items())
        )
        additional_section = (
            "\n## Additional scalar diagnostics\n\n"
            "| Key | Value |\n"
            "| --- | --- |\n"
            f"{additional_rows}\n"
        )
    return (
        "# MHX benchmark report\n\n"
        f"Run directory: `{report['run_dir']}`\n\n"
        "| Quantity | Value |\n"
        "| --- | --- |\n"
        f"{table}\n\n"
        f"{additional_section}\n"
        "This FAST report verifies plumbing and regression behavior. It is not yet a "
        "validated FKR tearing benchmark.\n"
    )


def _additional_scalar_diagnostics(
    diagnostics: dict[str, Any],
) -> dict[str, int | float | str | bool]:
    core_keys = {
        "diagnostic_mode",
        "diagnostic_plugin_entry_point_groups",
        "diagnostic_plugin_modules",
        "diagnostic_quantities",
        "equilibrium",
        "equilibrium_parameters",
        "final_kinetic_energy",
        "final_magnetic_divergence_linf",
        "final_magnetic_energy",
        "final_mode_amplitude",
        "final_time",
        "final_total_energy",
        "fit_sample_count",
        "fit_time_window",
        "gamma_fit",
        "initial_mode_amplitude",
        "initial_total_energy",
        "n_steps",
        "physics_plugin_entry_point_groups",
        "physics_plugin_modules",
        "physics_terms",
    }
    return {
        key: value
        for key, value in diagnostics.items()
        if key not in core_keys and isinstance(value, int | float | str | bool)
    }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and value == value and abs(value) != float("inf")
