"""Run-directory benchmark report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown summaries for a completed run directory."""
    directory = Path(run_dir)
    diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "schema": "mhx.benchmark_report.v1",
        "run_dir": str(directory),
        "diagnostics": diagnostics,
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
    return (
        "# MHX benchmark report\n\n"
        f"Run directory: `{report['run_dir']}`\n\n"
        "| Quantity | Value |\n"
        "| --- | --- |\n"
        f"{table}\n\n"
        "This FAST report verifies plumbing and regression behavior. It is not yet a "
        "validated FKR tearing benchmark.\n"
    )
