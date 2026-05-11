from __future__ import annotations

import json

from typer.testing import CliRunner

from mhx.benchmarks import (
    BENCHMARK_CATALOG_SCHEMA,
    validation_catalog_entries,
    write_benchmark_catalog,
)
from mhx.cli.main import app


def test_validation_catalog_entries_cover_expected_gates() -> None:
    entries = validation_catalog_entries()
    names = {entry.name for entry in entries}
    assert "resistive_decay" in names
    assert "harris_delta_prime" in names
    assert "fkr_growth_rate" in names
    assert "linear_tearing_eigenvalue" in names
    assert "linear_tearing_dispersion" in names
    assert "linear_tearing_layer" in names
    assert "linear_tearing_timedomain" in names
    assert "reduced_mhd_eigenmode" in names
    assert "periodic_current_sheet_eigenvalue" in names
    assert "periodic_current_sheet_timedomain" in names
    assert "periodic_current_sheet_nonlinear_bridge" in names
    assert "nonlinear_energy_budget" in names
    assert "nonlinear_duration_audit" in names
    assert "seed_robust_qi" in names
    assert "duration_policy" in names
    assert "arnoldi" in names
    assert all(entry.command.startswith("mhx benchmark") for entry in entries)
    assert all(entry.expected_outputs for entry in entries)


def test_write_benchmark_catalog_and_cli(tmp_path) -> None:
    json_path, markdown_path = write_benchmark_catalog(tmp_path)
    assert json_path == tmp_path / "benchmark_catalog.json"
    assert markdown_path == tmp_path / "benchmark_catalog.md"
    catalog = json.loads(json_path.read_text())
    assert catalog["schema"] == BENCHMARK_CATALOG_SCHEMA
    assert any(entry["name"] == "linearized_rhs" for entry in catalog["entries"])
    assert "MHX validation benchmark catalog" in markdown_path.read_text()
    assert (tmp_path / "manifest.json").exists()

    outdir = tmp_path / "cli-catalog"
    result = CliRunner().invoke(app, ["benchmark", "catalog", "--outdir", str(outdir)])
    assert result.exit_code == 0, result.stdout
    assert (outdir / "benchmark_catalog.json").exists()
    assert (outdir / "benchmark_catalog.md").exists()
