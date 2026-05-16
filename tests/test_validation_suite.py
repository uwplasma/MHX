from __future__ import annotations

import json

from typer.testing import CliRunner

from mhx.benchmarks import (
    VALIDATION_SUITE_SCHEMA,
    validation_suite_cases,
    write_validation_suite,
)
from mhx.cli.main import app


def test_validation_suite_cases_are_unique() -> None:
    names = [case.name for case in validation_suite_cases()]
    assert len(names) == len(set(names))
    assert "linear_tearing_fast" in names
    assert "harris_delta_prime" in names
    assert "fkr_growth_rate" in names
    assert "linear_tearing_eigenvalue" in names
    assert "linear_tearing_dispersion" in names
    assert "linear_tearing_layer" in names
    assert "linear_tearing_timedomain" in names
    assert "cosine_equilibrium_linearization" in names
    assert "periodic_current_sheet_eigenvalue" in names
    assert "periodic_current_sheet_timedomain" in names
    assert "periodic_current_sheet_nonlinear_bridge" in names
    assert "periodic_double_harris_nonlinear_growth" in names
    assert "periodic_double_harris_convergence" in names
    assert "nonlinear_energy_budget" in names
    assert "orszag_tang_vortex" in names
    assert "nonlinear_duration_audit" in names
    assert "seed_robust_qi" in names
    assert "seed_robust_qi_sweep" in names
    assert "neural_ode_reproducibility" in names
    assert "duration_policy" in names


def test_write_validation_suite_artifacts_and_cli(tmp_path) -> None:
    summary_path, summary = write_validation_suite(tmp_path / "suite")
    assert summary_path == tmp_path / "suite" / "validation_suite.json"
    assert summary["schema"] == VALIDATION_SUITE_SCHEMA
    assert summary["passed"] is True
    assert summary["jax_enable_x64"] is True
    assert summary["case_count"] == len(validation_suite_cases())
    assert (tmp_path / "suite" / "validation_suite.md").stat().st_size > 0
    assert (tmp_path / "suite" / "artifact_manifest.json").exists()
    assert (tmp_path / "suite" / "manifest.json").exists()
    assert (
        tmp_path
        / "suite"
        / "harris_delta_prime"
        / "figures"
        / "harris_delta_prime.png"
    ).stat().st_size > 0
    assert (
        tmp_path / "suite" / "fkr_growth_rate" / "figures" / "fkr_growth_rate.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "linear_tearing_eigenvalue"
        / "figures"
        / "linear_tearing_eigenvalue.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "linear_tearing_dispersion"
        / "figures"
        / "linear_tearing_dispersion.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "linear_tearing_layer"
        / "figures"
        / "linear_tearing_layer.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "linear_tearing_timedomain"
        / "figures"
        / "linear_tearing_timedomain.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "cosine_equilibrium_linearization"
        / "figures"
        / "cosine_equilibrium_linearization_errors.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "periodic_current_sheet_eigenvalue"
        / "figures"
        / "periodic_current_sheet_spectrum.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "periodic_current_sheet_timedomain"
        / "figures"
        / "periodic_current_sheet_timedomain.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "periodic_current_sheet_nonlinear_bridge"
        / "figures"
        / "periodic_current_sheet_nonlinear_bridge.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "periodic_double_harris_nonlinear_growth"
        / "figures"
        / "periodic_double_harris_nonlinear_growth.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "periodic_double_harris_convergence"
        / "figures"
        / "periodic_double_harris_convergence.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "nonlinear_energy_budget"
        / "figures"
        / "nonlinear_energy_budget.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "orszag_tang_vortex"
        / "figures"
        / "orszag_tang_summary.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "nonlinear_duration_audit"
        / "figures"
        / "nonlinear_duration_audit.png"
    ).stat().st_size > 0
    assert (
        tmp_path / "suite" / "seed_robust_qi" / "figures" / "qi_summary.png"
    ).stat().st_size > 0
    assert (
        tmp_path / "suite" / "seed_robust_qi_sweep" / "figures" / "qi_sweep_cv.png"
    ).stat().st_size > 0
    assert (
        tmp_path
        / "suite"
        / "neural_ode_reproducibility"
        / "figures"
        / "baseline_rmse.png"
    ).stat().st_size > 0
    assert (
        tmp_path / "suite" / "neural_ode_reproducibility" / "validation.json"
    ).exists()
    assert (tmp_path / "suite" / "duration_policy" / "duration_policy.json").exists()
    persisted = json.loads((tmp_path / "suite" / "validation_suite.json").read_text())
    assert persisted["cases"][0]["passed"] is True
    assert persisted["cases"][0]["claim_level"] == "smoke"
    assert {
        item["claim_level"] for item in persisted["cases"]
    } <= {"smoke", "validation"}
    artifact_manifest = json.loads((tmp_path / "suite" / "artifact_manifest.json").read_text())
    assert artifact_manifest["claim_levels"]["linear_tearing_fast/manifest.json"] == "smoke"

    outdir = tmp_path / "cli-suite"
    cli_result = CliRunner().invoke(app, ["validate", "all", "--outdir", str(outdir)])
    assert cli_result.exit_code == 0, cli_result.stdout
    assert (outdir / "validation_suite.json").exists()
