from __future__ import annotations

import json

from typer.testing import CliRunner

from mhx.cli.main import app
from mhx.io import REDUCED_MHD_TRAJECTORY_SCHEMA, read_reduced_mhd_trajectory_npz


def test_cli_run_writes_schema_files(tmp_path) -> None:
    outdir = tmp_path / "smoke"
    result = CliRunner().invoke(
        app,
        ["run", "examples/linear_tearing.toml", "--outdir", str(outdir)],
    )
    assert result.exit_code == 0, result.stdout
    manifest = json.loads((outdir / "manifest.json").read_text())
    diagnostics = json.loads((outdir / "diagnostics.json").read_text())
    assert manifest["schema"] == "mhx.manifest.v1"
    assert manifest["claim_level"] == "smoke"
    assert manifest["hashes"]["diagnostics"]
    assert manifest["hashes"]["trajectory"]
    assert diagnostics["grid_shape"] == [32, 32]
    assert diagnostics["mesh_lower"] == [0.0, 0.0]
    assert diagnostics["n_steps"] == 10.0
    assert diagnostics["diagnostic_mode"] == [1, 1]
    assert diagnostics["diagnostic_quantities"] == [
        "energy",
        "mode_growth",
        "divergence_error",
    ]
    assert diagnostics["final_magnetic_divergence_linf"] < 1.0e-10
    assert diagnostics["fit_time_window"] == [0.02, 0.1]
    assert diagnostics["fit_sample_count"] == 9.0
    assert diagnostics["gamma_fit"] < 0.0
    assert diagnostics["final_total_energy"] > 0.0
    assert diagnostics["final_total_energy"] <= diagnostics["initial_total_energy"]
    trajectory, _ = read_reduced_mhd_trajectory_npz(outdir / "trajectory.npz")
    assert trajectory.states.psi.shape == (10, 32, 32)


def test_cli_init_writes_config(tmp_path) -> None:
    path = tmp_path / "new_config.toml"
    result = CliRunner().invoke(app, ["init", str(path)])
    assert result.exit_code == 0
    assert "[physics]" in path.read_text()


def test_cli_init_refuses_existing_file(tmp_path) -> None:
    path = tmp_path / "new_config.toml"
    path.write_text("name = 'existing'\n")
    result = CliRunner().invoke(app, ["init", str(path)])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_cli_run_uses_config_output_dir(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    outdir = tmp_path / "configured-output"
    text = (
        'name = "configured"\n'
        f'output_dir = "{outdir.as_posix()}"\n'
        "\n[mesh]\n"
        "shape = [8, 8]\n"
        "lower = [0.0, 0.0]\n"
        "upper = [6.283185307179586, 6.283185307179586]\n"
        "periodic = [true, true]\n"
        "\n[time]\n"
        "t1 = 0.02\n"
        "dt = 0.01\n"
        "save_every = 1\n"
    )
    config_path.write_text(text)
    result = CliRunner().invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0, result.stdout
    assert (outdir / "manifest.json").exists()


def test_cli_figures_regenerates_pngs(tmp_path) -> None:
    outdir = tmp_path / "smoke"
    run_result = CliRunner().invoke(
        app,
        ["run", "examples/linear_tearing.toml", "--outdir", str(outdir)],
    )
    assert run_result.exit_code == 0, run_result.stdout
    figure_result = CliRunner().invoke(app, ["figures", str(outdir), "--gif"])
    assert figure_result.exit_code == 0, figure_result.stdout
    assert (outdir / "figures" / "energy_history.png").stat().st_size > 0
    assert (outdir / "figures" / "flux_final.png").stat().st_size > 0
    assert (outdir / "figures" / "mode_amplitude.png").stat().st_size > 0
    assert (outdir / "figures" / "flux_movie.gif").stat().st_size > 0


def test_cli_report_writes_json_and_markdown(tmp_path) -> None:
    outdir = tmp_path / "smoke"
    run_result = CliRunner().invoke(
        app,
        ["run", "examples/linear_tearing.toml", "--outdir", str(outdir)],
    )
    assert run_result.exit_code == 0, run_result.stdout
    report_result = CliRunner().invoke(app, ["report", str(outdir)])
    assert report_result.exit_code == 0, report_result.stdout
    assert "gamma fit" in (outdir / "report.md").read_text()
    report = json.loads((outdir / "report.json").read_text())
    assert report["schema"] == "mhx.benchmark_report.v1"


def test_benchmark_pipeline_and_validation(tmp_path) -> None:
    outdir = tmp_path / "benchmark"
    run_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run",
            "--config",
            "examples/linear_tearing.toml",
            "--outdir",
            str(outdir),
            "--gif",
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout
    assert (outdir / "manifest.json").exists()
    assert (outdir / "figures" / "energy_history.png").exists()
    assert (outdir / "figures" / "flux_movie.gif").exists()
    assert (outdir / "report.md").exists()
    validate_result = CliRunner().invoke(app, ["benchmark", "validate", str(outdir)])
    assert validate_result.exit_code == 0, validate_result.stdout
    validation = json.loads((outdir / "validation.json").read_text())
    assert validation["passed"] is True


def test_benchmark_validation_failure_exits_nonzero(tmp_path) -> None:
    outdir = tmp_path / "benchmark"
    run_result = CliRunner().invoke(
        app,
        ["benchmark", "run", "--outdir", str(outdir), "--no-figures", "--no-report"],
    )
    assert run_result.exit_code == 0, run_result.stdout
    result = CliRunner().invoke(
        app,
        ["benchmark", "validate", str(outdir), "--max-relative-energy-growth", "-1.0"],
    )
    assert result.exit_code == 1


def test_npz_schema_constant_is_versioned() -> None:
    assert REDUCED_MHD_TRAJECTORY_SCHEMA == "mhx.reduced_mhd.trajectory.v1"


def test_cli_diagnostics_list() -> None:
    result = CliRunner().invoke(app, ["diagnostics", "list"])
    assert result.exit_code == 0
    assert "energy" in result.output
    assert "mode_growth" in result.output
    assert "divergence_error" in result.output
