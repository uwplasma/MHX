from __future__ import annotations

import json

from typer.testing import CliRunner

from mhx.cli.main import app


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
    assert manifest["hashes"]["diagnostics"]
    assert diagnostics["grid_shape"] == [32, 32]
    assert diagnostics["spectral_smoke_max_error"] < 1.0e-10


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
    )
    config_path.write_text(text)
    result = CliRunner().invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0, result.stdout
    assert (outdir / "manifest.json").exists()
