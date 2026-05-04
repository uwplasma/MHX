from __future__ import annotations

from typer.testing import CliRunner

from mhx import __version__
from mhx.cli.main import app


def test_import_version_string() -> None:
    assert __version__
    assert __version__.count(".") >= 2


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MHX differentiable MHD workflows" in result.stdout

