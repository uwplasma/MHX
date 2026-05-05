from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mhx import MHX_PUBLIC_API_VERSION, api_version_info
from mhx.cli.main import app
from mhx.versioning import (
    REDUCED_MHD_TRAJECTORY_SCHEMA,
    active_api_version,
    require_supported_api_version,
)


def test_api_version_info_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("MHX_API_VERSION", raising=False)
    assert active_api_version() == "v1"
    assert MHX_PUBLIC_API_VERSION == "v1"
    info = api_version_info()
    assert info.public_api_version == "v1"
    assert info.trajectory_schema == REDUCED_MHD_TRAJECTORY_SCHEMA

    monkeypatch.setenv("MHX_API_VERSION", "v2")
    assert active_api_version() == "v2"
    with pytest.raises(ValueError, match="unsupported API version"):
        require_supported_api_version(context="test")


def test_cli_api_status_and_deprecations() -> None:
    runner = CliRunner()
    status = runner.invoke(app, ["api", "status", "--json"])
    assert status.exit_code == 0, status.stdout
    payload = json.loads(status.stdout)
    assert payload["public_api_version"] == "v1"
    assert payload["physics_api_version"] == "mhx.physics.v1"

    deprecations = runner.invoke(app, ["api", "deprecations"])
    assert deprecations.exit_code == 0, deprecations.stdout
    assert "legacy/old_mhx" in deprecations.stdout
