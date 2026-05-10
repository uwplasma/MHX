from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mhx.cli.main import app
from mhx.io import write_artifact_manifest, write_manifest


def test_write_artifact_manifest_hashes_recursive_files(tmp_path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "figures" / "b.txt").write_text("beta", encoding="utf-8")
    write_manifest(
        tmp_path / "manifest.json",
        config={"name": "unit"},
        outputs={"a": "a.txt"},
        claim_level="validation",
        claim_scope="Unit-test nested manifest.",
    )

    manifest = write_artifact_manifest(tmp_path)
    paths = [item["path"] for item in manifest["files"]]
    assert manifest["schema"] == "mhx.artifacts.v1"
    assert paths == ["a.txt", "figures/b.txt", "manifest.json"]
    assert manifest["claim_levels"] == {"manifest.json": "validation"}
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    assert json.loads((tmp_path / "artifact_manifest.json").read_text())["files"] == manifest[
        "files"
    ]


def test_artifact_manifest_cli_and_missing_root(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["artifact-manifest", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 files" in result.stdout

    with pytest.raises(FileNotFoundError, match="artifact root"):
        write_artifact_manifest(tmp_path / "missing")
