"""Manifest writing utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mhx._version import __version__


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: str | Path,
    *,
    config: dict[str, Any],
    outputs: dict[str, str],
) -> dict[str, Any]:
    """Write a JSON manifest with output file hashes."""
    manifest_path = Path(path)
    run_dir = manifest_path.parent
    hashes = {
        key: _sha256_file(run_dir / relative_path)
        for key, relative_path in outputs.items()
        if (run_dir / relative_path).exists()
    }
    manifest = {
        "schema": "mhx.manifest.v1",
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "mhx_version": __version__,
        "config": config,
        "outputs": outputs,
        "hashes": hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_artifact_manifest(
    root: str | Path,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a recursive artifact manifest with file sizes and SHA-256 hashes."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"artifact root does not exist or is not a directory: {root_path}")
    manifest_path = Path(path) if path is not None else root_path / "artifact_manifest.json"
    files = []
    for artifact_path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        if artifact_path.resolve() == manifest_path.resolve():
            continue
        relative_path = artifact_path.relative_to(root_path).as_posix()
        files.append(
            {
                "path": relative_path,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": _sha256_file(artifact_path),
            }
        )
    manifest = {
        "schema": "mhx.artifacts.v1",
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "mhx_version": __version__,
        "root": str(root_path),
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
