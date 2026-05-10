"""Manifest writing utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mhx._version import __version__
from mhx.versioning import (
    ARTIFACT_MANIFEST_SCHEMA,
    MANIFEST_SCHEMA,
    require_supported_api_version,
    require_supported_claim_level,
)


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
    claim_level: str = "unspecified",
    claim_scope: str = "",
) -> dict[str, Any]:
    """Write a JSON manifest with output file hashes."""
    manifest_path = Path(path)
    run_dir = manifest_path.parent
    validated_claim_level = require_supported_claim_level(
        claim_level,
        context="manifest writer",
    )
    hashes = {
        key: _sha256_file(run_dir / relative_path)
        for key, relative_path in outputs.items()
        if (run_dir / relative_path).exists()
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "api_version": require_supported_api_version(context="manifest writer"),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "mhx_version": __version__,
        "claim_level": validated_claim_level,
        "claim_scope": claim_scope,
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
    claim_levels = {}
    for artifact_path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        if artifact_path.resolve() == manifest_path.resolve():
            continue
        relative_path = artifact_path.relative_to(root_path).as_posix()
        if artifact_path.name == "manifest.json":
            claim_level = _read_nested_claim_level(artifact_path)
            if claim_level is not None:
                claim_levels[relative_path] = claim_level
        files.append(
            {
                "path": relative_path,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": _sha256_file(artifact_path),
            }
        )
    manifest = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "api_version": require_supported_api_version(context="artifact-manifest writer"),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "mhx_version": __version__,
        "root": str(root_path),
        "claim_levels": claim_levels,
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _read_nested_claim_level(path: Path) -> str | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    claim_level = manifest.get("claim_level")
    if not isinstance(claim_level, str):
        return None
    return require_supported_claim_level(claim_level, context=f"nested manifest {path}")
