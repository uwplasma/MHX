"""Verify paper figure and artifact manifests before reviewer handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib
from mhx.versioning import ARTIFACT_MANIFEST_SCHEMA, CLAIM_LEVELS

ROOT = Path(__file__).resolve().parents[2]
DOCS_FIGURE_SCHEMA = "mhx.docs.figures_manifest.v1"
PAPER_CLAIM_LEVELS = set(CLAIM_LEVELS) - {"unspecified"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def relative_display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def verify_figure_manifest(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    figures = manifest.get("figures", [])

    require(manifest.get("schema") == DOCS_FIGURE_SCHEMA, "figure manifest schema mismatch", errors)
    require(isinstance(figures, list) and figures, "figure manifest has no figures", errors)

    seen_ids: set[str] = set()
    for index, figure in enumerate(figures):
        figure_id = figure.get("id", f"<entry {index}>")
        if figure_id in seen_ids:
            errors.append(f"duplicate figure id: {figure_id}")
        seen_ids.add(figure_id)

        for key in ("path", "command", "claim_level", "claim_scope", "sources", "tests", "sha256"):
            require(key in figure, f"{figure_id}: missing required field {key}", errors)

        claim_level = figure.get("claim_level")
        require(
            claim_level in PAPER_CLAIM_LEVELS,
            f"{figure_id}: unsupported paper claim level {claim_level!r}",
            errors,
        )
        if claim_level == "validation":
            claim_scope = str(figure.get("claim_scope", "")).lower()
            require(
                any(marker in claim_scope for marker in ("not ", "only", "no ")),
                f"{figure_id}: validation claim_scope lacks a production-boundary limitation",
                errors,
            )

        figure_path = repository_path(figure.get("path", ""))
        if figure_path.is_file():
            expected_hash = figure.get("sha256")
            actual_hash = sha256_file(figure_path)
            require(
                expected_hash == actual_hash,
                f"{figure_id}: sha256 mismatch for {relative_display(figure_path)}",
                errors,
            )
        else:
            errors.append(f"{figure_id}: missing figure file {relative_display(figure_path)}")

        for list_key in ("sources", "tests"):
            paths = figure.get(list_key, [])
            require(
                isinstance(paths, list) and paths,
                f"{figure_id}: {list_key} must be a non-empty list",
                errors,
            )
            for entry_path in paths if isinstance(paths, list) else []:
                path = repository_path(entry_path)
                require(
                    path.exists(),
                    f"{figure_id}: missing {list_key[:-1]} {relative_display(path)}",
                    errors,
                )

    return errors


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{relative_display(path)}: cannot read JSON ({exc})")
        return None


def verify_artifact_manifest(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = load_json(manifest_path, errors)
    if manifest is None:
        return errors

    root = manifest_path.parent
    require(
        manifest.get("schema") == ARTIFACT_MANIFEST_SCHEMA,
        f"{relative_display(manifest_path)}: artifact schema mismatch",
        errors,
    )

    files = manifest.get("files", [])
    require(
        isinstance(files, list),
        f"{relative_display(manifest_path)}: files is not a list",
        errors,
    )
    seen_paths: set[str] = set()
    listed_paths: list[str] = []
    for entry in files if isinstance(files, list) else []:
        relative_path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative_path, str):
            errors.append(f"{relative_display(manifest_path)}: file entry without string path")
            continue
        listed_paths.append(relative_path)
        if relative_path in seen_paths:
            errors.append(f"{relative_display(manifest_path)}: duplicate file {relative_path}")
        seen_paths.add(relative_path)
        artifact_path = (root / relative_path).resolve()
        require(
            artifact_path.is_relative_to(root.resolve()),
            f"{relative_display(manifest_path)}: file escapes manifest root: {relative_path}",
            errors,
        )
        if not artifact_path.is_file():
            errors.append(f"{relative_display(manifest_path)}: missing file {relative_path}")
            continue
        require(
            entry.get("size_bytes") == artifact_path.stat().st_size,
            f"{relative_display(manifest_path)}: size mismatch for {relative_path}",
            errors,
        )
        require(
            entry.get("sha256") == sha256_file(artifact_path),
            f"{relative_display(manifest_path)}: sha256 mismatch for {relative_path}",
            errors,
        )

    require(
        listed_paths == sorted(listed_paths),
        f"{relative_display(manifest_path)}: files are not sorted",
        errors,
    )
    require(
        "artifact_manifest.json" not in listed_paths,
        f"{relative_display(manifest_path)}: manifest includes itself",
        errors,
    )

    claim_levels = manifest.get("claim_levels", {})
    require(
        isinstance(claim_levels, dict),
        f"{relative_display(manifest_path)}: claim_levels is not an object",
        errors,
    )
    for nested_manifest in sorted(root.rglob("manifest.json")):
        relative_path = nested_manifest.relative_to(root).as_posix()
        nested = load_json(nested_manifest, errors)
        if nested is None:
            continue
        nested_claim_level = nested.get("claim_level")
        require(
            nested_claim_level in CLAIM_LEVELS,
            f"{relative_display(nested_manifest)}: unsupported claim level {nested_claim_level!r}",
            errors,
        )
        require(
            claim_levels.get(relative_path) == nested_claim_level,
            f"{relative_display(manifest_path)}: missing/stale claim level for {relative_path}",
            errors,
        )

    return errors


def artifact_manifests_for_roots(roots: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for root in roots:
        if root.name == "artifact_manifest.json" and root.is_file():
            manifests.append(root)
        elif root.is_dir():
            manifests.extend(sorted(root.rglob("artifact_manifest.json")))
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-manifest",
        default="docs/figures/manifest.toml",
        help="Documentation figure manifest to verify.",
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="Artifact root or artifact_manifest.json to verify; may be repeated.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only verify recursive artifact manifests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    if not args.skip_figures:
        figure_manifest = repository_path(args.figure_manifest)
        errors.extend(verify_figure_manifest(figure_manifest))

    artifact_roots = [repository_path(path) for path in args.artifact_root]
    artifact_manifests = artifact_manifests_for_roots(artifact_roots)
    for artifact_manifest in artifact_manifests:
        errors.extend(verify_artifact_manifest(artifact_manifest))

    for artifact_root in artifact_roots:
        if not artifact_root.exists():
            errors.append(f"missing artifact root {relative_display(artifact_root)}")
    if artifact_roots and not artifact_manifests:
        errors.append("no artifact_manifest.json files found under requested artifact roots")

    if errors:
        print("paper artifact verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "paper artifact verification passed: "
        f"figures={'skipped' if args.skip_figures else 'checked'}, "
        f"artifact_manifests={len(artifact_manifests)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
