"""Offline media rendering and strict checks for committed documentation assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
README_MEDIA_DIR = ROOT / "docs" / "_static" / "readme"
MOVIES_DIR = ROOT / "docs" / "_static" / "movies"
PROVENANCE_DIR = ROOT / "docs" / "_static" / "media_records"
MANIFEST_PATH = ROOT / "docs" / "figures" / "manifest.toml"
SCHEMA = "mhx.docs.figures_manifest.v1"
REQUIRED = {"id", "path", "sha256", "command", "claim_level", "claim_scope", "sources", "tests"}
CLAIMS = {"demonstration", "smoke", "validation", "production_template"}
SIZE_LIMITS = {".gif": 20_000_000, ".mp4": 6_000_000, ".png": 3_000_000}
CAMPAIGN_SIZE_LIMIT = 75_000_000
CURATED_RENDER_IDS = {
    "decaying_mhd_turbulence_current_256",
    "double_harris_current_sheet",
    "double_harris_island_64",
    "double_harris_reconnection",
    "forced_2d_turbulence",
    "forced_2d_turbulence_spectrum",
    "forced_3d_turbulence_current",
    "kelvin_helmholtz",
    "orszag_tang_3d_current",
    "orszag_tang_current",
}
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
VIDEO_RE = re.compile(r"```\{video\}\s+(?P<target>\S+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature_matches(path: Path) -> bool:
    header = path.read_bytes()[:12]
    return {
        ".png": header.startswith(b"\x89PNG\r\n\x1a\n"),
        ".gif": header.startswith((b"GIF87a", b"GIF89a")),
        ".mp4": len(header) >= 8 and header[4:8] == b"ftyp",
    }.get(path.suffix.lower(), True)


def _media_links() -> list[tuple[Path, str]]:
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    links: list[tuple[Path, str]] = []
    for page in pages:
        if "_build" in page.parts:
            continue
        text = page.read_text(encoding="utf-8")
        links.extend((page, match.group("target")) for match in IMAGE_RE.finditer(text))
        links.extend((page, match.group("target")) for match in VIDEO_RE.finditer(text))
    return links


def _collect_provenance_errors() -> list[str]:
    """Validate promoted outputs against their portable render records."""
    errors: list[str] = []
    render_ids: set[str] = set()
    output_paths: set[str] = set()
    total_bytes = 0
    for record_path in sorted(PROVENANCE_DIR.glob("*.json")):
        label = record_path.relative_to(ROOT).as_posix()
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: cannot read provenance: {exc}")
            continue
        render_id = record.get("id")
        if not isinstance(render_id, str) or not render_id:
            errors.append(f"{label}: missing render id")
            continue
        if render_id in render_ids:
            errors.append(f"duplicate render id: {render_id}")
        render_ids.add(render_id)
        if record.get("schema") != "mhx.media.render.v1":
            errors.append(f"{render_id}: unsupported provenance schema")
        source_hash = record.get("source_sha256")
        if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            errors.append(f"{render_id}: invalid source_sha256")
        outputs = record.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            errors.append(f"{render_id}: outputs must be a non-empty table")
            continue
        for output_name, output in outputs.items():
            if not isinstance(output, dict):
                errors.append(f"{render_id}: malformed output {output_name}")
                continue
            relative = output.get("path")
            pure = PurePosixPath(relative) if isinstance(relative, str) else None
            if (
                pure is None
                or pure.is_absolute()
                or ".." in pure.parts
                or pure.parts[:2] != ("docs", "_static")
            ):
                errors.append(f"{render_id}: invalid output path {relative!r}")
                continue
            if relative in output_paths:
                errors.append(f"duplicate promoted output path: {relative}")
                continue
            output_paths.add(relative)
            asset = ROOT / pure
            if not asset.is_file():
                errors.append(f"{render_id}: missing promoted output {relative}")
                continue
            actual_bytes = asset.stat().st_size
            total_bytes += actual_bytes
            if output.get("bytes") != actual_bytes:
                errors.append(f"{render_id}: byte count mismatch for {relative}")
            if output.get("sha256") != _sha256(asset):
                errors.append(f"{render_id}: sha256 mismatch for {relative}")
            if not _signature_matches(asset):
                errors.append(f"{render_id}: content does not match extension for {relative}")
            limit = SIZE_LIMITS.get(asset.suffix.lower())
            if limit is not None and actual_bytes > limit:
                errors.append(
                    f"{render_id}: {actual_bytes} bytes exceeds {limit}-byte budget for {relative}"
                )
    missing = sorted(CURATED_RENDER_IDS - render_ids)
    unexpected = sorted(render_ids - CURATED_RENDER_IDS)
    if missing:
        errors.append(f"missing curated render records: {missing}")
    if unexpected:
        errors.append(f"unexpected curated render records: {unexpected}")
    if total_bytes > CAMPAIGN_SIZE_LIMIT:
        errors.append(
            f"promoted campaign is {total_bytes} bytes; budget is {CAMPAIGN_SIZE_LIMIT} bytes"
        )
    return errors


def collect_media_errors() -> list[str]:
    """Return all read-only manifest, hash, link, type, and size failures."""
    try:
        manifest = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"cannot read manifest: {exc}"]

    errors: list[str] = _collect_provenance_errors()
    if manifest.get("schema") != SCHEMA:
        errors.append(f"manifest schema must be {SCHEMA!r}")
    entries = manifest.get("figures")
    if not isinstance(entries, list):
        return [*errors, "manifest must contain [[figures]] entries"]

    ids: set[str] = set()
    paths: set[str] = set()
    registered_movies: set[str] = set()
    for number, entry in enumerate(entries, 1):
        label = entry.get("id", f"entry {number}") if isinstance(entry, dict) else f"entry {number}"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry is not a table")
            continue
        missing = sorted(REQUIRED - entry.keys())
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue
        if not isinstance(entry["id"], str) or not entry["id"]:
            errors.append(f"{label}: id must be a non-empty string")
        elif entry["id"] in ids:
            errors.append(f"{label}: duplicate id")
        ids.add(entry["id"])

        asset_path = entry["path"]
        pure = PurePosixPath(asset_path) if isinstance(asset_path, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.parts[:2] != ("docs", "_static")
        ):
            errors.append(f"{label}: path must stay under docs/_static: {asset_path!r}")
            continue
        if asset_path in paths:
            errors.append(f"{label}: duplicate path {asset_path}")
        paths.add(asset_path)
        if pure.suffix.lower() == ".mp4":
            registered_movies.add(asset_path)

        path = ROOT / pure
        if not path.is_file():
            errors.append(f"{label}: missing file {asset_path}")
            continue
        actual_hash = _sha256(path)
        expected_hash = entry["sha256"]
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            errors.append(f"{label}: invalid sha256")
        elif expected_hash != actual_hash:
            errors.append(
                f"{label}: sha256 mismatch (manifest {expected_hash}, actual {actual_hash})"
            )
        if not _signature_matches(path):
            errors.append(f"{label}: content does not match {path.suffix.lower()} extension")
        limit = SIZE_LIMITS.get(path.suffix.lower())
        if limit is not None and path.stat().st_size > limit:
            errors.append(f"{label}: {path.stat().st_size} bytes exceeds {limit}-byte budget")
        if entry["claim_level"] not in CLAIMS:
            errors.append(f"{label}: unsupported claim_level {entry['claim_level']!r}")
        for field in ("command", "claim_scope"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                errors.append(f"{label}: {field} must be non-empty")
        for field in ("sources", "tests"):
            values = entry[field]
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value for value in values)
            ):
                errors.append(f"{label}: {field} must be a non-empty string list")
                continue
            for value in values:
                if (
                    value.startswith(("src/", "examples/", "docs/", "tests/", "tools/"))
                    and not (ROOT / value).exists()
                ):
                    errors.append(f"{label}: missing {field[:-1]} path {value}")

    actual_movies = {path.relative_to(ROOT).as_posix() for path in MOVIES_DIR.glob("*.mp4")}
    errors.extend(
        f"unregistered documentation movie: {path}"
        for path in sorted(actual_movies - registered_movies)
    )

    for page, target in _media_links():
        if urlsplit(target).scheme:
            continue
        linked = (page.parent / Path(unquote(urlsplit(target).path))).resolve()
        try:
            linked.relative_to(ROOT)
        except ValueError:
            errors.append(f"{page.relative_to(ROOT)}: media target escapes repository: {target}")
            continue
        if not linked.is_file():
            errors.append(f"{page.relative_to(ROOT)}: missing media target {target}")
        elif (
            linked.suffix.lower() == ".mp4"
            and linked.relative_to(ROOT).as_posix() not in registered_movies
        ):
            errors.append(f"{page.relative_to(ROOT)}: unregistered movie target {target}")
    return errors


def check_media_assets() -> bool:
    errors = collect_media_errors()
    if errors:
        print("Committed media check failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        return False
    print(
        "Committed media check passed: manifest, provenance, hashes, links, types, "
        "and size budgets agree."
    )
    return True


def render_media(preset: str) -> None:
    """Delegate all staged rendering to the unified example campaign."""
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/media/run_all.py"),
            "render",
            "--preset",
            preset,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate without rendering.")
    parser.add_argument("--render-all", action="store_true", help="Render explicit sources.")
    parser.add_argument("--preset", choices=("preview", "final"), default="final")
    args = parser.parse_args()
    if args.check_only or len(sys.argv) == 1:
        raise SystemExit(0 if check_media_assets() else 1)
    if args.render_all:
        render_media(args.preset)
    raise SystemExit(0 if check_media_assets() else 1)


if __name__ == "__main__":
    main()
