from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)
README_GIF_BUDGET = 20_000_000
MEDIA_CASES = {
    "decaying_turbulence.py",
    "double_harris_reconnection.py",
    "forced_turbulence_2d.py",
    "orszag_tang_2d.py",
    "orszag_tang_3d.py",
    "kelvin_helmholtz.py",
    "turbulence_3d.py",
}


def _local_image_targets(markdown_path: Path) -> list[str]:
    targets = []
    for match in IMAGE_LINK_RE.finditer(markdown_path.read_text(encoding="utf-8")):
        target = match.group("target")
        if not urlsplit(target).scheme:
            targets.append(target)
    return targets


def _resolve(markdown_path: Path, target: str) -> Path:
    relative = Path(unquote(urlsplit(target).path))
    return (markdown_path.parent / relative).resolve()


def test_readme_image_links_point_to_existing_files() -> None:
    readme = ROOT / "README.md"
    missing = [
        target for target in _local_image_targets(readme) if not _resolve(readme, target).is_file()
    ]
    assert missing == []


def test_readme_gifs_stay_within_the_repository_budget() -> None:
    readme = ROOT / "README.md"
    oversized = []
    for target in _local_image_targets(readme):
        if target.endswith(".gif"):
            size = _resolve(readme, target).stat().st_size
            if size > README_GIF_BUDGET:
                oversized.append((target, size))
    assert oversized == []


def test_media_campaign_replaces_the_monolithic_generator() -> None:
    media_dir = ROOT / "examples" / "media"
    assert not (ROOT / "examples" / "make_readme_media.py").exists()
    assert {path.name for path in media_dir.glob("*.py")} >= MEDIA_CASES
    assert (media_dir / "run_all.py").is_file()
    assert (media_dir / "common.py").is_file()


def test_readme_uses_only_landing_page_media() -> None:
    targets = _local_image_targets(ROOT / "README.md")
    assert all("docs/_static/readme/" in target for target in targets)


def test_readme_excludes_internal_reviewer_sections() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = {match.group("title") for match in HEADING_RE.finditer(text)}
    assert headings.isdisjoint({"Reviewer Trail", "Landing-Page Audit"})


def test_readme_has_auto_updated_coverage_badge() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    payload = json.loads((ROOT / "badges" / "coverage.json").read_text())
    assert "img.shields.io/endpoint" in text
    assert "raw.githubusercontent.com/uwplasma/MHX/main/badges/coverage.json" in text
    assert payload["schemaVersion"] == 1
    assert payload["label"] == "coverage"
    assert payload["message"].endswith("%")
