from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")

README_GIF_BUDGETS = {
    "docs/_static/readme/double_harris_reconnection.gif": 250_000,
    "docs/_static/readme/double_harris_current_sheet.gif": 300_000,
    "docs/_static/readme/harris_layer_sweep.gif": 200_000,
    "docs/_static/readme/plasmoid_scaling_schematic.gif": 300_000,
    "docs/_static/readme/mhd_turbulence_cascade.gif": 500_000,
}
DEFAULT_README_GIF_BUDGET = 750_000


def _local_image_targets(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    targets = []
    for match in IMAGE_LINK_RE.finditer(text):
        target = match.group("target")
        if urlsplit(target).scheme:
            continue
        targets.append(target)
    return targets


def _resolve_markdown_target(markdown_path: Path, target: str) -> Path:
    split_target = urlsplit(target)
    relative_path = Path(unquote(split_target.path))
    return (markdown_path.parent / relative_path).resolve()


def test_readme_image_links_point_to_existing_files() -> None:
    readme_path = ROOT / "README.md"
    missing = [
        target
        for target in _local_image_targets(readme_path)
        if not _resolve_markdown_target(readme_path, target).is_file()
    ]
    assert missing == []


def test_readme_gifs_are_compact() -> None:
    readme_path = ROOT / "README.md"
    gif_targets = [
        target for target in _local_image_targets(readme_path) if target.endswith(".gif")
    ]
    assert gif_targets

    oversized = []
    for target in gif_targets:
        gif_path = _resolve_markdown_target(readme_path, target)
        budget = README_GIF_BUDGETS.get(target, DEFAULT_README_GIF_BUDGET)
        if gif_path.stat().st_size >= budget:
            oversized.append((target, gif_path.stat().st_size, budget))
    assert oversized == []


def test_readme_uses_only_landing_page_media() -> None:
    readme_path = ROOT / "README.md"
    local_targets = _local_image_targets(readme_path)
    assert all("docs/_static/readme/" in target for target in local_targets)
    assert all(not target.endswith(".png") for target in local_targets)
