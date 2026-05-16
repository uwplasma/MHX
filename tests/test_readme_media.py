from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import imageio.v2 as imageio

from mhx.benchmarks import double_harris_seeded_long_run_presets

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)

README_GIF_BUDGETS = {
    "docs/_static/readme/double_harris_reconnection.gif": 350_000,
    "docs/_static/readme/double_harris_current_sheet.gif": 400_000,
    "docs/_static/readme/orszag_tang_current.gif": 250_000,
    "docs/_static/readme/orszag_tang_vorticity.gif": 350_000,
    "docs/_static/readme/orszag_tang_flux.gif": 400_000,
    "docs/_static/readme/harris_layer_sweep.gif": 200_000,
    "docs/_static/readme/plasmoid_scaling_schematic.gif": 300_000,
    "docs/_static/readme/mhd_turbulence_cascade.gif": 500_000,
}
DEFAULT_README_GIF_BUDGET = 750_000
INTERNAL_README_SECTIONS = {
    "Reviewer Trail",
    "Landing-Page Audit",
}
README_SOLVER_MEDIA_TARGETS = {
    "docs/_static/readme/double_harris_reconnection.gif",
    "docs/_static/readme/double_harris_current_sheet.gif",
    "docs/_static/readme/orszag_tang_current.gif",
    "docs/_static/readme/orszag_tang_vorticity.gif",
    "docs/_static/readme/orszag_tang_flux.gif",
}


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


def test_readme_excludes_internal_reviewer_sections() -> None:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = {match.group("title") for match in HEADING_RE.finditer(readme_text)}

    assert headings.isdisjoint(INTERNAL_README_SECTIONS)


def test_readme_has_auto_updated_coverage_badge() -> None:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    badge_payload = json.loads((ROOT / "badges" / "coverage.json").read_text())

    assert "img.shields.io/endpoint" in readme_text
    assert "raw.githubusercontent.com/uwplasma/MHX/main/badges/coverage.json" in readme_text
    assert badge_payload["schemaVersion"] == 1
    assert badge_payload["label"] == "coverage"
    assert badge_payload["message"].endswith("%")


def test_readme_solver_media_has_longer_validation_provenance() -> None:
    media_text = (ROOT / "docs" / "media.md").read_text(encoding="utf-8")
    qa_manifest = json.loads(
        (ROOT / "docs" / "_static" / "readme" / "readme_media_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    local_targets = _local_image_targets(ROOT / "README.md")
    solver_targets = [
        target
        for target in local_targets
        if "double_harris" in Path(target).name or "orszag_tang" in Path(target).name
    ]

    assert solver_targets
    assert set(solver_targets) <= README_SOLVER_MEDIA_TARGETS
    assert qa_manifest["schema"] == "mhx.readme_media_visual_qa.v1"
    assert "claim_level = \"validation\"" in media_text
    assert "readme_media_visual_qa.json" in media_text

    manifest_by_path = {item["path"]: item for item in qa_manifest["media"]}
    for readme_target in solver_targets:
        readme_movie = ROOT / readme_target
        manifest_entry = manifest_by_path[readme_target]

        if "double_harris" in readme_target:
            assert manifest_entry["t_end"] >= 100.0
            assert manifest_entry["source"]["source_shape"] == [128, 128]
            assert "seeded-minus-base" in manifest_entry["notes"]
        if "orszag_tang" in readme_target:
            assert manifest_entry["t_end"] >= 10.0
            assert manifest_entry["source"]["source_shape"] == [96, 96]
            assert "Orszag-Tang" in manifest_entry["notes"]
        assert manifest_entry["source"]["validation_passed"] is True
        assert readme_target.removeprefix("docs/") in media_text
        assert len(imageio.mimread(readme_movie)) >= 20

    ot_metrics = qa_manifest["visual_qa"]["orszag_tang"]["metrics"]
    assert ot_metrics["current_high_k_peak"] > ot_metrics["current_high_k_first"]
    assert ot_metrics["vorticity_high_k_peak"] > ot_metrics["vorticity_high_k_first"]
    assert ot_metrics["relative_energy_drop"] > 0.1


def test_readme_source_media_policy_exceeds_documented_minimum() -> None:
    preset = double_harris_seeded_long_run_presets()["readme_release_media"]

    assert preset["duration_label"] == "readme_release_media"
    assert preset["t_end"] > preset["documented_minimum_t_end"]
    assert "double_harris_seeded_long_run_presets" in (
        ROOT / "examples" / "make_validation_media.py"
    ).read_text(encoding="utf-8")
