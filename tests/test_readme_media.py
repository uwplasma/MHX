from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import imageio.v2 as imageio
import numpy as np

from mhx.benchmarks import double_harris_seeded_long_run_presets

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)

README_GIF_BUDGETS = {
    "docs/_static/readme/double_harris_reconnection.gif": 500_000,
    "docs/_static/readme/double_harris_current_sheet.gif": 600_000,
    "docs/_static/readme/forced_turbulent_reconnection.gif": 700_000,
    "docs/_static/readme/decaying_mhd_turbulence_current.gif": 750_000,
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
    "docs/_static/readme/forced_turbulent_reconnection.gif",
    "docs/_static/readme/decaying_mhd_turbulence_current.gif",
}
README_SOLVER_GIF_MIN_FRAMES = {
    "docs/_static/readme/double_harris_reconnection.gif": 18,
    "docs/_static/readme/double_harris_current_sheet.gif": 18,
    "docs/_static/readme/orszag_tang_current.gif": 30,
    "docs/_static/readme/orszag_tang_vorticity.gif": 30,
    "docs/_static/readme/orszag_tang_flux.gif": 30,
    "docs/_static/readme/forced_turbulent_reconnection.gif": 20,
    "docs/_static/readme/decaying_mhd_turbulence_current.gif": 20,
}
README_SOLVER_GIF_MIN_DYNAMIC_RANGE = 64.0
README_SOLVER_GIF_MIN_MEAN_FRAME_DELTA = 1.0
README_SOLVER_GIF_MIN_CHANGED_PIXEL_FRACTION = 0.10
README_SOLVER_GIF_PIXEL_CHANGE_DELTA = 2.0


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


@functools.cache
def _read_visual_qa_manifest() -> dict[str, object]:
    manifest_path = ROOT / "docs" / "_static" / "readme" / "readme_media_visual_qa.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@functools.cache
def _read_gif_frames(gif_path: Path) -> tuple[np.ndarray, int | float | None]:
    reader = imageio.get_reader(gif_path)
    try:
        metadata = reader.get_meta_data()
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()

    if not frames:
        raise AssertionError(f"{gif_path} did not contain any readable frames")

    try:
        frame_array = np.stack(frames)
    except ValueError as exc:
        raise AssertionError(f"{gif_path} frames have inconsistent shapes") from exc

    duration_ms = metadata.get("duration")
    return frame_array, duration_ms


def _gif_grayscale_frames(frame_array: np.ndarray) -> np.ndarray:
    if frame_array.ndim == 4 and frame_array.shape[-1] >= 3:
        return frame_array[..., :3].astype(np.float32).mean(axis=-1)
    return frame_array.astype(np.float32)


def _gif_visual_metrics(gif_path: Path) -> dict[str, float]:
    frame_array, _ = _read_gif_frames(gif_path)
    grayscale_frames = _gif_grayscale_frames(frame_array)
    if frame_array.shape[0] > 1:
        frame_deltas = np.abs(np.diff(grayscale_frames, axis=0))
        mean_frame_delta = float(frame_deltas.mean())
        changed_pixel_fraction = float(
            np.mean(frame_deltas > README_SOLVER_GIF_PIXEL_CHANGE_DELTA)
        )
    else:
        mean_frame_delta = 0.0
        changed_pixel_fraction = 0.0

    return {
        "frame_count": float(frame_array.shape[0]),
        "dynamic_range": float(grayscale_frames.max() - grayscale_frames.min()),
        "mean_frame_delta": mean_frame_delta,
        "changed_pixel_fraction": changed_pixel_fraction,
    }


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


def test_readme_solver_gifs_are_visually_dynamic() -> None:
    failures = []

    for target in sorted(README_SOLVER_MEDIA_TARGETS):
        metrics = _gif_visual_metrics(ROOT / target)
        thresholds = {
            "frame_count": float(README_SOLVER_GIF_MIN_FRAMES[target]),
            "dynamic_range": README_SOLVER_GIF_MIN_DYNAMIC_RANGE,
            "mean_frame_delta": README_SOLVER_GIF_MIN_MEAN_FRAME_DELTA,
            "changed_pixel_fraction": README_SOLVER_GIF_MIN_CHANGED_PIXEL_FRACTION,
        }

        for metric_name, threshold in thresholds.items():
            observed = metrics[metric_name]
            if observed < threshold:
                failures.append(
                    f"{target}: {metric_name}={observed:.3g} < threshold {threshold:.3g}"
                )

    assert failures == []


def test_readme_solver_gifs_match_visual_qa_manifest() -> None:
    qa_manifest = _read_visual_qa_manifest()
    assert qa_manifest["schema"] == "mhx.readme_media_visual_qa.v1"

    manifest_by_path = {item["path"]: item for item in qa_manifest["media"]}
    missing_targets = sorted(README_SOLVER_MEDIA_TARGETS - set(manifest_by_path))
    assert missing_targets == []

    failures = []
    for target in sorted(README_SOLVER_MEDIA_TARGETS):
        frame_array, duration_ms = _read_gif_frames(ROOT / target)
        manifest_entry = manifest_by_path[target]
        manifest_source = manifest_entry.get("source")
        if not isinstance(manifest_source, dict):
            failures.append(f"{target}: manifest source is not solver provenance metadata")
            continue

        actual_frame_count = frame_array.shape[0]
        actual_frame_shape = list(frame_array[0].shape)
        if manifest_entry.get("frame_count") != actual_frame_count:
            failures.append(
                f"{target}: manifest frame_count={manifest_entry.get('frame_count')} "
                f"!= actual {actual_frame_count}"
            )
        if manifest_entry.get("frame_shape") != actual_frame_shape:
            failures.append(
                f"{target}: manifest frame_shape={manifest_entry.get('frame_shape')} "
                f"!= actual {actual_frame_shape}"
            )
        if manifest_entry.get("duration_ms") != duration_ms:
            failures.append(
                f"{target}: manifest duration_ms={manifest_entry.get('duration_ms')} "
                f"!= actual {duration_ms}"
            )
        for field_name in ("t_end", "time_span"):
            if manifest_entry.get(field_name) != manifest_source.get(field_name):
                failures.append(
                    f"{target}: manifest {field_name}={manifest_entry.get(field_name)} "
                    f"!= source {manifest_source.get(field_name)}"
                )
        if manifest_source.get("validation_passed") is not True:
            failures.append(f"{target}: source validation_passed is not true")
        if manifest_source.get("source_samples", 0) < actual_frame_count:
            failures.append(
                f"{target}: source_samples={manifest_source.get('source_samples')} "
                f"< frame_count {actual_frame_count}"
            )

    assert failures == []


def test_readme_uses_only_landing_page_media() -> None:
    readme_path = ROOT / "README.md"
    local_targets = _local_image_targets(readme_path)
    assert all("docs/_static/readme/" in target for target in local_targets)
    assert all(
        not target.endswith(".png") or target.endswith("/strong_scaling.png")
        for target in local_targets
    )


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
    qa_manifest = _read_visual_qa_manifest()
    local_targets = _local_image_targets(ROOT / "README.md")
    readme_solver_targets = [
        target
        for target in local_targets
        if (
            "double_harris" in Path(target).name
            or "orszag_tang" in Path(target).name
            or "turbulence" in Path(target).name
            or "turbulent_reconnection" in Path(target).name
        )
    ]

    assert readme_solver_targets
    assert set(readme_solver_targets) <= README_SOLVER_MEDIA_TARGETS
    assert qa_manifest["schema"] == "mhx.readme_media_visual_qa.v1"
    assert "claim_level = \"validation\"" in media_text
    assert "readme_media_visual_qa.json" in media_text

    manifest_by_path = {item["path"]: item for item in qa_manifest["media"]}
    assert set(manifest_by_path) >= README_SOLVER_MEDIA_TARGETS
    for readme_target in sorted(README_SOLVER_MEDIA_TARGETS):
        readme_movie = ROOT / readme_target
        manifest_entry = manifest_by_path[readme_target]

        if "double_harris" in readme_target:
            assert manifest_entry["t_end"] >= 160.0
            assert manifest_entry["source"]["source_shape"] == [128, 128]
            assert "gpu_nonlinear_20260522_085049" in manifest_entry["source"]["source"]
            assert "residual" in manifest_entry["notes"]
            assert "total magnetic-flux/Az contours" in manifest_entry["notes"]
            assert "static total-field" not in manifest_entry["notes"]
            assert "X/O" in (
                manifest_entry["notes"]
                + " "
                + " ".join(qa_manifest["visual_qa"]["double_harris"]["observations"])
            )
        if "orszag_tang" in readme_target:
            assert manifest_entry["t_end"] >= 10.0
            assert manifest_entry["source"]["source_shape"] == [96, 96]
            assert "Orszag-Tang" in manifest_entry["notes"]
        if "decaying_mhd_turbulence" in readme_target:
            assert manifest_entry["t_end"] >= 8.0
            assert manifest_entry["source"]["source_shape"] == [64, 64]
            assert "decaying reduced-MHD turbulence" in manifest_entry["notes"]
        if "forced_turbulent_reconnection" in readme_target:
            assert manifest_entry["t_end"] >= 80.0
            assert manifest_entry["source"]["source_shape"] == [64, 64]
            assert "reconnection-rate proxy" in manifest_entry["notes"]
        assert manifest_entry["source"]["validation_passed"] is True
        assert readme_target.removeprefix("docs/") in media_text
        frame_array, _ = _read_gif_frames(readme_movie)
        assert frame_array.shape[0] >= README_SOLVER_GIF_MIN_FRAMES[readme_target]

    dh_metrics = qa_manifest["visual_qa"]["double_harris"]["metrics"]
    assert dh_metrics["flux_delta_growth_factor"] > 4.0
    assert dh_metrics["flux_delta_linf_final"] > dh_metrics["flux_delta_linf_first"]
    assert dh_metrics["reconnected_flux_amplification"] > 5.0
    assert dh_metrics["delta_current_linf_peak"] > dh_metrics["delta_current_linf_first"]

    ot_metrics = qa_manifest["visual_qa"]["orszag_tang"]["metrics"]
    assert ot_metrics["current_high_k_peak"] > ot_metrics["current_high_k_first"]
    assert ot_metrics["vorticity_high_k_peak"] > ot_metrics["vorticity_high_k_first"]
    assert ot_metrics["relative_energy_drop"] > 0.1

    turbulence_qa = qa_manifest["visual_qa"]["turbulence"]
    assert (
        turbulence_qa["decaying_mhd_turbulence"]["metrics"]["current_linf_peak"]
        > turbulence_qa["decaying_mhd_turbulence"]["metrics"]["current_linf_first"]
    )
    assert (
        turbulence_qa["forced_turbulent_reconnection"]["metrics"][
            "reconnection_proxy_change"
        ]
        > 1.0
    )


def test_readme_source_media_policy_exceeds_documented_minimum() -> None:
    preset = double_harris_seeded_long_run_presets()["readme_release_media"]

    assert preset["duration_label"] == "readme_release_media"
    assert preset["t_end"] > preset["documented_minimum_t_end"]
    assert "double_harris_seeded_long_run_presets" in (
        ROOT / "examples" / "make_validation_media.py"
    ).read_text(encoding="utf-8")
