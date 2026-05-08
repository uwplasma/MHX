from __future__ import annotations

from pathlib import Path


def test_readme_media_files_are_present_and_compact() -> None:
    media_dir = Path("docs/_static/readme")
    harris = media_dir / "harris_layer_sweep.gif"
    plasmoid = media_dir / "plasmoid_scaling_schematic.gif"
    assert harris.exists()
    assert plasmoid.exists()
    assert harris.stat().st_size < 200_000
    assert plasmoid.stat().st_size < 300_000
