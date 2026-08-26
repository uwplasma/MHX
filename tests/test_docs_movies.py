"""Quality gates for the mp4 movies that documentation pages embed."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MOVIES = DOCS / "_static" / "movies"

VIDEO_DIRECTIVE_RE = re.compile(r"```{video} (?P<target>\S+)")
GIF_EMBED_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+\.gif)(?:\s+[^)]*)?\)")

MAX_MOVIE_BYTES = 6_000_000

# Mean absolute inter-frame difference, in 8-bit counts, averaged over the
# movie. A visually static movie sits far below this.
MIN_MEAN_MOTION = 0.5

# The Harris layer sweep advances a highlighted parameter marker but keeps the
# reference eigenfunctions fixed, so it is intentionally exempt from this gate.
MOTION_MOVIES = [
    path
    for path in sorted(MOVIES.glob("*.mp4"))
    if path.name != "harris_layer_sweep.mp4"
]


def _docs_pages() -> list[Path]:
    return [
        page
        for page in DOCS.rglob("*.md")
        if "_build" not in page.parts
    ]


def _video_targets() -> dict[Path, list[Path]]:
    targets: dict[Path, list[Path]] = {}
    for page in _docs_pages():
        found = [
            (page.parent / match.group("target")).resolve()
            for match in VIDEO_DIRECTIVE_RE.finditer(page.read_text(encoding="utf-8"))
        ]
        if found:
            targets[page] = found
    return targets


def test_video_directives_point_to_existing_movies() -> None:
    targets = _video_targets()
    assert targets, "expected at least one video embed in the docs"
    missing = [
        f"{page.relative_to(ROOT)} -> {target}"
        for page, page_targets in targets.items()
        for target in page_targets
        if not target.is_file()
    ]
    assert missing == []


def test_movies_stay_small() -> None:
    movies = sorted(MOVIES.glob("*.mp4"))
    assert movies, "expected committed movies under docs/_static/movies"
    oversized = [
        f"{movie.name}: {movie.stat().st_size} bytes"
        for movie in movies
        if movie.stat().st_size > MAX_MOVIE_BYTES
    ]
    assert oversized == []


def test_docs_pages_outside_project_embed_no_gifs() -> None:
    """Docs pages embed mp4; GIF embeds stay in the project records."""
    offenders = []
    for page in _docs_pages():
        if "project" in page.relative_to(DOCS).parts:
            continue
        for match in GIF_EMBED_RE.finditer(page.read_text(encoding="utf-8")):
            offenders.append(f"{page.relative_to(ROOT)} -> {match.group('target')}")
    assert offenders == []


def test_every_committed_movie_is_registered() -> None:
    manifest_text = (DOCS / "figures" / "manifest.toml").read_text(encoding="utf-8")
    unregistered = [
        movie.name
        for movie in sorted(MOVIES.glob("*.mp4"))
        if f"docs/_static/movies/{movie.name}" not in manifest_text
    ]
    assert unregistered == []


@pytest.mark.parametrize(
    "movie", MOTION_MOVIES, ids=lambda path: path.name
)
def test_movies_show_motion(movie: Path) -> None:
    """Reject visually static movies before they reach a reader."""
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")

    reader = imageio.get_reader(movie)
    frames = [np.asarray(frame, dtype=np.float64) for frame in reader]
    reader.close()
    assert len(frames) >= 4, f"{movie.name} has too few frames"

    deltas = [
        float(np.mean(np.abs(second - first)))
        for first, second in zip(frames[:-1], frames[1:], strict=True)
    ]
    mean_motion = float(np.mean(deltas))
    assert mean_motion >= MIN_MEAN_MOTION, (
        f"{movie.name} mean inter-frame motion {mean_motion:.3f} "
        f"is below {MIN_MEAN_MOTION}"
    )
