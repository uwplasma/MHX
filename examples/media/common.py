"""Shared paths, encoders, and provenance helpers for documentation media."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "outputs" / "media-campaign"
PREVIEW_ROOT = ROOT / "outputs" / "media-preview"
README_PREVIEW = PREVIEW_ROOT / "readme"
MOVIE_PREVIEW = PREVIEW_ROOT / "movies"
POSTER_PREVIEW = PREVIEW_ROOT / "posters"


def source_dir(case: str, preset: str) -> Path:
    """Return the standard source-bundle directory for one case."""
    return SOURCE_ROOT / preset / case


def require_source(path: Path, label: str) -> Path:
    """Return an existing source path or fail with an actionable message."""
    if not path.is_file():
        raise FileNotFoundError(f"missing {label} source: {path}")
    return path


def render_stem(stem: str, source_metadata: dict[str, Any]) -> str:
    """Keep preview outputs distinct from canonical final-campaign names."""
    if source_metadata.get("preset") == "preview":
        return f"{stem}_preview"
    return stem


@cache
def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_indices(frame_count: int, maximum: int) -> list[int]:
    """Select uniformly spaced frame indices, retaining both endpoints."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if frame_count <= maximum:
        return list(range(frame_count))
    import numpy as np

    return np.unique(np.linspace(0, frame_count - 1, maximum, dtype=int)).tolist()


def figure_frame(figure: Any) -> Any:
    """Convert a Matplotlib figure canvas to an RGB uint8 array."""
    import numpy as np

    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def encode_frames(
    frames: Sequence[Any],
    *,
    stem: str,
    source: Path,
    source_metadata: dict[str, Any],
    times: Sequence[float],
    write_gif: bool = True,
    write_mp4: bool = True,
    fps: int = 8,
    poster_index: int | None = None,
) -> dict[str, Path]:
    """Encode one plotted frame sequence for README and documentation use."""
    if not frames:
        raise ValueError("cannot encode an empty frame sequence")
    if len(frames) != len(times):
        raise ValueError("frames and times must have equal length")
    if poster_index is not None and not 0 <= poster_index < len(frames):
        raise ValueError("poster_index must select an encoded frame")

    # Preview renders must never replace canonical final-campaign filenames.
    stem = render_stem(stem, source_metadata)

    import imageio.v2 as imageio
    import numpy as np

    README_PREVIEW.mkdir(parents=True, exist_ok=True)
    MOVIE_PREVIEW.mkdir(parents=True, exist_ok=True)
    POSTER_PREVIEW.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if write_gif:
        gif_path = README_PREVIEW / f"{stem}.gif"
        imageio.mimsave(
            gif_path,
            list(frames),
            duration=max(1, round(1000 / fps)),
            loop=0,
            palettesize=64,
        )
        outputs["readme_gif"] = gif_path

    if write_mp4:
        mp4_path = MOVIE_PREVIEW / f"{stem}.mp4"
        writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=7)
        try:
            for frame in frames:
                height, width = frame.shape[:2]
                pad_height = (-height) % 16
                pad_width = (-width) % 16
                encoded = np.pad(
                    frame,
                    ((0, pad_height), (0, pad_width), (0, 0)),
                    mode="edge",
                )
                writer.append_data(encoded)
        finally:
            writer.close()
        outputs["docs_movie"] = mp4_path

    poster_path = POSTER_PREVIEW / f"{stem}.png"
    selected_poster = len(frames) // 2 if poster_index is None else poster_index
    imageio.imwrite(poster_path, frames[selected_poster])
    outputs["poster"] = poster_path
    write_render_record(
        stem=stem,
        source=source,
        source_metadata=source_metadata,
        times=times,
        outputs=outputs,
    )
    return outputs


def write_render_record(
    *,
    stem: str,
    source: Path,
    source_metadata: dict[str, Any],
    times: Sequence[float],
    outputs: dict[str, Path],
) -> Path:
    """Write per-case provenance for a staged render."""
    record_dir = PREVIEW_ROOT / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "mhx.media.render.v1",
        "id": stem,
        "source": str(source),
        "source_sha256": sha256(source),
        "source_metadata": source_metadata,
        "frame_count": len(times),
        "times": [float(value) for value in times],
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    path = record_dir / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    records = [
        json.loads(record_path.read_text(encoding="utf-8"))
        for record_path in sorted(record_dir.glob("*.json"))
    ]
    aggregate = {
        "schema": "mhx.media.build.v1",
        "renders": records,
    }
    (PREVIEW_ROOT / "media_build.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def write_source_metadata(outdir: Path, metadata: dict[str, Any]) -> Path:
    """Write normalized metadata beside a simulation source bundle."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "media_source.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_source_metadata(source: Path) -> dict[str, Any]:
    """Load optional normalized metadata beside a source bundle."""
    path = source.parent / "media_source.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_array_shape(
    actual: Iterable[int], expected: Iterable[int], *, label: str
) -> None:
    """Reject source bundles whose stored shape disagrees with provenance."""
    actual_shape = tuple(int(value) for value in actual)
    expected_shape = tuple(int(value) for value in expected)
    if actual_shape != expected_shape:
        raise ValueError(f"{label} shape {actual_shape} != recorded shape {expected_shape}")
