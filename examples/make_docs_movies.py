"""Build the mp4 movies that documentation pages embed.

Three lanes:

1. Transcode committed, validated GIF movies to H.264 mp4. The frames and
   claim levels are unchanged. Only the container changes, because mp4 is
   five to ten times smaller than GIF at equal quality.
2. Render the island movie that ``docs/getting_started/first_movie.md``
   teaches, at exactly the settings the page shows.
3. When ``MHX_HERO_BUNDLE`` names a ``double-harris-long-run`` trajectory
   NPZ, render the landing-page hero from that bundle: residual island flux
   with total-flux contours and X/O markers. The bundle itself stays outside
   the repository; only the rendered movie is committed.

Render into staging from the repository root:

    python examples/make_docs_movies.py \
        --readme-source-dir outputs/media-preview/readme \
        --output-dir outputs/media-preview/movies

Requires ffmpeg through ``imageio-ffmpeg`` (installed by the ``dev`` extra).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "_static" / "movies"

# Committed validated sources -> mp4 basename. Claim levels stay with the
# source entries in docs/figures/manifest.toml.
TRANSCODES = {
    "docs/_static/readme/double_harris_reconnection.gif": "double_harris_reconnection.mp4",
    "docs/_static/readme/double_harris_current_sheet.gif": "double_harris_current_sheet.mp4",
    "docs/_static/readme/orszag_tang_current.gif": "orszag_tang_current.mp4",
    "docs/_static/readme/orszag_tang_vorticity.gif": "orszag_tang_vorticity.mp4",
    "docs/_static/readme/orszag_tang_flux.gif": "orszag_tang_flux.mp4",
    "docs/_static/readme/decaying_mhd_turbulence_current.gif": (
        "decaying_mhd_turbulence_current.mp4"
    ),
    "docs/_static/readme/forced_turbulent_reconnection.gif": "forced_turbulent_reconnection.mp4",
    "docs/_static/readme/harris_layer_sweep.gif": "harris_layer_sweep.mp4",
}
# The historical seeded flux movie is deliberately absent: its total-field
# view fails the motion gate, which is the policy working as intended. The
# residual views above carry the same physics visibly.


def ffmpeg_exe() -> str:
    """Return an ffmpeg executable, preferring the bundled one."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def transcode(source: Path, target: Path, *, slow: float = 1.0) -> None:
    """Transcode one GIF to H.264 mp4 with even dimensions.

    ``slow`` stretches playback time by that factor, so a reader can follow
    the dynamics. It changes pacing only, never the frames.
    """
    command = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-movflags",
        "faststart",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        f"setpts={slow}*PTS,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)


def render_island_movie(target: Path) -> None:
    """Render the first_movie.md island example at its documented settings."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import numpy as np

    import mhx

    simulation = mhx.Simulation(
        shape=(64, 64),
        equilibrium=mhx.PeriodicDoubleHarrisEquilibrium(
            width=0.4,
            perturbation_amplitude=4.0e-3,
            perturbation_mode=(2, 1),
        ),
        resistivity=5.0e-3,
        viscosity=5.0e-3,
        dt=2.0e-2,
        t_end=40.0,
        save_every=100,
        verbose=False,
    )
    result = simulation.run()

    flux = np.asarray(result.trajectory.states.psi)
    island = flux - flux.mean(axis=2, keepdims=True)
    times = np.asarray(result.trajectory.times)
    scale = float(np.abs(island).max())

    writer = imageio.get_writer(target, fps=5, codec="libx264", quality=7)
    for index, time in enumerate(times):
        figure, axis = plt.subplots(figsize=(4, 4), dpi=120)
        axis.imshow(
            island[index].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
        )
        axis.contour(flux[index].T, colors="black", linewidths=0.4, levels=12)
        axis.set_title(f"island flux, t = {time:.0f}")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.tight_layout()
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        plt.close(figure)
        # H.264 requires even dimensions.
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        writer.append_data(frame)
    writer.close()


def render_hero_movie(bundle: Path, target: Path, *, fps: int = 6) -> None:
    """Render the landing hero from a validated long-run trajectory NPZ.

    The view zooms on the right current sheet, where the seeded island and
    its X point are visible. The color shows the island flux, the deviation
    of the flux from its y average, on a scale fixed across all frames.
    """
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import numpy as np

    from mhx.diagnostics import detect_flux_critical_points

    data = np.load(bundle)
    flux = np.asarray(data["psi"])
    times = np.asarray(data["times"])
    island = flux - flux.mean(axis=2, keepdims=True)
    scale = float(np.abs(island).max())

    length = 2.0 * np.pi
    x = np.linspace(0.0, length, flux.shape[1], endpoint=False)
    y = np.linspace(0.0, length, flux.shape[2], endpoint=False)
    sheet = 3.0 * length / 4.0
    left = int(np.searchsorted(x, sheet - 1.6))
    right = min(int(np.searchsorted(x, sheet + 1.6)), flux.shape[1] - 1)

    writer = imageio.get_writer(target, fps=fps, codec="libx264", quality=7)
    for index, time in enumerate(times):
        figure, axis = plt.subplots(figsize=(6.4, 6.0), dpi=120)
        axis.imshow(
            island[index, left:right].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            extent=(x[left], x[right], 0.0, length),
            aspect="auto",
        )
        axis.contour(
            x[left:right],
            y,
            flux[index, left:right].T,
            colors="black",
            linewidths=0.5,
            levels=28,
        )
        for point in detect_flux_critical_points(flux[index], max_points=8):
            if not x[left] <= point.position[0] <= x[right]:
                continue
            marker = "x" if point.kind == "X" else "o"
            color = "black" if point.kind == "X" else "white"
            axis.scatter(*point.position, marker=marker, s=80, c=color, zorder=3)
        axis.set_title(f"double-Harris island flux, t = {time:.0f}")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.tight_layout()
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        plt.close(figure)
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        writer.append_data(frame)
    writer.close()


def render_ot3d_movie(views: Path, target: Path, *, fps: int = 5) -> None:
    """Render the 3D Orszag--Tang current movie from saved views.

    ``views`` is the NPZ written by the campaign extraction: times, the
    midplane slice of |j|, and the max-intensity projection of |j| along
    z. Both panels keep one fixed color scale across all frames.
    """
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.load(views)
    times = np.asarray(data["times"])
    midplane = np.asarray(data["midplane"])
    projection = np.asarray(data["projection"])
    slice_scale = float(np.percentile(midplane, 99.5))
    projection_scale = float(np.percentile(projection, 99.5))

    writer = imageio.get_writer(target, fps=fps, codec="libx264", quality=7)
    for index, time in enumerate(times):
        figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), dpi=120)
        for axis, field, scale, title in (
            (axes[0], midplane[index], slice_scale, "|j|, midplane"),
            (axes[1], projection[index], projection_scale, "|j|, max projection"),
        ):
            axis.imshow(
                field.T, origin="lower", cmap="inferno", vmin=0.0, vmax=scale
            )
            axis.set_title(f"{title}, t = {time:.1f}")
            axis.set_xticks([])
            axis.set_yticks([])
        figure.tight_layout()
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        plt.close(figure)
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        writer.append_data(frame)
    writer.close()


def _require_source(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"requested {label} source does not exist: {path}")
    return path


def main(
    *,
    readme_source_dir: Path,
    output_dir: Path,
    hero_bundle: Path | None = None,
    ot3d_views: Path | None = None,
    skip_island: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in TRANSCODES.items():
        source = _require_source(readme_source_dir / Path(source_name).name, target_name)
        target = output_dir / target_name
        transcode(source, target)
        print(f"{target_name}: {target.stat().st_size / 1024:.0f} KiB")

    if not skip_island:
        island_target = output_dir / "double_harris_island_64.mp4"
        render_island_movie(island_target)
        print(f"double_harris_island_64.mp4: {island_target.stat().st_size / 1024:.0f} KiB")

    if hero_bundle is not None:
        hero_bundle = _require_source(hero_bundle, "double-Harris hero")
        hero_target = output_dir / "double_harris_reconnection_256.mp4"
        render_hero_movie(hero_bundle, hero_target)
        print(f"double_harris_reconnection_256.mp4: {hero_target.stat().st_size / 1024:.0f} KiB")

    if ot3d_views is not None:
        ot3d_views = _require_source(ot3d_views, "3-D Orszag-Tang")
        ot3d_target = output_dir / "orszag_tang_3d_current.mp4"
        render_ot3d_movie(ot3d_views, ot3d_target)
        print(f"orszag_tang_3d_current.mp4: {ot3d_target.stat().st_size / 1024:.0f} KiB")

    for path in sorted(output_dir.glob("*.mp4")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path} sha256 {digest}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readme-source-dir",
        type=Path,
        default=Path("outputs/media-preview/readme"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/media-preview/movies"),
    )
    parser.add_argument(
        "--hero-bundle",
        type=Path,
        default=Path(os.environ["MHX_HERO_BUNDLE"])
        if os.environ.get("MHX_HERO_BUNDLE")
        else None,
    )
    parser.add_argument(
        "--ot3d-views",
        type=Path,
        default=Path(os.environ["MHX_OT3D_VIEWS"])
        if os.environ.get("MHX_OT3D_VIEWS")
        else None,
    )
    parser.add_argument("--skip-island", action="store_true")
    args = parser.parse_args()
    main(
        readme_source_dir=args.readme_source_dir,
        output_dir=args.output_dir,
        hero_bundle=args.hero_bundle,
        ot3d_views=args.ot3d_views,
        skip_island=args.skip_island,
    )
