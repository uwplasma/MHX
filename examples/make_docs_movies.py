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

Run from the repository root:

    python examples/make_docs_movies.py

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
    "docs/_static/validation/periodic_double_harris_seeded_long_run/figures/"
    "periodic_double_harris_current.gif": "periodic_double_harris_current.mp4",
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


def transcode(source: Path, target: Path) -> None:
    """Transcode one GIF to H.264 mp4 with even dimensions."""
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
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
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


def render_hero_movie(bundle: Path, target: Path) -> None:
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

    writer = imageio.get_writer(target, fps=10, codec="libx264", quality=7)
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in TRANSCODES.items():
        source = ROOT / source_name
        target = OUTPUT_DIR / target_name
        transcode(source, target)
        print(f"{target_name}: {target.stat().st_size / 1024:.0f} KiB")
    island_target = OUTPUT_DIR / "double_harris_island_64.mp4"
    render_island_movie(island_target)
    print(f"double_harris_island_64.mp4: {island_target.stat().st_size / 1024:.0f} KiB")

    for path in sorted(OUTPUT_DIR.glob("*.mp4")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(ROOT)} sha256 {digest}")


if __name__ == "__main__":
    main()
