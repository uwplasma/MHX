"""Build the mp4 movies that documentation pages embed.

Two lanes:

1. Transcode committed, validated GIF movies to H.264 mp4. The frames and
   claim levels are unchanged. Only the container changes, because mp4 is
   five to ten times smaller than GIF at equal quality.
2. Render the island movie that ``docs/getting_started/first_movie.md``
   teaches, at exactly the settings the page shows.

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
