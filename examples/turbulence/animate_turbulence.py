"""Animate the turbulent ``j_z`` evolution from a saved ``turbulent_spectrum.npz``."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps


def animate_turbulence() -> None:
    root_dir = Path(__file__).resolve().parent.parent.parent

    npz_path = root_dir / "outputs" / "turbulent_spectrum_output" / "turbulent_spectrum.npz"
    if not npz_path.exists():
        print(f"Error: Could not find {npz_path}")
        print("Please ensure your turbulence NPZ file exists first.")
        return

    print(f"Loading {npz_path} (this might take a second for large files)...")
    data = np.load(npz_path)

    current_density = data["current_density"]
    num_frames = current_density.shape[0]
    print(f"Loaded {num_frames} frames of current density.")

    out_path = root_dir / "outputs" / "turbulent_spectrum_output" / "turbulence_jz_evolution.gif"

    # Use 3 standard deviations of the final frame (fully developed turbulence)
    # to set a consistent color scale across the whole movie.
    vmax = 3.0 * float(np.std(current_density[-1]))
    vmax = max(vmax, 1.0)
    vmin = -vmax

    cmap = colormaps["RdBu_r"]

    frames = []
    print(f"Generating GIF frames (0/{num_frames})...")

    for i, j_z in enumerate(current_density):
        if i % 10 == 0:
            sys.stdout.write(f"\rGenerating GIF frames ({i}/{num_frames})...")
            sys.stdout.flush()

        normalized = np.clip((j_z.T - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb_frame = (255.0 * cmap(normalized)[..., :3]).astype(np.uint8)
        frames.append(rgb_frame)

    print(f"\rGenerating GIF frames ({num_frames}/{num_frames})... Done!")
    print(f"Saving GIF to {out_path}...")

    imageio.mimsave(out_path, frames, duration=66, loop=0, palettesize=64)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    animate_turbulence()
