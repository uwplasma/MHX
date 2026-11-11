"""Animate the turbulent ``j_z`` evolution from a saved ``turbulent_spectrum.npz``.

Writes ``turbulence_jz_evolution.gif`` next to the input file; takes seconds to a
minute. Run ``benchmark_turbulent_spectrum.py`` first, then from the repository
root:

    python examples/turbulence/animate_turbulence.py

To animate a bigger run, point ``NPZ`` at its file.
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps

# --- Input parameters ------------------------------------------------------
NPZ = Path("outputs/examples/turbulence_2d/turbulent_spectrum.npz")  # saved run
FRAME_DURATION_MS = 66  # display time of each GIF frame
PROGRESS_EVERY = 10  # print progress every this many frames

# --- Run -------------------------------------------------------------------
if not NPZ.exists():
    raise SystemExit(f"Could not find {NPZ}; run benchmark_turbulent_spectrum.py first.")

print(f"Loading {NPZ} (this might take a second for large files)...", flush=True)
with np.load(NPZ) as data:
    current_density = data["current_density"]
num_frames = current_density.shape[0]
print(f"Loaded {num_frames} frames of current density.", flush=True)

# 3 standard deviations of the final frame (fully developed turbulence) set a
# consistent color scale across the whole movie.
vmax = max(3.0 * float(np.std(current_density[-1])), 1.0)
vmin = -vmax
cmap = colormaps["RdBu_r"]

frames = []
for i, j_z in enumerate(current_density):
    if i % PROGRESS_EVERY == 0:
        print(f"Generating GIF frames ({i}/{num_frames})...", flush=True)
    normalized = np.clip((j_z.T - vmin) / (vmax - vmin), 0.0, 1.0)
    frames.append((255.0 * cmap(normalized)[..., :3]).astype(np.uint8))
print(f"Generated {num_frames}/{num_frames} frames.", flush=True)

# --- Results ---------------------------------------------------------------
out_path = NPZ.parent / "turbulence_jz_evolution.gif"
imageio.mimsave(out_path, frames, duration=FRAME_DURATION_MS, loop=0, palettesize=64)
print(f"wrote {out_path}")
