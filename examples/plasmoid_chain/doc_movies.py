"""Render the documentation movies from a saved high-resolution plasmoid run.

This is a manual, developer-step generator: it consumes the ``plasmoids_trajectory.npz``
produced by ``plasmoids.py`` (git-ignored, several GB at production resolution) and
writes the small committed GIF/MP4s that the README and docs embed. It is NOT part of
CI: the committed media is what ships, and CI never needs the big trajectory.

Render choices follow the docs honesty rules: a few sampled frames, the residual
(island) field so the growing plasmoid chain is visible against the static
equilibrium, a small pixel canvas, and a fixed colour scale. Each frame comes
from the full-resolution 1024×1024 run (psi is unpacked once into memory; the
stored current_density is not re-read).

Usage (from the repo root, requires numpy + MHX installed):
    python examples/plasmoid_chain/doc_movies.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Number of frames sampled evenly across the run (keeps committed GIFs small).
DOC_MAX_FRAMES = 18
# Small pixel canvas width for the committed GIFs.
DOC_WIDTH = 320
# Stride used to read a single frame from the in-memory trajectory. We downscale
# a 1024² frame to ~341² before rasterizing so the small GUI canvas stays cheap.
# 1024//3 = 341 → close to DOC_WIDTH (320).
READ_STEP = 3
# Absolute symmetric colourbar limit for the j_z (current) snapshots. The sharp
# early current sheets set a large vmax that would wash out the fainter late-time
# j_z; capping the colourbar at this fixed magnitude keeps the late structure
# visible (values beyond the cutoff saturate/clip). Tune to taste.
JZ_SNAPSHOT_LIMIT = 10.0


def load_plasmoid_trajectory(path: Path) -> dict[str, Any]:
    """Load and validate the plasmoid trajectory.

    The trajectory is a ``np.savez_compressed`` file: compressed arrays cannot be
    memmapped/streamed, and reading any element forces a full uncompressed copy
    into memory. We therefore decompress ``psi`` once and keep it in memory, and
    compute ``j_z = -∇²ψ`` on demand from :data:`psi` instead of reading the
    (equally large) stored ``current_density`` array. This keeps peak memory to
    a single ~2 GB psi copy plus the small sampled frames.
    """
    data = np.load(path)  # not memmap: compressed arrays must be decompressed anyway
    required = ("time", "psi", "nx", "ny", "lx", "ly")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(
            f"Trajectory {path} is missing expected keys: {missing}; "
            f"present keys: {list(data.files)}"
        )
    times = np.asarray(data["time"], dtype=float)
    psi_all = np.asarray(data["psi"], dtype=float)  # (frames, ny, nx) — one unpack
    nx = int(data["nx"])
    ny = int(data["ny"])
    lx = float(data["lx"])
    ly = float(data["ly"])
    step = READ_STEP
    lengths = (lx, ly)

    def psi(index: int) -> np.ndarray:
        return psi_all[index, ::step, ::step]

    def current(index: int) -> np.ndarray:
        # Compute j_z = -∇²ψ on demand, then downsample to the small canvas.
        # This avoids reading the (equally large) stored current_density array.
        from mhx.equations.reduced_mhd import current_density

        jz = np.asarray(current_density(psi_all[index], lengths=lengths), dtype=float)
        return jz[::step, ::step]

    # eager: hold psi in memory once (compressed npz cannot stream) and drop the file handle
    data = None
    print(f"  loaded {path.name}: {times.shape[0]} frames, psi {psi_all.shape}, "
          f"domain=({lx:.1f}, {ly:.1f}), in-memory psi ~{psi_all.nbytes / 1e6:.0f} MB")
    return {
        "time": times,
        "psi": psi,
        "current": current,
        "nx": nx,
        "ny": ny,
        "lx": lx,
        "ly": ly,
        "raw_psi": psi_all,
        "n_frames": times.shape[0],
    }


def _sample_indices(count: int, max_frames: int) -> np.ndarray:
    if count <= max_frames:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, max_frames, dtype=int))


def _downscale(field: np.ndarray, target: int = DOC_WIDTH) -> tuple[np.ndarray, int]:
    """Downscale a 2-D field to a target width for rendering.

    This is a render-only reduction: it operates on the pixels of one frame, not
    on the underlying simulation data (the full 1024² trajectory is untouched).
    Returns the resized array and its new height.
    """
    ny, nx = field.shape
    scale = target / float(nx)
    height = max(int(round(ny * scale)), 1)
    if nx == target and height == ny:
        return field, ny
    rows = np.linspace(0, ny - 1, height, dtype=int)
    cols = np.linspace(0, nx - 1, target, dtype=int)
    return field[np.ix_(rows, cols)], height


def _island(field: np.ndarray) -> np.ndarray:
    """Return the island / residual view: the field minus its y-mean.

    This removes the static double-Harris equilibrium and the slow resistive
    spreading, leaving the reconnecting island/plasmoid signal that grows during
    the run. It is the view the original README movies used.
    """
    return field - field.mean(axis=1, keepdims=True)


def _render_frame(
    field: np.ndarray,
    contour_psi: np.ndarray,
    *,
    lx: float,
    ly: float,
    title: str,
    width: int = DOC_WIDTH,
    label: str = r"$\Delta\psi$",
) -> np.ndarray:
    """Render one residual/island-field frame with a colourbar at a small canvas.

    The residual field (psi minus its y-mean) is used so the plasmoid signal is
    not crushed by the static equilibrium. Limits are a symmetric percentile of
    the frame's values (matching the docs honesty rule), and a colourbar labels
    the scale.
    """
    small, _ = _downscale(_island(field), width)
    small_psi, _ = _downscale(contour_psi, width)
    fig, ax = plt.subplots(figsize=(width / 100.0, width / 100.0), dpi=100)
    vabs = max(float(np.percentile(np.abs(small), 99.0)), np.finfo(float).eps)
    image = ax.imshow(
        small.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vabs,
        vmax=vabs,
        extent=(0.0, lx, 0.0, ly),
        aspect="auto",
    )
    levels = np.linspace(
        float(np.percentile(small_psi, 5.0)),
        float(np.percentile(small_psi, 95.0)),
        18,
    )
    # The contour MUST share the imshow extent/origin, otherwise contour uses
    # raw array indices (0..width) and resets the axis limits, squeezing the
    # imshow image (drawn over the physical extent) into the bottom-left corner.
    ax.contour(
        small_psi.T, levels=levels, colors="black", linewidths=0.35,
        extent=(0.0, lx, 0.0, ly), origin="lower",
    )
    fig.colorbar(image, ax=ax, shrink=0.8, label=label)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return frame


def write_snapshot_contact_sheet(
    traj: dict[str, Any],
    indices: np.ndarray,
    *,
    path: Path,
    field_server: str,
    title: str,
    clip: float = 1.0,
    limit_abs: float | None = None,
) -> None:
    """Write a 4-panel contact sheet of residual/island field snapshots.

    Reproduces the README *_snapshots.png convention from the plasmoid
    trajectory: four frames (first, thirds, last) of the residual field with
    total-flux contours and a shared fixed colour scale.

    ``clip`` shrinks the symmetric colour limit to ``vmax * clip`` (values beyond
    it are saturated/clipped). Use ``clip < 1`` for fields like ``j_z`` whose
    sharp early structure sets a large ``vmax`` and washes out the fainter late
    signal.

    ``limit_abs``, if given, takes precedence: the symmetric colour limit is set
    to this fixed absolute magnitude (values beyond it saturate/clip), regardless
    of the computed ``vmax``.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)
    sample_indices = np.unique(
        np.linspace(0, len(traj["time"]) - 1, 4, dtype=int)
    )
    vmax: float | None = None
    images = []
    for idx in sample_indices:
        field = traj["psi"](idx) if field_server == "psi" else traj["current"](idx)
        res = _island(field)
        vabs = float(np.percentile(np.abs(res), 99.0))
        vmax = vabs if vmax is None else max(vmax, vabs)
        images.append((idx, res))
    limit = limit_abs if limit_abs is not None else (vmax or 1.0) * clip
    if limit <= 0.0 or not np.isfinite(limit):
        limit = 1.0
    for ax, (idx, res) in zip(axes, images, strict=True):
        psi = traj["psi"](idx)
        small_psi, _ = _downscale(psi, DOC_WIDTH)
        image = ax.imshow(
            res.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(0.0, traj["lx"], 0.0, traj["ly"]),
            aspect="auto",
        )
        lv = np.linspace(
            float(np.percentile(small_psi, 5.0)),
            float(np.percentile(small_psi, 95.0)),
            18,
        )
        # Must share the imshow extent/origin (see _render_frame note).
        ax.contour(
            small_psi.T, levels=lv, colors="black", linewidths=0.35,
            extent=(0.0, traj["lx"], 0.0, traj["ly"]), origin="lower",
        )
        ax.set_title(f"t={traj['time'][idx]:.0f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        image, ax=axes, shrink=0.76,
        label=r"$\Delta j_z$" if field_server == "current" else r"$\Delta\psi$",
    )
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("outputs/plasmoids/plasmoids_trajectory.npz"),
        help="Path to the plasmoid trajectory npz (default outputs/plasmoids/...).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("docs/_static/readme"),
        help="Output directory for the GIFs.",
    )
    parser.add_argument(
        "--movie-outdir",
        type=Path,
        default=Path("docs/_static/movies"),
        help="Output directory for the MP4s.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DOC_MAX_FRAMES,
        help=f"Number of frames to sample (default {DOC_MAX_FRAMES}). Lower for a quick test.",
    )
    args = parser.parse_args()

    if not args.trajectory.exists():
        raise FileNotFoundError(
            f"Plasmoid trajectory not found at {args.trajectory}. "
            "Run plasmoids.py (high resolution) and pull plasmoids_trajectory.npz first."
        )

    traj = load_plasmoid_trajectory(args.trajectory)
    indices = _sample_indices(traj["n_frames"], args.max_frames)
    print(f"rendering {len(indices)} frames from {args.trajectory.name} "
          f"(first few may be slow: reading a multi-GB npz off /mnt/c)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.movie_outdir.mkdir(parents=True, exist_ok=True)

    # --- Full-domain reconnection (island psi) and current-sheet (jz) GIFs ----
    reconnection_frames = []
    current_frames = []
    for n, i in enumerate(indices, start=1):
        t0 = time.time()
        psi = traj["psi"](i)
        jz = traj["current"](i)
        t = traj["time"][i]
        reconnection_frames.append(
            _render_frame(
                psi, psi, lx=traj["lx"], ly=traj["ly"],
                title=f"double-Harris plasmoid chain, t={t:.0f}",
                label=r"$\Delta\psi$",
            )
        )
        current_frames.append(
            _render_frame(
                jz, psi, lx=traj["lx"], ly=traj["ly"],
                title=rf"$\Delta j_z$, t={t:.0f}",
                label=r"$\Delta j_z$",
            )
        )
        print(f"  frame {n}/{len(indices)} (idx {i}) done in {time.time() - t0:.1f}s",
              flush=True)
    imageio.mimsave(
        args.outdir / "double_harris_reconnection.gif",
        reconnection_frames, duration=90, loop=0, palettesize=32,
    )
    imageio.mimsave(
        args.outdir / "double_harris_current_sheet.gif",
        current_frames, duration=90, loop=0, palettesize=32,
    )
    print(f"wrote {args.outdir / 'double_harris_reconnection.gif'}")
    print(f"wrote {args.outdir / 'double_harris_current_sheet.gif'}")

    # --- Snapshot contact sheets (reconnection flux, full current, current sheet)
    write_snapshot_contact_sheet(
        traj, indices, path=args.outdir / "double_harris_flux_snapshots.png",
        field_server="psi",
        title="Many-plasmoid double-Harris chain: residual-flux snapshots",
    )
    write_snapshot_contact_sheet(
        traj, indices, path=args.outdir / "double_harris_current_snapshots.png",
        field_server="current",
        title="Many-plasmoid double-Harris chain: residual-current-brightness snapshots",
        limit_abs=JZ_SNAPSHOT_LIMIT,
    )
    write_snapshot_contact_sheet(
        traj, indices, path=args.outdir / "double_harris_current_sheet_snapshots.png",
        field_server="current",
        title="Many-plasmoid double-Harris chain: current-density snapshots",
        limit_abs=JZ_SNAPSHOT_LIMIT,
    )
    print("wrote double_harris_*_snapshots.png")

    print("done")


if __name__ == "__main__":
    main()
