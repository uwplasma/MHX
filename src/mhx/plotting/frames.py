"""Frame selection helpers for movies and snapshot grids."""

from __future__ import annotations

import numpy as np


def sample_indices(frame_count: int, maximum: int) -> list[int]:
    """Select uniformly spaced frame indices, retaining both endpoints.

    Args:
        frame_count: Number of available frames.
        maximum: Largest number of frames to keep.

    Returns:
        All indices when ``frame_count <= maximum``, otherwise ``maximum``
        (or fewer, when rounding repeats an index) increasing indices from
        ``0`` to ``frame_count - 1``.

    Raises:
        ValueError: If ``frame_count`` or ``maximum`` is not positive.
    """
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if frame_count <= maximum:
        return list(range(frame_count))
    return np.unique(np.linspace(0, frame_count - 1, maximum, dtype=int)).tolist()
