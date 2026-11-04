"""Tests for tracked-point and paired-flux plots (Agg backend; files, not pixels)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mhx.diagnostics import FluxPairRegion, analyze_planar_flux  # noqa: E402
from mhx.plotting.planar_flux import plot_planar_flux, plot_tracked_points  # noqa: E402

N = 32
LENGTHS = (2 * np.pi, 2 * np.pi)
TIME = np.array([0.0, 0.2, 0.4, 0.6])
BOUNDS = ((-1.0, 1.0), (0.0, LENGTHS[1]))


def _fields():
    axis = np.arange(N) * LENGTHS[0] / N
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return x, y, [np.cos(x) + (0.2 + 0.1 * t) * np.cos(y) for t in TIME]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _arrays(kinds, positions, ids, time=0.0, accepted=None):
    n = len(kinds)
    return {
        "point_time": np.full(n, time),
        "point_accepted": np.ones(n, bool) if accepted is None else np.asarray(accepted),
        "point_kind": np.asarray(kinds, dtype="U1"),
        "point_position": np.asarray(positions, dtype=float).reshape(-1, 2),
        "point_track_id": np.asarray(ids),
    }


def test_plot_planar_flux_with_pairs_writes_nonempty_png(tmp_path) -> None:
    _, _, fields = _fields()
    region = FluxPairRegion("sheet", BOUNDS, BOUNDS)
    x, y, _ = _fields()
    electric = [0.1 * np.cos(y) + 0.0 * x for _ in TIME]
    result = analyze_planar_flux(
        TIME, fields, lengths=LENGTHS, regions=(region,), electric_fields=electric
    )
    path = tmp_path / "nested" / "planar.png"
    returned = plot_planar_flux(result, path, psi_final=fields[-1], lengths=LENGTHS)
    assert returned == path
    assert path.stat().st_size > 1000
    assert not plt.get_fignums()  # the figure is closed after saving


def test_plot_planar_flux_without_pairs_or_field_and_with_constant_field(tmp_path) -> None:
    _, _, fields = _fields()
    result = analyze_planar_flux(TIME, fields, lengths=LENGTHS)
    bare = plot_planar_flux(result, str(tmp_path / "bare.png"), lengths=LENGTHS)
    assert bare.exists() and bare.stat().st_size > 0
    constant = plot_planar_flux(
        result, tmp_path / "const.png", psi_final=np.ones((N, N)), lengths=LENGTHS, lower=(0.0, 0.0)
    )
    assert constant.exists() and constant.stat().st_size > 0


def test_plot_tracked_points_marks_only_accepted_points_at_selected_time() -> None:
    arrays = _arrays(
        ["X", "O", "X", "O"],
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        [10, 11, 12, 13],
        accepted=[True, True, False, True],
    )
    arrays["point_time"][3] = 5.0  # a different frame
    figure, axis = plt.subplots()
    axis.set_xlim(0, 6)
    axis.set_ylim(0, 6)
    figure.canvas.draw()
    plot_tracked_points(axis, arrays, 0.0)
    labels = sorted(text.get_text() for text in axis.texts)
    assert labels == ["10", "11"]
    sizes = sorted(len(c.get_offsets()) for c in axis.collections)
    assert sizes == [1, 1]


def test_plot_tracked_points_label_placement_and_overflow() -> None:
    # Six coincident points: only four non-overlapping label slots exist.
    arrays = _arrays(["X"] * 6, [[3.0, 3.0]] * 6, list(range(6)))
    figure, axis = plt.subplots()
    axis.set_xlim(0, 6)
    axis.set_ylim(0, 6)
    figure.canvas.draw()
    plot_tracked_points(axis, arrays, 0.0)
    assert len(axis.texts) == 4
    # Labels flip below the marker near the top edge and above it elsewhere.
    figure, axis = plt.subplots()
    axis.set_xlim(0, 6)
    axis.set_ylim(0, 6)
    figure.canvas.draw()
    plot_tracked_points(axis, _arrays(["O", "O"], [[1.0, 5.9], [1.0, 0.5]], [1, 2]), 0.0)
    by_text = {t.get_text(): t.get_va() for t in axis.texts}
    assert by_text == {"1": "top", "2": "bottom"}


def test_plot_tracked_points_wraps_into_image_extent() -> None:
    arrays = _arrays(["X"], [[-1.0, 7.0]], [3])
    figure, axis = plt.subplots()
    axis.imshow(np.zeros((8, 8)), origin="lower", extent=(0.0, 2 * np.pi, 0.0, 2 * np.pi))
    figure.canvas.draw()
    plot_tracked_points(axis, arrays, 0.0, lengths=np.asarray(LENGTHS), color="red")
    (collection,) = axis.collections
    np.testing.assert_allclose(
        collection.get_offsets()[0], [2 * np.pi - 1.0, 7.0 - 2 * np.pi], atol=1e-12
    )
