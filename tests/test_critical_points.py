from __future__ import annotations

import numpy as np
import pytest

from mhx.diagnostics import (
    critical_points_by_kind,
    detect_flux_critical_points,
    track_critical_points,
)


def test_detect_flux_critical_points_classifies_cosine_saddles() -> None:
    n = 64
    lengths = (2.0 * np.pi, 2.0 * np.pi)
    x = np.linspace(0.0, lengths[0], n, endpoint=False)
    y = np.linspace(0.0, lengths[1], n, endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    psi = np.cos(x_mesh) + np.cos(y_mesh)

    points = detect_flux_critical_points(
        psi,
        lengths=lengths,
        max_points=8,
        min_separation=0.5,
    )
    x_points = critical_points_by_kind(points, "X")
    o_points = critical_points_by_kind(points, "O")

    assert len(x_points) == 2
    assert len(o_points) == 2
    assert all(point.gradient_norm < 1.0e-12 for point in points)
    assert {point.kind for point in points} == {"X", "O"}


def test_detect_flux_critical_points_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        detect_flux_critical_points(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError, match="at least four"):
        detect_flux_critical_points(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="positive domain"):
        detect_flux_critical_points(np.zeros((4, 4)), lengths=(1.0, 0.0))
    with pytest.raises(ValueError, match="max_points"):
        detect_flux_critical_points(np.zeros((4, 4)), max_points=0)
    with pytest.raises(ValueError, match="determinant_rtol"):
        detect_flux_critical_points(np.zeros((4, 4)), determinant_rtol=-1.0)
    with pytest.raises(ValueError, match="max_refinement_fraction"):
        detect_flux_critical_points(np.zeros((4, 4)), max_refinement_fraction=0.0)
    with pytest.raises(ValueError, match="min_separation"):
        detect_flux_critical_points(np.zeros((4, 4)), min_separation=-1.0)
    with pytest.raises(ValueError, match="kind"):
        critical_points_by_kind((), "Z")


def test_detect_flux_critical_points_nonperiodic_branch() -> None:
    n = 16
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    psi = x_mesh**2 - y_mesh**2

    points = detect_flux_critical_points(
        psi,
        lengths=(2.0, 2.0),
        periodic=(False, False),
        max_points=4,
        min_separation=0.0,
        determinant_rtol=0.0,
    )

    assert points
    assert all(point.kind == "X" for point in points)
    assert all(point.hessian_determinant < 0.0 for point in points)


def test_refined_flux_critical_points_improve_subcell_locations() -> None:
    n = 64
    lengths = (2.0 * np.pi, 2.0 * np.pi)
    x0 = 0.37 * lengths[0] / n
    y0 = 0.41 * lengths[1] / n
    x = np.linspace(0.0, lengths[0], n, endpoint=False)
    y = np.linspace(0.0, lengths[1], n, endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    psi = np.cos(x_mesh - x0) + np.cos(y_mesh - y0)
    expected_x_points = ((x0 + np.pi, y0), (x0, y0 + np.pi))

    coarse = critical_points_by_kind(
        detect_flux_critical_points(
            psi,
            lengths=lengths,
            max_points=8,
            min_separation=0.5,
        ),
        "X",
    )
    refined = critical_points_by_kind(
        detect_flux_critical_points(
            psi,
            lengths=lengths,
            max_points=8,
            min_separation=0.5,
            refine=True,
        ),
        "X",
    )

    coarse_error = _mean_nearest_error(coarse, expected_x_points, lengths)
    refined_error = _mean_nearest_error(refined, expected_x_points, lengths)
    assert all(point.refined for point in refined)
    assert all(point.newton_iterations == 1 for point in refined)
    assert refined_error < 0.25 * coarse_error
    assert refined_error < 2.0e-4


def test_track_critical_points_preserves_ids_across_small_motion() -> None:
    n = 64
    lengths = (2.0 * np.pi, 2.0 * np.pi)
    x = np.linspace(0.0, lengths[0], n, endpoint=False)
    y = np.linspace(0.0, lengths[1], n, endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    frames = []
    for offset in (0.0, 0.03):
        psi = np.cos(x_mesh - offset) + np.cos(y_mesh)
        frames.append(
            critical_points_by_kind(
                detect_flux_critical_points(
                    psi,
                    lengths=lengths,
                    max_points=8,
                    min_separation=0.5,
                    refine=True,
                ),
                "X",
            )
        )

    tracked = track_critical_points(
        tuple(frames),
        lengths=lengths,
        max_link_distance=0.2,
    )

    first_ids = {track_id for track_id, _ in tracked[: len(frames[0])]}
    second_ids = {track_id for track_id, _ in tracked[len(frames[0]) :]}
    assert first_ids == second_ids
    with pytest.raises(ValueError, match="max_link_distance"):
        track_critical_points(tuple(frames), max_link_distance=-1.0)


def _mean_nearest_error(
    points,
    expected_positions: tuple[tuple[float, float], ...],
    lengths: tuple[float, float],
) -> float:
    errors = []
    for expected in expected_positions:
        errors.append(
            min(
                _periodic_distance(point.position, expected, lengths)
                for point in points
            )
        )
    return float(np.mean(errors))


def _periodic_distance(
    left: tuple[float, float],
    right: tuple[float, float],
    lengths: tuple[float, float],
) -> float:
    dx = min(abs(left[0] - right[0]), lengths[0] - abs(left[0] - right[0]))
    dy = min(abs(left[1] - right[1]), lengths[1] - abs(left[1] - right[1]))
    return float(np.hypot(dx, dy))
