from __future__ import annotations

import numpy as np
import pytest

from mhx.diagnostics import critical_points_by_kind, detect_flux_critical_points


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
