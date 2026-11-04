"""JAX backend of the flux critical-point diagnostics agrees with the NumPy path."""

from __future__ import annotations

import numpy as np
import pytest
from jax.experimental import disable_x64

from mhx.diagnostics.critical_points import (
    PeriodicFluxInterpolator,
    detect_flux_critical_points,
)
from mhx.diagnostics.critical_points_jax import JaxPeriodicFluxInterpolator

TWO_PI = 2.0 * np.pi


def _psi(n=32, lengths=(TWO_PI, TWO_PI), shift=(0.0, 0.0)):
    x = np.arange(n) * lengths[0] / n
    y = np.arange(n) * lengths[1] / n
    xm, ym = np.meshgrid(x, y, indexing="ij")
    return np.cos(xm - shift[0]) + 0.5 * np.cos(ym - shift[1]) + 0.1 * np.cos(xm + ym)


def test_jax_values_match_numpy_interpolator():
    psi = _psi()
    ref = PeriodicFluxInterpolator(psi, (TWO_PI, TWO_PI), (0.0, 0.0))
    dev = JaxPeriodicFluxInterpolator(psi, (TWO_PI, TWO_PI), (0.0, 0.0))
    rng = np.random.default_rng(0)
    positions = rng.uniform(-3.0, 9.0, size=(70, 2))  # more than one batch, outside the box
    values = dev.sample_values(positions)
    assert values.shape == (70,)
    expected = np.array([ref.evaluate(p)[0] for p in positions])
    np.testing.assert_allclose(values, expected, atol=1e-12)
    assert dev.sample_values(np.empty((0, 2))).shape[0] == 0


def test_refine_many_matches_scalar_newton_and_reports_full_derivatives():
    psi = _psi(shift=(0.31, 0.17))
    ref = PeriodicFluxInterpolator(psi)
    dev = JaxPeriodicFluxInterpolator(psi)
    spacing = (TWO_PI / 32,) * 2
    seeds = np.array([[0.3, 0.2], [np.pi + 0.3, 0.1], [0.4, np.pi - 0.2]])
    pos, counts, samples = dev.refine_many(
        seeds, spacing=spacing, fraction=0.75, tolerance=1e-12, max_iterations=20
    )
    assert pos.shape == (3, 2) and counts.shape == (3,) and samples.shape == (3, 6)
    assert (counts >= 1).all()
    for p, row in zip(pos, samples, strict=True):
        value, grad, hess = ref.evaluate(p)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)
        np.testing.assert_allclose(row[0], value, atol=1e-12)
        np.testing.assert_allclose(row[1:3], grad, atol=1e-12)
        np.testing.assert_allclose(row[3:], [hess[0, 0], hess[0, 1], hess[1, 1]], atol=1e-12)


def test_refine_many_empty_input_and_multi_batch():
    dev = JaxPeriodicFluxInterpolator(_psi())
    pos, counts, samples = dev.refine_many(
        np.empty((0, 2)), spacing=(0.2, 0.2), fraction=0.75, tolerance=1e-10, max_iterations=4
    )
    assert pos.shape == (0, 2) and counts.shape == (0,) and samples.shape == (0, 6)
    seeds = np.tile([[0.2, 0.1]], (dev.batch_size + 5, 1))
    pos, counts, _ = dev.refine_many(
        seeds, spacing=(0.2, 0.2), fraction=0.75, tolerance=1e-10, max_iterations=20
    )
    assert pos.shape == (dev.batch_size + 5, 2)
    np.testing.assert_allclose(pos, np.broadcast_to(pos[0], pos.shape), atol=1e-12)
    assert (counts == counts[0]).all()


def test_requires_x64():
    psi = _psi()
    with disable_x64(), pytest.raises(ValueError, match="enable_x64"):
        JaxPeriodicFluxInterpolator(psi)


def _sorted(points):
    return sorted(points, key=lambda p: (p.kind, round(p.position[0], 6), round(p.position[1], 6)))


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.31, 0.17)])
def test_backends_find_same_points(shift):
    psi = _psi(shift=shift)
    numpy_report = detect_flux_critical_points(psi, return_report=True)
    jax_report = detect_flux_critical_points(psi, backend="jax", return_report=True)
    assert jax_report.method == numpy_report.method == "fourier_newton"
    assert len(numpy_report.points) == len(jax_report.points) == 4
    for a, b in zip(_sorted(numpy_report.points), _sorted(jax_report.points), strict=True):
        assert a.kind == b.kind
        np.testing.assert_allclose(a.position, b.position, atol=1e-8)
        assert a.psi == pytest.approx(b.psi, abs=1e-10)
        assert a.hessian_determinant == pytest.approx(b.hessian_determinant, abs=1e-8)
        assert a.hessian_trace == pytest.approx(b.hessian_trace, abs=1e-8)
        assert b.quality == "accepted" and b.refined


def test_jax_backend_finds_known_cosine_x_and_o_points():
    n = 32
    x, y = np.meshgrid(np.arange(n) * TWO_PI / n, np.arange(n) * TWO_PI / n, indexing="ij")
    points = detect_flux_critical_points(np.cos(x) + np.cos(y), backend="jax")
    o = sorted((p for p in points if p.kind == "O"), key=lambda p: p.psi)
    assert [round(p.psi) for p in o] == [-2, 2]
    assert np.allclose(o[1].position, (0.0, 0.0), atol=1e-8)
    xs = [p for p in points if p.kind == "X"]
    assert len(xs) == 2
    assert all(p.psi == pytest.approx(0.0, abs=1e-10) for p in xs)
    assert all(p.residual_norm < 1e-9 for p in points)


def test_jax_backend_flat_field_is_degenerate_and_nonuniform_domain():
    report = detect_flux_critical_points(np.full((8, 8), 1.0), backend="jax", return_report=True)
    assert not report.points
    assert [p.quality for p in report.rejected] == ["degenerate"]
    lengths, lower = (3.0, 5.0), (-1.0, 2.0)
    n = 32
    x, y = np.meshgrid(
        lower[0] + np.arange(n) * lengths[0] / n,
        lower[1] + np.arange(n) * lengths[1] / n,
        indexing="ij",
    )
    psi = np.cos(TWO_PI * (x + 0.5) / lengths[0]) + np.cos(TWO_PI * (y - 2.5) / lengths[1])
    a = detect_flux_critical_points(psi, lengths=lengths, lower=lower)
    b = detect_flux_critical_points(psi, lengths=lengths, lower=lower, backend="jax")
    assert len(a) == len(b) == 4
    for pa, pb in zip(_sorted(a), _sorted(b), strict=True):
        np.testing.assert_allclose(pa.position, pb.position, atol=1e-8)
