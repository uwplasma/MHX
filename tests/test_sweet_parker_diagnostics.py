"""Sweet--Parker sheet measurement, prediction and steady-window selection."""

from __future__ import annotations

import numpy as np
import pytest

from mhx.diagnostics import sweet_parker as sp

TWO_PI = 2.0 * np.pi
EPS = 0.2
FLOW = 0.5
ETA = 1.0e-2
N = 32


def _sheet_fields():
    """Return ``psi = cos x (1 + eps cos y)`` and ``omega = -2 phi`` for ``phi = A sin x sin y``.

    The centreline is ``x = 0``; the X-point is at ``(0, π)`` with ``j_X = 1 - 2 eps`` and
    Hessian ``diag(-(1 - eps), eps)``. The flow is ``v_y = A cos x sin y``.
    """
    x = np.arange(N) * TWO_PI / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    psi = np.cos(xx) * (1.0 + EPS * np.cos(yy))
    omega = -2.0 * FLOW * np.sin(xx) * np.sin(yy)
    return psi, omega


@pytest.fixture(scope="module")
def measurement():
    psi, omega = _sheet_fields()
    return sp.measure_sweet_parker_sheet(
        psi, omega, lengths=(TWO_PI, TWO_PI), resistivity=ETA, sheet_x=0.0, sheet_half_window=1.5
    )


def test_xpoint_location_current_and_rate(measurement) -> None:
    dx = TWO_PI / N
    assert measurement.x_point == pytest.approx((0.0, 16 * dx))
    assert measurement.current_x == pytest.approx(1.0 - 2.0 * EPS, rel=1e-10)
    assert measurement.reconnection_rate == pytest.approx(ETA * (1.0 - 2.0 * EPS), rel=1e-10)
    assert measurement.sheet_x_point_count == 1


def test_sheet_thickness_matches_half_maximum_of_cosine_profile(measurement) -> None:
    # j(x, y_X) = j_X cos x falls to half of j_X at x = π/3.
    assert measurement.delta == pytest.approx(np.pi / 3.0, rel=0.06)
    assert measurement.delta_over_dx == pytest.approx(measurement.delta / (TWO_PI / N))
    # The current has its minimum, not a maximum, on the centreline of this construction, so
    # no half-level crossing exists along the sheet.
    assert np.isnan(measurement.length_current)


def test_outflow_length_and_speed(measurement) -> None:
    # v_y = A sin(y) on the centreline peaks a quarter period (π/2) from the X-point.
    assert measurement.length == pytest.approx(np.pi / 2.0, rel=1e-10)
    assert measurement.v_outflow == pytest.approx(FLOW, rel=1e-10)
    assert 0.0 < measurement.v_inflow < FLOW
    assert 0.0 < measurement.b_upstream < 1.0


def test_hessian_geometry(measurement) -> None:
    expected_angle = 2.0 * np.degrees(np.arctan(np.sqrt(EPS / (1.0 - EPS))))
    assert measurement.opening_angle_deg == pytest.approx(expected_angle, rel=1e-8)
    assert measurement.tilt_deg == pytest.approx(0.0, abs=1e-6)


def test_island_width_is_extent_inside_separatrix(measurement) -> None:
    # psi(x, O) = (1 + eps) cos x exceeds psi_X = 1 - eps for cos x > (1-eps)/(1+eps).
    half = np.arccos((1.0 - EPS) / (1.0 + EPS))
    assert measurement.island_width == pytest.approx(2.0 * half, rel=0.06)


def test_derived_quantities_and_dict(measurement) -> None:
    m = measurement
    assert m.lundquist == pytest.approx(m.length * m.b_upstream / ETA, rel=1e-10)
    assert m.normalized_rate == pytest.approx(m.reconnection_rate / m.b_upstream**2)
    flat = m.as_dict()
    assert "x_point" not in flat
    assert flat["x_point_x"] == m.x_point[0] and flat["x_point_y"] == m.x_point[1]
    assert flat["lundquist"] == m.lundquist
    assert flat["normalized_rate"] == m.normalized_rate
    assert flat["delta"] == m.delta


def test_default_window_too_narrow_gives_nan_upstream() -> None:
    psi, omega = _sheet_fields()
    m = sp.measure_sweet_parker_sheet(
        psi, omega, lengths=(TWO_PI, TWO_PI), resistivity=ETA, sheet_x=0.0
    )
    assert np.isnan(m.delta)
    assert np.isnan(m.b_upstream) and np.isnan(m.v_inflow)
    assert m.current_x == pytest.approx(1.0 - 2.0 * EPS, rel=1e-10)


def test_measure_rejects_bad_shapes() -> None:
    psi, omega = _sheet_fields()
    with pytest.raises(ValueError, match="two-dimensional"):
        sp.measure_sweet_parker_sheet(
            psi, omega[:-1], lengths=(TWO_PI, TWO_PI), resistivity=ETA, sheet_x=0.0
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        sp.measure_sweet_parker_sheet(
            psi[0], omega[0], lengths=(TWO_PI, TWO_PI), resistivity=ETA, sheet_x=0.0
        )


def test_sweet_parker_prediction_scalings() -> None:
    out = sp.sweet_parker_prediction(resistivity=1e-3, viscosity=1e-3, length=2.0, b_upstream=0.5)
    s = 2.0 * 0.5 / 1e-3
    assert out["lundquist"] == pytest.approx(s)
    assert out["magnetic_prandtl"] == pytest.approx(1.0)
    assert out["delta"] == pytest.approx(2.0 * 2**0.25 / np.sqrt(s))
    assert out["v_inflow"] == pytest.approx(0.5 * 2**-0.25 / np.sqrt(s))
    assert out["v_outflow"] == pytest.approx(0.5 * 2**-0.5)
    assert out["normalized_rate"] == pytest.approx(2**-0.25 / np.sqrt(s))
    # Mass conservation: delta * v_out = L * v_in.
    assert out["delta"] * out["v_outflow"] == pytest.approx(2.0 * out["v_inflow"])
    inviscid = sp.sweet_parker_prediction(
        resistivity=1e-3, viscosity=0.0, length=2.0, b_upstream=0.5
    )
    assert inviscid["delta"] == pytest.approx(2.0 / np.sqrt(s))
    assert inviscid["v_outflow"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resistivity": 0.0, "viscosity": 0.0, "length": 1.0, "b_upstream": 1.0},
        {"resistivity": 1.0, "viscosity": -1.0, "length": 1.0, "b_upstream": 1.0},
        {"resistivity": 1.0, "viscosity": 0.0, "length": 0.0, "b_upstream": 1.0},
        {"resistivity": 1.0, "viscosity": 0.0, "length": 1.0, "b_upstream": 0.0},
    ],
)
def test_sweet_parker_prediction_rejects_bad_input(kwargs) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        sp.sweet_parker_prediction(**kwargs)


def test_steady_window_skips_transient_and_picks_flat_stretch() -> None:
    times = np.arange(8.0)
    rate = np.array([1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    window = sp.select_steady_window(times, rate, min_duration=3.0)
    assert window == (2.0, 5.0, 0.0)


def test_steady_window_none_when_too_short_or_too_noisy() -> None:
    times = np.arange(5.0)
    assert sp.select_steady_window(times, np.ones(5), min_duration=10.0) is None
    noisy = np.array([1.0, 3.0, 1.0, 3.0, 1.0])
    assert sp.select_steady_window(times, noisy, min_duration=2.0, max_cv=0.1) is None


def test_steady_window_rejects_drifting_constraint() -> None:
    times = np.arange(10.0)
    rate = np.ones(10)
    ramp = 1.0 + 0.5 * times
    assert sp.select_steady_window(times, rate, min_duration=3.0, constraints=(ramp,)) is None
    ok = sp.select_steady_window(
        times, rate, min_duration=3.0, constraints=(ramp,), max_drift=10.0
    )
    assert ok == (0.0, 3.0, 0.0)


def test_steady_window_rejects_nonfinite_and_zero_mean_series() -> None:
    times = np.arange(6.0)
    rate = np.ones(6)
    with_nan = np.ones(6)
    with_nan[2] = np.nan
    # Windows containing the NaN sample are skipped; the first clean window starts at index 3.
    assert sp.select_steady_window(times, with_nan, min_duration=2.0) == (3.0, 5.0, 0.0)
    zero_mean = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    assert sp.select_steady_window(times, zero_mean, min_duration=1.0) is None
    assert sp.select_steady_window(times, rate, min_duration=2.0, constraints=(zero_mean,)) is None
    clean = sp.select_steady_window(times, rate, min_duration=2.0, constraints=(with_nan,))
    assert clean == (3.0, 5.0, 0.0)


def test_steady_window_input_validation() -> None:
    times = np.arange(5.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        sp.select_steady_window(times, np.ones(4), min_duration=1.0)
    with pytest.raises(ValueError, match="constraints must match"):
        sp.select_steady_window(times, np.ones(5), min_duration=1.0, constraints=(np.ones(3),))
    with pytest.raises(ValueError, match="min_duration"):
        sp.select_steady_window(times, np.ones(5), min_duration=0.0)


def test_count_prominent_maxima() -> None:
    y = np.linspace(0.0, TWO_PI, 64, endpoint=False)
    assert sp._count_prominent_maxima(np.cos(y)) == 1
    assert sp._count_prominent_maxima(np.cos(2.0 * y)) == 2
    # A tiny wiggle on the flank of a single hump is ignored.
    wiggle = np.cos(y) + 1.0e-6 * np.sin(20.0 * y)
    assert sp._count_prominent_maxima(wiggle) == 1
    assert sp._count_prominent_maxima(np.ones(8)) == 1


def test_level_crossing_and_half_level_helpers() -> None:
    negative = np.array([-2.0, -1.0, 1.0, 3.0])
    assert sp._level_crossing_distance(negative, 0, 1, 3) == pytest.approx(1.5)
    assert np.isnan(sp._level_crossing_distance(negative, 0, 1, 1))
    profile = np.array([4.0, 3.0, 1.0, 0.0])
    # Drops below 2 between index 1 (3.0) and index 2 (1.0): halfway.
    assert sp._half_level_distance(profile, 0, 1, 2.0, 3) == pytest.approx(1.5)
    assert np.isnan(sp._half_level_distance(profile, 0, 1, -1.0, 3))
    assert sp._argmax_distance(np.array([0.0, 1.0, 5.0, 2.0]), 0, 1, 3) == 2
    assert sp._periodic_offset(3.5 * np.pi, TWO_PI) == pytest.approx(0.5 * np.pi)


def test_count_prominent_maxima_merges_close_extrema_pair() -> None:
    # A +/-5e-4 dimple on the top of the hump is below the 1e-3 * ptp tolerance.
    values = np.array([0.0, 1.0, 2.0, 3.0, 2.9995, 3.001, 2.0, 1.0])
    assert sp._count_prominent_maxima(values) == 1


def test_steady_window_of_empty_series_is_none() -> None:
    assert sp.select_steady_window(np.array([]), np.array([]), min_duration=1.0) is None
