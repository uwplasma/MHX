from __future__ import annotations

import numpy as np
import pytest

from mhx.diagnostics import (
    CriticalPointTracker,
    FluxCriticalPoint,
    FluxPairRegion,
    PeriodicFluxInterpolator,
    analyze_planar_flux,
    critical_points_by_kind,
    detect_flux_critical_points,
    join_planar_flux,
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
            refine=False,
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
    assert all(1 <= point.newton_iterations <= 12 for point in refined)
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


def test_independent_residual_weak_islands_caps_and_degeneracy() -> None:
    n = 32
    x, y = np.meshgrid(np.arange(n) * 2 * np.pi / n, np.arange(n) * 2 * np.pi / n, indexing="ij")
    psi = np.cos(x - 0.037) + 0.0002 * np.cos(y - 0.081)
    report = detect_flux_critical_points(psi, return_report=True)
    assert len(report.points) == 4 and not report.truncated
    interpolator = PeriodicFluxInterpolator(psi)
    for point in report.points:
        residual = np.linalg.norm(interpolator.evaluate(point.position)[1])
        assert point.residual_norm == pytest.approx(residual, abs=1e-14)
        assert residual < 1e-9 and point.accepted
    expected = (
        (0.037, 0.081),
        (0.037, 0.081 + np.pi),
        (0.037 + np.pi, 0.081),
        (0.037 + np.pi, 0.081 + np.pi),
    )
    assert _mean_nearest_error(report.points, expected, (2 * np.pi,) * 2) < 1e-8
    assert detect_flux_critical_points(psi, max_points=1, return_report=True).truncated
    degenerate = detect_flux_critical_points(np.cos(x), return_report=True)
    assert not degenerate.points
    assert any(point.quality == "degenerate" for point in degenerate.rejected)


def test_periodic_refinement_converges_positions_flux_and_sampling() -> None:
    lengths, lower = (3.2, 5.0), (-1.0, 2.5)
    x0, y0 = -0.963, 2.581
    positions = (
        (x0, y0),
        (x0 + lengths[0] / 2, y0),
        (x0, y0 + lengths[1] / 2),
        (x0 + lengths[0] / 2, y0 + lengths[1] / 2),
    )
    expected_flux = (
        np.exp(0.5) + 0.2 * np.exp(0.4),
        np.exp(-0.5) + 0.2 * np.exp(0.4),
        np.exp(0.5) + 0.2 * np.exp(-0.4),
        np.exp(-0.5) + 0.2 * np.exp(-0.4),
    )
    errors = []
    for n in (8, 16, 32):
        x, y = np.meshgrid(
            lower[0] + np.arange(n) * lengths[0] / n,
            lower[1] + np.arange(n) * lengths[1] / n,
            indexing="ij",
        )
        psi = np.exp(0.5 * np.cos(2 * np.pi * (x - x0) / lengths[0]))
        psi += 0.2 * np.exp(0.4 * np.cos(2 * np.pi * (y - y0) / lengths[1]))
        points = detect_flux_critical_points(psi, lengths=lengths, lower=lower)
        assert len(points) == 4
        flux_error = max(
            abs(
                min(points, key=lambda p, pos=pos: _periodic_distance(p.position, pos, lengths)).psi
                - value
            )
            for pos, value in zip(positions, expected_flux, strict=True)
        )
        errors.append(flux_error + _mean_nearest_error(points, positions, lengths))
    assert errors[1] < 0.01 * errors[0] and errors[2] < 1e-9
    x, y = np.meshgrid(
        np.arange(16) * 2 * np.pi / 16, np.arange(16) * 2 * np.pi / 16, indexing="ij"
    )
    bounds = ((-1.0, 1.0), (0.0, 2 * np.pi))
    region = FluxPairRegion("first", bounds, bounds)
    sampling_errors = []
    for count in (4, 8, 16):
        time = np.linspace(0.0, 1.0, count + 1)
        result = analyze_planar_flux(
            time, [np.cos(x) + 0.2 * np.exp(t) * np.cos(y) for t in time], regions=(region,)
        )
        i = count // 2
        sampling_errors.append(
            abs(result.arrays["reconnection_rate"][i, 0] - 0.4 * np.exp(time[i]))
        )
    assert sampling_errors[0] / sampling_errors[1] > 3.9
    assert sampling_errors[1] / sampling_errors[2] > 3.9


def test_tracking_global_assignment_order_periodic_crossing_and_ambiguity() -> None:
    def point(x):
        return FluxCriticalPoint("X", (0, 0), (x, 0.0), 0.0, 0.0, -1.0, 0.0)

    tracker = CriticalPointTracker((10.0, 10.0), (False, False), max_link_distance=1.5)
    first = tracker.update((point(2.0), point(0.0)))
    second = tracker.update((point(3.0), point(1.2)))
    assert dict((p.position[0], i) for i, p in first) == {0.0: 0, 2.0: 1}
    assert dict((p.position[0], i) for i, p in second) == {1.2: 0, 3.0: 1}
    assert not tracker.events
    restored = CriticalPointTracker.from_state(tracker.state())
    assert restored.update((point(3.1), point(1.3))) == tracker.update((point(1.3), point(3.1)))
    periodic = CriticalPointTracker((10.0, 10.0), max_link_distance=0.5)
    assert periodic.update((point(9.9),))[0][0] == periodic.update((point(0.1),))[0][0]
    ambiguous = CriticalPointTracker((10.0, 10.0), max_link_distance=2.0)
    ambiguous.update((point(0.0), point(2.0)))
    ambiguous.update((point(1.0),))
    assert {event["event"] for event in ambiguous.events} == {"ambiguity", "birth", "disappearance"}


def test_paired_flux_signed_rates_gauge_and_restart() -> None:
    n, lengths = 32, (2 * np.pi,) * 2
    x, y = np.meshgrid(np.arange(n) * lengths[0] / n, np.arange(n) * lengths[1] / n, indexing="ij")
    time = np.array([0.0, 0.13, 0.31, 0.5, 0.7])
    offsets = -0.05 + 0.2 * time
    fields = [
        np.cos(x - offset) + (0.2 + 0.1 * t**2) * np.cos(y)
        for t, offset in zip(time, offsets, strict=True)
    ]
    electric = [
        -0.2 * np.sin(x - offset) - 0.2 * t * np.cos(y)
        for t, offset in zip(time, offsets, strict=True)
    ]
    bounds = ((-1.0, 1.0), (0.0, lengths[1]))
    region = FluxPairRegion("first_sheet", bounds, bounds)
    whole = analyze_planar_flux(
        time, fields, lengths=lengths, regions=(region,), electric_fields=electric
    )
    shifted = analyze_planar_flux(
        time, [field + 27.3 for field in fields], lengths=lengths, regions=(region,)
    )
    assert whole.arrays["flux_valid"].all()
    np.testing.assert_allclose(whole.arrays["paired_flux"][:, 0], 0.4 + 0.2 * time**2, atol=1e-12)
    np.testing.assert_allclose(
        shifted.arrays["paired_flux"], whole.arrays["paired_flux"], atol=1e-12
    )
    np.testing.assert_allclose(
        whole.arrays["reconnection_rate"][1:-1, 0], 0.4 * time[1:-1], atol=1e-10
    )
    np.testing.assert_allclose(whole.arrays["electric_field_rate"][:, 0], 0.4 * time, atol=1e-10)
    assert not whole.arrays["rate_valid"][[0, -1]].any()
    first = analyze_planar_flux(
        time[:3], fields[:3], lengths=lengths, regions=(region,), electric_fields=electric[:3]
    )
    second = analyze_planar_flux(
        time[2:],
        fields[2:],
        lengths=lengths,
        regions=(region,),
        electric_fields=electric[2:],
        initial_tracking_state=first.tracking["snapshots"][-1]["state"],
    )
    joined = join_planar_flux(first, second)
    for key in ("point_frame", "point_track_id", "paired_flux", "reconnection_rate", "rate_valid"):
        np.testing.assert_allclose(joined.arrays[key], whole.arrays[key], atol=1e-10)
    missing = analyze_planar_flux(
        time,
        [fields[0], fields[1], np.cos(x), fields[3], fields[4]],
        lengths=lengths,
        regions=(region,),
    )
    assert not missing.arrays["flux_valid"][2].any()
    assert not missing.arrays["rate_valid"].any()
    assert np.isnan(missing.arrays["reconnection_rate"]).all()
    full_bounds = ((0.0, lengths[0]), (0.0, lengths[1]))
    ambiguous = analyze_planar_flux(
        time,
        fields,
        lengths=lengths,
        regions=(FluxPairRegion("ambiguous", full_bounds, full_bounds),),
    )
    assert (ambiguous.arrays["pair_status"] == "ambiguous_selection").all()


def _mean_nearest_error(
    points,
    expected_positions: tuple[tuple[float, float], ...],
    lengths: tuple[float, float],
) -> float:
    errors = []
    for expected in expected_positions:
        errors.append(
            min(_periodic_distance(point.position, expected, lengths) for point in points)
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
