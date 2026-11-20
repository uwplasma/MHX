"""Extra tests for flux critical-point validation, resolved selection, and tracking."""

from __future__ import annotations

import numpy as np
import pytest

from mhx.diagnostics.critical_points import (
    CriticalPointTracker,
    FluxCriticalPoint,
    PeriodicFluxInterpolator,
    _minimum_assignment,
    _periodic_excursion_labels,
    _saddle_branches,
    detect_flux_critical_points,
    select_resolved_flux_points,
    track_critical_points,
)

TWO_PI = 2.0 * np.pi


def _grid(n, lengths=(TWO_PI, TWO_PI)):
    x = np.arange(n) * lengths[0] / n
    y = np.arange(n) * lengths[1] / n
    return np.meshgrid(x, y, indexing="ij")


def _point(kind, x, y=0.0, trace=0.0, accepted=True):
    det = 1.0 if kind == "O" else -1.0
    return FluxCriticalPoint(kind, (0, 0), (x, y), 0.0, 0.0, det, trace, accepted=accepted)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "cuda"}, "backend must be"),
        ({"backend": "jax", "refine": False}, "jax backend requires"),
        ({"backend": "jax", "periodic": (True, False)}, "jax backend requires"),
        ({"residual_atol": -1.0}, "residual tolerances"),
        ({"residual_rtol": -1.0}, "residual tolerances"),
        ({"max_iterations": 0}, "residual tolerances"),
        ({"lower": (0.0, 0.0, 0.0)}, "finite psi"),
        ({"periodic": (True,)}, "finite psi"),
        ({"lengths": (1.0, np.inf)}, "domain coordinates"),
        ({"lower": (np.nan, 0.0)}, "domain coordinates"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"seed_scale_cells": -1.0}, "selection scales"),
        ({"min_flux_contrast_rtol": np.nan}, "selection scales"),
        ({"min_island_cells": 0.5}, "selection scales"),
        ({"seed_scale_cells": 1.0, "periodic": (True, False)}, "periodic field"),
        ({"min_flux_contrast_rtol": 0.1, "periodic": (False, True)}, "periodic field"),
    ],
)
def test_detect_rejects_bad_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        detect_flux_critical_points(np.zeros((8, 8)), **kwargs)


def test_detect_rejects_nonfinite_psi():
    psi = np.zeros((8, 8))
    psi[2, 3] = np.nan
    with pytest.raises(ValueError, match="finite psi"):
        detect_flux_critical_points(psi)


def test_flat_field_reports_single_degenerate_candidate():
    report = detect_flux_critical_points(np.full((8, 8), 3.0), return_report=True)
    assert report.candidate_count == 1
    assert not report.points
    assert [p.quality for p in report.rejected] == ["degenerate"]
    assert report.method == "fourier_newton"


def test_max_candidates_caps_and_flags_truncation():
    x, y = _grid(32)
    psi = np.cos(x) + np.cos(y)
    report = detect_flux_critical_points(psi, max_candidates=1, return_report=True)
    assert report.truncated
    assert len(report.points) + len(report.rejected) == 1


def test_cosine_points_have_known_positions_and_flux_and_wrap_to_lower_edge():
    x, y = _grid(32)
    psi = np.cos(x) + np.cos(y)
    points = detect_flux_critical_points(psi)
    for p in points:
        assert 0.0 <= p.position[0] < TWO_PI
        assert 0.0 <= p.position[1] < TWO_PI
    o_points = sorted((p for p in points if p.kind == "O"), key=lambda p: p.psi)
    assert o_points[0].psi == pytest.approx(-2.0, abs=1e-10)
    assert o_points[-1].psi == pytest.approx(2.0, abs=1e-10)
    assert np.allclose(o_points[-1].position, (0.0, 0.0), atol=1e-8)
    assert np.allclose(o_points[0].position, (np.pi, np.pi), atol=1e-8)
    xs = [p for p in points if p.kind == "X"]
    assert len(xs) == 2
    assert all(p.psi == pytest.approx(0.0, abs=1e-10) for p in xs)
    assert all(p.hessian_determinant < 0 for p in xs)
    assert all(p.hessian_trace == pytest.approx(0.0, abs=1e-10) for p in xs)


def test_mixed_periodicity_uses_interpolated_newton():
    n = 24
    x = np.arange(n) * TWO_PI / n
    y = np.linspace(-1.0, 1.0, n)
    xm, ym = np.meshgrid(x, y, indexing="ij")
    psi = np.cos(xm) - ym**2
    report = detect_flux_critical_points(
        psi,
        lengths=(TWO_PI, 2.0),
        lower=(0.0, -1.0),
        periodic=(True, False),
        return_report=True,
    )
    assert report.method == "interpolated_newton"
    assert report.points
    # cos x - y^2 has a maximum (O) at (0, 0) and a saddle (X) at (pi, 0)
    assert any(p.kind == "O" for p in report.points)


def test_nonperiodic_unrefined_marks_points_unrefined():
    n = 16
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    psi = x**2 - y**2
    report = detect_flux_critical_points(
        psi,
        lengths=(2.0, 2.0),
        lower=(-1.0, -1.0),
        periodic=(False, False),
        refine=False,
        min_separation=0.0,
        determinant_rtol=0.0,
        return_report=True,
    )
    assert report.points
    assert all(p.quality == "unrefined" and not p.refined for p in report.points)
    assert all(p.newton_iterations == 0 for p in report.points)


def test_nonperiodic_refined_saddle_found_at_origin():
    n = 33
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    psi = x**2 - y**2 + 0.3 * x * y
    points = detect_flux_critical_points(
        psi, lengths=(2.0, 2.0), lower=(-1.0, -1.0), periodic=(False, False)
    )
    assert any(p.kind == "X" and np.allclose(p.position, (0.0, 0.0), atol=1e-6) for p in points)


def test_singular_hessian_roots_are_degenerate():
    x, _ = _grid(16)
    report = detect_flux_critical_points(np.cos(x), return_report=True)
    assert report.rejected
    assert all(p.quality == "degenerate" for p in report.rejected)


def test_poor_newton_budget_rejects_roots():
    x, y = _grid(32)
    psi = np.cos(x) + np.cos(y) + 0.3 * np.sin(2 * x + y)
    report = detect_flux_critical_points(
        psi, max_iterations=1, residual_atol=0.0, residual_rtol=0.0, return_report=True
    )
    qualities = {p.quality for p in report.rejected}
    assert qualities & {"residual_rejected", "not_converged"}
    assert all(not p.accepted for p in report.rejected)


def test_duplicate_roots_are_rejected_by_min_separation():
    x, y = _grid(32)
    psi = np.cos(x) + np.cos(y)
    report = detect_flux_critical_points(psi, min_separation=20.0, return_report=True)
    assert len(report.points) == 2
    assert {p.kind for p in report.points} == {"X", "O"}
    assert any(p.quality == "duplicate_root" for p in report.rejected)


def test_interpolator_matches_field_and_derivatives():
    n, (lx, ly) = 16, (3.0, 5.0)
    x, y = _grid(n, (lx, ly))
    kx, ky = TWO_PI / lx, TWO_PI / ly
    psi = np.sin(kx * x) * np.cos(2 * ky * y) + 0.7
    interp = PeriodicFluxInterpolator(psi, (lx, ly), (1.0, -2.0))
    ax, ay = 0.37, 1.91
    value, grad, hess = interp.evaluate((1.0 + ax, -2.0 + ay))
    assert value == pytest.approx(np.sin(kx * ax) * np.cos(2 * ky * ay) + 0.7)
    assert grad[0] == pytest.approx(kx * np.cos(kx * ax) * np.cos(2 * ky * ay))
    assert grad[1] == pytest.approx(-2 * ky * np.sin(kx * ax) * np.sin(2 * ky * ay))
    assert hess[0, 0] == pytest.approx(-(kx**2) * np.sin(kx * ax) * np.cos(2 * ky * ay))
    assert hess[0, 1] == hess[1, 0]
    wrapped = interp.evaluate((1.0 + ax + lx, -2.0 + ay - ly))
    assert wrapped[0] == pytest.approx(value)


def test_seed_scale_discovery_with_weak_perturbation_finds_all_four_points():
    x, y = _grid(64)
    psi = np.cos(x) + np.cos(y) + 0.03 * np.cos(4 * x) * np.cos(4 * y)
    points = detect_flux_critical_points(psi, seed_scale_cells=2.0, max_points=64)
    assert sorted((p.kind, round(p.psi, 2)) for p in points) == [
        ("O", -1.97),
        ("O", 2.03),
        ("X", 0.03),
        ("X", 0.03),
    ]


def test_flux_contrast_threshold_keeps_or_rejects_whole_structure():
    x, y = _grid(64)
    psi = np.cos(x) + np.cos(y)  # peak-to-peak 4
    kept = detect_flux_critical_points(
        psi, seed_scale_cells=1.0, min_flux_contrast_rtol=0.1, return_report=True
    )
    assert sorted((p.kind, round(p.psi, 2)) for p in kept.points) == [
        ("O", -2.0),
        ("O", 2.0),
        ("X", 0.0),
        ("X", 0.0),
    ]
    strict = detect_flux_critical_points(
        psi, seed_scale_cells=1.0, min_flux_contrast_rtol=0.6, return_report=True
    )
    assert not strict.points
    assert sorted(p.quality for p in strict.rejected) == [
        "insignificant_saddle",
        "insignificant_saddle",
        "low_flux_contrast",
        "low_flux_contrast",
    ]
    assert all(not p.accepted for p in strict.rejected)


def test_select_resolved_empty_returns_empty():
    kept, rejected = select_resolved_flux_points(
        [], np.zeros((8, 8)), lengths=(TWO_PI, TWO_PI), min_flux_contrast=0.1
    )
    assert kept == [] and rejected == []


def test_select_resolved_keeps_islands_and_connecting_saddles():
    x, y = _grid(64)
    psi = np.cos(x) + np.cos(y)
    points = list(detect_flux_critical_points(psi, max_points=500))
    kept, rejected = select_resolved_flux_points(
        points, psi, lengths=(TWO_PI, TWO_PI), min_flux_contrast=0.1, flow_scale_cells=1.0
    )
    assert len(kept) + len(rejected) == len(points)
    assert sorted(round(p.psi) for p in kept if p.kind == "O") == [-2, 2]
    assert any(p.kind == "X" for p in kept)
    assert all(not p.accepted for p in rejected)


def test_select_resolved_marks_small_island_unresolved():
    x, y = _grid(64)
    psi = np.exp(-(((x - 3) ** 2 + (y - 3) ** 2) / (2 * 0.12**2))) + 0.1 * np.cos(x)
    points = [p for p in detect_flux_critical_points(psi, max_points=50) if p.kind == "O"]
    assert points
    _, rejected = select_resolved_flux_points(
        points, psi, lengths=(TWO_PI, TWO_PI), min_flux_contrast=1e-9, min_island_cells=40.0
    )
    assert rejected
    assert {p.quality for p in rejected} <= {"unresolved_island", "low_flux_contrast"}


def test_periodic_excursion_labels_merge_across_seams():
    mask = np.zeros((6, 6), bool)
    mask[0, 2] = mask[5, 2] = True
    mask[3, 0] = mask[3, 5] = True
    mask[2, 2] = True
    labels = _periodic_excursion_labels(mask)
    assert labels[0, 2] == labels[5, 2] != 0
    assert labels[3, 0] == labels[3, 5] != 0
    assert labels[2, 2] not in (0, labels[0, 2], labels[3, 0])
    assert (labels[~mask] == 0).all()


def test_saddle_branches_returns_none_for_degenerate_flow_fields():
    n = 16
    spacing = np.full(2, TWO_PI / n)
    zero, ones = np.zeros((n, n)), np.ones((n, n))
    point = _point("X", 1.0, 1.0)
    below = np.zeros((n, n), int)

    def run(derivatives):
        return _saddle_branches(
            point,
            1,
            below,
            zero,
            derivatives,
            lower=np.zeros(2),
            spacing=spacing,
            lengths=(TWO_PI, TWO_PI),
        )

    assert run([zero] * 5) is None  # singular Hessian
    assert run([np.full((n, n), np.nan)] * 5) is None  # nonfinite step
    assert run([zero, zero, ones, zero, ones]) is None  # minimum, not a saddle
    assert run([ones, zero, ones, zero, -ones]) is None  # Newton drifts away


def test_tracker_constructor_validation():
    with pytest.raises(ValueError, match="positive domain"):
        CriticalPointTracker((1.0, 0.0))
    with pytest.raises(ValueError, match="positive domain"):
        CriticalPointTracker((1.0,))
    with pytest.raises(ValueError, match="max_link_distance"):
        CriticalPointTracker(max_link_distance=-0.1)
    with pytest.raises(ValueError, match="ambiguity_distance"):
        CriticalPointTracker(ambiguity_distance=-0.1)


def test_tracker_defaults_and_state_round_trip_preserves_ids():
    tracker = CriticalPointTracker((8.0, 4.0))
    assert tracker.max_link_distance == pytest.approx(1.0)
    assert tracker.ambiguity_distance == pytest.approx(4e-6)
    tracker.update((_point("X", 1.0), _point("O", 3.0, trace=-2.0)), time=0.5)
    state = tracker.state()
    assert state["frame"] == 0 and state["time"] == 0.5 and state["next_track_id"] == 2
    restored = CriticalPointTracker.from_state(state)
    assert restored.frame == 0 and restored.time == 0.5
    ids_old = {p.kind: i for i, p in tracker.previous}
    out = restored.update((_point("X", 1.1), _point("O", 3.1, trace=-2.0)), time=1.0)
    assert {p.kind: i for i, p in out} == ids_old
    assert not restored.events


def test_tracker_does_not_link_across_kind_or_o_point_sign():
    tracker = CriticalPointTracker((10.0, 10.0), max_link_distance=1.0)
    tracker.update((_point("O", 5.0, trace=-1.0),))
    out = tracker.update((_point("X", 5.0), _point("O", 5.1, trace=+1.0)))
    assert sorted(i for i, _ in out) == [1, 2]
    events = {(e["kind"], e["event"]) for e in tracker.events}
    assert ("O", "disappearance") in events
    assert ("X", "birth") in events and ("O", "birth") in events


def test_tracker_ignores_unaccepted_points_and_reports_birth_and_death():
    tracker = CriticalPointTracker((10.0, 10.0), (False, False), max_link_distance=1.0)
    first = tracker.update((_point("X", 1.0), _point("X", 5.0, accepted=False)), time=0.0)
    assert len(first) == 1 and first[0][0] == 0
    assert [e["event"] for e in tracker.events] == ["birth"]
    out = tracker.update((_point("X", 8.0),), time=1.0)
    assert out[0][0] == 1
    assert {(e["event"], e["track_id"]) for e in tracker.events} == {
        ("birth", 1),
        ("disappearance", 0),
    }
    assert tracker.update(()) == ()
    assert [e["event"] for e in tracker.events] == ["disappearance"]


def test_tracker_follows_a_drifting_x_point_through_periodic_seam():
    tracker = CriticalPointTracker((10.0, 10.0), max_link_distance=0.6)
    ids = []
    for x in (9.5, 9.9, 0.3, 0.7):
        ((track_id, _),) = tracker.update((_point("X", x),))
        ids.append(track_id)
    assert ids == [0, 0, 0, 0]


def test_track_critical_points_flattens_frames_in_order():
    frames = (
        (_point("X", 1.0), _point("O", 4.0, trace=-1.0)),
        (_point("O", 4.2, trace=-1.0), _point("X", 1.2)),
    )
    tracked = track_critical_points(frames, lengths=(10.0, 10.0), max_link_distance=1.0)
    assert len(tracked) == 4
    by_frame = {(i // 2, p.kind): t for i, (t, p) in enumerate(tracked)}
    assert by_frame[(0, "X")] == by_frame[(1, "X")]
    assert by_frame[(0, "O")] == by_frame[(1, "O")]
    assert by_frame[(0, "X")] != by_frame[(0, "O")]


def test_minimum_assignment_is_optimal_on_known_matrix():
    costs = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    assignment = _minimum_assignment(costs)
    assert sorted(assignment) == [0, 1, 2]
    assert costs[np.arange(3), assignment].sum() == pytest.approx(5.0)
