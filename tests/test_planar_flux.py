"""Tests for host-side tracked planar flux: regions, rate stencil, validation, restarts."""

from __future__ import annotations

import numpy as np
import pytest

from mhx.diagnostics import FluxPairRegion, analyze_planar_flux, join_planar_flux
from mhx.diagnostics.planar_flux import (
    PLANAR_FLUX_SCHEMA,
    continuous_flux_rates,
    double_sheet_pair_regions,
)

N = 32
LENGTHS = (2 * np.pi, 2 * np.pi)
BOUNDS = ((-1.0, 1.0), (0.0, LENGTHS[1]))
REGION = FluxPairRegion("sheet", BOUNDS, BOUNDS)
TIME = np.array([0.0, 0.13, 0.31, 0.5, 0.7])


def _mesh():
    axis = np.arange(N) * LENGTHS[0] / N
    return np.meshgrid(axis, axis, indexing="ij")


def _fields(time=TIME):
    """psi = cos(x - s(t)) + a(t) cos(y): one O and one X in the box, flux = 2 a(t)."""
    x, y = _mesh()
    return [np.cos(x + 0.05 - 0.2 * t) + (0.2 + 0.1 * t**2) * np.cos(y) for t in time]


def test_double_sheet_pair_regions_default_and_mode() -> None:
    regions = double_sheet_pair_regions((8.0, 4.0))
    assert [r.name for r in regions] == ["sheet_0", "sheet_1"]
    assert regions[0].o_bounds == ((1.0, 3.0), (0.0, 4.0))
    assert regions[1].x_bounds == ((5.0, 7.0), (0.0, 4.0))
    shifted = double_sheet_pair_regions((8.0, 4.0), lower=(1.0, 2.0), centers=(3.0,), mode_y=2)
    assert len(shifted) == 1
    assert shifted[0].o_bounds == ((2.0, 4.0), (1.5, 3.5))
    with pytest.raises(ValueError, match="mode_y"):
        double_sheet_pair_regions((8.0, 4.0), mode_y=0)


def test_continuous_flux_rates_nonuniform_stencil_and_identity_switch() -> None:
    time = np.array([0.0, 0.3, 0.4, 1.0, 1.5])
    flux = (2.0 * time**2 - time + 1.0)[:, None]
    valid = np.ones((5, 1), bool)
    ids = np.zeros((5, 1), int)
    rate, rate_valid = continuous_flux_rates(time, flux, valid, ids, ids)
    # The nonuniform three-point stencil is exact for quadratics.
    np.testing.assert_allclose(rate[1:4, 0], 4.0 * time[1:4] - 1.0, atol=1e-12)
    assert rate_valid[:, 0].tolist() == [False, True, True, True, False]
    assert np.isnan(rate[[0, -1]]).all()
    switched = ids.copy()
    switched[3, 0] = 7
    _, rate_valid = continuous_flux_rates(time, flux, valid, switched, ids)
    assert rate_valid[:, 0].tolist() == [False, True, False, False, False]
    broken = valid.copy()
    broken[2, 0] = False
    _, rate_valid = continuous_flux_rates(time, flux, broken, ids, ids)
    assert not rate_valid.any()


def test_analyze_planar_flux_known_flux_callable_electric_and_progress() -> None:
    fields = _fields()
    x, y = _mesh()
    electric = [-0.2 * t * np.cos(y) - 0.2 * np.sin(x) for t in TIME]
    seen = []
    result = analyze_planar_flux(
        TIME,
        fields,
        lengths=LENGTHS,
        regions=(REGION,),
        electric_fields=lambda frame: electric[frame],
        progress=lambda frame, t, report: seen.append((frame, t, len(report.points))),
    )
    assert [s[0] for s in seen] == list(range(len(TIME)))
    assert all(s[2] == 4 for s in seen)  # 2 O + 2 X in the periodic box
    arrays = result.arrays
    assert arrays["flux_valid"].all()
    np.testing.assert_allclose(arrays["paired_flux"][:, 0], 0.4 + 0.2 * TIME**2, atol=1e-12)
    assert (arrays["pair_status"] == "valid").all()
    assert result.diagnostics["schema"] == PLANAR_FLUX_SCHEMA
    assert result.diagnostics["valid_flux_samples"] == len(TIME)
    assert result.diagnostics["valid_rate_samples"] == len(TIME) - 2
    # Every frame keeps two O and two X accepted points with persistent ids.
    accepted = arrays["point_accepted"]
    assert accepted.sum() == 4 * len(TIME)
    assert set(arrays["point_kind"][accepted]) == {"O", "X"}
    assert len(set(arrays["point_track_id"][accepted])) == 4
    assert arrays["o_track_id"][:, 0].tolist() == [arrays["o_track_id"][0, 0]] * len(TIME)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"time": [], "fields": []}, "strictly increasing"),
        ({"time": [0.0, 0.0], "fields": [0, 0]}, "strictly increasing"),
        ({"time": [0.0, np.nan], "fields": [0, 0]}, "strictly increasing"),
        ({"time": [[0.0, 1.0]], "fields": [0]}, "strictly increasing"),
        ({"time": [0.0, 1.0], "fields": [0]}, "must match"),
        ({"time": [0.0, 1.0], "fields": [0, 0], "electric": [0]}, "must match"),
        ({"time": [0.0], "fields": [0], "regions": (REGION, REGION)}, "unique"),
        (
            {
                "time": [0.0],
                "fields": [0],
                "regions": (FluxPairRegion("bad", ((1.0, 1.0), (0.0, 1.0)), BOUNDS),),
            },
            "positive width",
        ),
        (
            {
                "time": [0.0],
                "fields": [0],
                "regions": (FluxPairRegion("big", BOUNDS, ((0.0, 7.0), (0.0, 1.0))),),
            },
            "positive width",
        ),
    ],
)
def test_analyze_planar_flux_validates_inputs(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        analyze_planar_flux(
            kwargs["time"],
            kwargs["fields"],
            lengths=LENGTHS,
            regions=kwargs.get("regions", ()),
            electric_fields=kwargs.get("electric"),
        )


def test_no_regions_still_tracks_points() -> None:
    result = analyze_planar_flux(TIME[:2], _fields(TIME[:2]), lengths=LENGTHS)
    assert result.arrays["paired_flux"].shape == (2, 0)
    assert result.arrays["pair_names"].size == 0
    assert result.arrays["point_accepted"].sum() == 8


def test_truncated_detection_marks_pair_status() -> None:
    result = analyze_planar_flux(
        TIME[:2],
        _fields(TIME[:2]),
        lengths=LENGTHS,
        regions=(REGION,),
        detector_options={"max_points": 1},
    )
    assert (result.arrays["pair_status"] == "detection_truncated").all()
    assert not result.arrays["flux_valid"].any()
    assert result.diagnostics["truncated_frames"] == 2
    pair_events = [e for e in result.tracking["events"] if "pair" in e]
    assert len(pair_events) == 2
    assert all(e["event"] == "detection_truncated" for e in pair_events)


def test_ambiguous_selection_and_missing_pair() -> None:
    full = ((0.0, LENGTHS[0]), (0.0, LENGTHS[1]))
    # The whole periodic box holds two O and two X points: the selection is ambiguous.
    ambiguous = analyze_planar_flux(
        TIME[:2],
        _fields(TIME[:2]),
        lengths=LENGTHS,
        regions=(FluxPairRegion("all", full, full),),
    )
    assert (ambiguous.arrays["pair_status"] == "ambiguous_selection").all()
    assert not ambiguous.arrays["flux_valid"].any()
    events = [e["event"] for e in ambiguous.tracking["events"] if "pair" in e]
    assert events and set(events) == {"ambiguous_selection"}
    empty = ((3.0, 3.1), (3.0, 3.1))
    missing = analyze_planar_flux(
        TIME[:2], _fields(TIME[:2]), lengths=LENGTHS, regions=(FluxPairRegion("e", empty, empty),)
    )
    assert (missing.arrays["pair_status"] == "missing").all()
    assert (missing.arrays["o_track_id"] == -1).all()


def test_pair_switch_event_when_identities_change() -> None:
    x, y = _mesh()
    # Frame 1 shifts the island farther than max_link_distance, so ids change.
    fields = [np.cos(x) + 0.2 * np.cos(y), np.cos(x - 0.5) + 0.2 * np.cos(y)]
    result = analyze_planar_flux(
        [0.0, 1.0], fields, lengths=LENGTHS, regions=(REGION,), max_link_distance=0.1
    )
    assert result.arrays["flux_valid"].all()
    ids = result.arrays["o_track_id"][:, 0]
    assert ids[0] != ids[1]
    assert any(e.get("event") == "pair_switch" for e in result.tracking["events"])


def test_restart_geometry_and_topology_mismatch() -> None:
    fields = _fields()
    first = analyze_planar_flux(TIME[:3], fields[:3], lengths=LENGTHS, regions=(REGION,))
    state = first.tracking["snapshots"][-1]["state"]
    with pytest.raises(ValueError, match="geometry"):
        analyze_planar_flux(
            TIME[2:], fields[2:], lengths=(4 * np.pi, 4 * np.pi), initial_tracking_state=state
        )
    x, y = _mesh()
    other = np.cos(2 * x) + 0.2 * np.cos(2 * y)
    with pytest.raises(ValueError, match="topology disagrees"):
        analyze_planar_flux(
            TIME[2:4], [other, other], lengths=LENGTHS, initial_tracking_state=state
        )


def test_join_planar_flux_paths() -> None:
    fields = _fields()
    first = analyze_planar_flux(TIME[:3], fields[:3], lengths=LENGTHS, regions=(REGION,))
    second = analyze_planar_flux(TIME[3:], fields[3:], lengths=LENGTHS, regions=(REGION,))
    assert join_planar_flux(None, first) is first
    # Disjoint histories (no shared checkpoint) are concatenated.
    gap = join_planar_flux(first, second)
    np.testing.assert_allclose(gap.arrays["time"], TIME)
    assert gap.diagnostics["valid_flux_samples"] == len(TIME)
    whole = analyze_planar_flux(TIME, fields, lengths=LENGTHS, regions=(REGION,))
    assert gap.arrays["point_time"].shape == whole.arrays["point_time"].shape
    # Overlapping histories that do not share a time stamp are rejected.
    overlap = analyze_planar_flux(TIME[1:] + 0.01, fields[1:], lengths=LENGTHS, regions=(REGION,))
    with pytest.raises(ValueError, match="share a checkpoint"):
        join_planar_flux(first, overlap)
    renamed = analyze_planar_flux(
        TIME[2:], fields[2:], lengths=LENGTHS, regions=(FluxPairRegion("other", BOUNDS, BOUNDS),)
    )
    with pytest.raises(ValueError, match="settings changed"):
        join_planar_flux(first, renamed)
    moved_bounds = ((-0.9, 0.9), (0.0, LENGTHS[1]))
    moved = analyze_planar_flux(
        TIME[2:],
        fields[2:],
        lengths=LENGTHS,
        regions=(FluxPairRegion("sheet", moved_bounds, moved_bounds),),
    )
    with pytest.raises(ValueError, match="boxes changed"):
        join_planar_flux(first, moved)


def test_join_accepts_json_roundtripped_regions() -> None:
    fields = _fields()
    first = analyze_planar_flux(TIME[:3], fields[:3], lengths=LENGTHS, regions=(REGION,))
    second = analyze_planar_flux(TIME[2:], fields[2:], lengths=LENGTHS, regions=(REGION,))
    listified = [
        {k: ([list(b) for b in v] if isinstance(v, tuple) else v) for k, v in r.items()}
        for r in first.diagnostics["regions"]
    ]
    first.diagnostics["regions"] = listified
    joined = join_planar_flux(first, second)
    assert joined.arrays["time"].tolist() == TIME.tolist()


def test_jax_backend_matches_numpy_electric_rate() -> None:
    fields = _fields(TIME[:3])
    x, y = _mesh()
    electric = [np.sin(x) * np.cos(y) + t for t in TIME[:3]]
    kwargs = {"lengths": LENGTHS, "regions": (REGION,), "electric_fields": electric}
    reference = analyze_planar_flux(TIME[:3], fields, **kwargs)
    jax_result = analyze_planar_flux(
        TIME[:3], fields, detector_options={"backend": "jax"}, **kwargs
    )
    np.testing.assert_allclose(
        jax_result.arrays["paired_flux"], reference.arrays["paired_flux"], atol=1e-9
    )
    np.testing.assert_allclose(
        jax_result.arrays["electric_field_rate"],
        reference.arrays["electric_field_rate"],
        atol=1e-8,
    )
