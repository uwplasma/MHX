"""Host-side tracked planar flux; no proxy fallback or differentiation through topology."""

from dataclasses import asdict, dataclass

import numpy as np

from mhx.diagnostics.critical_points import (
    CriticalPointTracker,
    PeriodicFluxInterpolator,
    detect_flux_critical_points,
)

PLANAR_FLUX_SCHEMA = "mhx.planar_flux.v1"


@dataclass(frozen=True)
class FluxPairRegion:
    """Declared O/X selection boxes enclosing one island and its bounded separatrix."""

    name: str
    o_bounds: tuple[tuple[float, float], tuple[float, float]]
    x_bounds: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class PlanarFluxResult:
    arrays: dict
    tracking: dict
    diagnostics: dict


def double_sheet_pair_regions(lengths, *, lower=(0.0, 0.0), centers=None, mode_y=1):
    """One declared island wavelength on each sheet; extra points invalidate selection."""
    if mode_y < 1:
        raise ValueError("mode_y must be positive")
    lx, ly = lengths
    if centers is None:
        centers = (lower[0] + lx / 4, lower[0] + 3 * lx / 4)
    wavelength = ly / mode_y
    y_bounds = (
        (lower[1], lower[1] + ly)
        if mode_y == 1
        else (lower[1] - wavelength / 4, lower[1] + 3 * wavelength / 4)
    )
    return tuple(
        FluxPairRegion(
            f"sheet_{i}",
            ((center - lx / 8, center + lx / 8), y_bounds),
            ((center - lx / 8, center + lx / 8), y_bounds),
        )
        for i, center in enumerate(centers)
    )


def continuous_flux_rates(time, flux, valid, o_ids, x_ids):
    """Nonuniform three-point central stencil; endpoints and identity switches are invalid."""
    time = np.asarray(time)
    rate = np.full_like(flux, np.nan, dtype=float)
    rate_valid = np.zeros_like(valid, dtype=bool)
    for i in range(1, len(time) - 1):
        same = (
            valid[i - 1 : i + 2].all(axis=0)
            & (o_ids[i - 1] == o_ids[i])
            & (o_ids[i] == o_ids[i + 1])
            & (x_ids[i - 1] == x_ids[i])
            & (x_ids[i] == x_ids[i + 1])
        )
        left, right = time[i] - time[i - 1], time[i + 1] - time[i]
        derivative = (
            -right / (left * (left + right)) * flux[i - 1]
            + (right - left) / (left * right) * flux[i]
            + left / (right * (left + right)) * flux[i + 1]
        )
        rate[i, same], rate_valid[i, same] = derivative[same], True
    return rate, rate_valid


def analyze_planar_flux(
    time,
    psi_fields,
    *,
    lengths=(2 * np.pi,) * 2,
    lower=(0.0, 0.0),
    regions=(),
    electric_fields=None,
    initial_tracking_state=None,
    detector_options=None,
    max_link_distance=None,
    ambiguity_distance=None,
    progress=None,
):
    """Track accepted points, select only unambiguous declared pairs and measure signed flux.

    Electric fields are the actual ``E_z=-RHS_psi``. The comparable signed rate
    is ``E_z(X)-E_z(O)``. Every point and event retains frame/time and quality.
    """
    time = np.asarray(time, dtype=float)
    if not time.size or time.ndim != 1 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be nonempty and strictly increasing")
    if len(psi_fields) != len(time) or (
        electric_fields is not None
        and not callable(electric_fields)
        and len(electric_fields) != len(time)
    ):
        raise ValueError("field and time histories must match")
    if len({region.name for region in regions}) != len(regions):
        raise ValueError("pair region names must be unique")
    for region in regions:
        for bounds in (region.o_bounds, region.x_bounds):
            if len(bounds) != 2 or any(
                not 0 < hi - lo <= length * (1 + 1e-12)
                for (lo, hi), length in zip(bounds, lengths, strict=True)
            ):
                raise ValueError(
                    "selection bounds must have positive width within the periodic box"
                )
    options = {"max_points": 64, **(detector_options or {})}
    options.update(refine=True, return_report=True, lengths=lengths, lower=lower)
    tracker = (
        CriticalPointTracker.from_state(initial_tracking_state)
        if initial_tracking_state is not None
        else CriticalPointTracker(
            lengths, max_link_distance=max_link_distance, ambiguity_distance=ambiguity_distance
        )
    )
    if tuple(tracker.lengths) != tuple(lengths) or not all(tracker.periodic):
        raise ValueError("tracking state geometry differs from the periodic analysis")
    shape = (len(time), len(regions))
    flux, electric_rate = np.full(shape, np.nan), np.full(shape, np.nan)
    valid, o_ids, x_ids = np.zeros(shape, bool), np.full(shape, -1), np.full(shape, -1)
    status = np.full(shape, "missing", dtype="U24")
    rows, events, snapshots, reports = [], [], [], []
    previous_pairs = None
    for frame, (t, psi) in enumerate(zip(time, psi_fields, strict=True)):
        report = detect_flux_critical_points(psi, **options)
        repeated_checkpoint = (
            frame == 0
            and tracker.time is not None
            and np.isclose(t, tracker.time, rtol=1e-12, atol=1e-12)
        )
        previous_ids = {i for i, _ in tracker.previous}
        if repeated_checkpoint:
            tracker.frame -= 1
        points = tracker.update(report.points, time=float(t))
        if repeated_checkpoint and previous_ids != {i for i, _ in points}:
            raise ValueError("checkpoint topology disagrees with saved tracking state")
        events.extend(tracker.events)
        snapshots.append(dict(time=float(t), state=tracker.state()))
        reports.append(report)
        for track_id, point in (*points, *((-1, p) for p in report.rejected)):
            rows.append(dict(frame=tracker.frame, time=t, track_id=track_id, **asdict(point)))
        ambiguous_ids = {
            event["track_id"] for event in tracker.events if event["event"] == "ambiguity"
        }
        electric_values = (
            electric_fields(frame)
            if callable(electric_fields)
            else electric_fields[frame]
            if electric_fields is not None
            else None
        )
        interpolator_type = PeriodicFluxInterpolator
        if options.get("backend") == "jax":
            from mhx.diagnostics.critical_points_jax import JaxPeriodicFluxInterpolator

            interpolator_type = JaxPeriodicFluxInterpolator
        electric = (
            interpolator_type(electric_values, lengths, lower)
            if electric_fields is not None
            else None
        )
        electric_at_points = None
        if electric is not None and options.get("backend") == "jax":
            electric_at_points = dict(
                zip(
                    (i for i, _ in points),
                    electric.sample_values([p.position for _, p in points]),
                    strict=True,
                )
            )
        for j, region in enumerate(regions):
            selected = []
            for kind, bounds in (("O", region.o_bounds), ("X", region.x_bounds)):
                selected.append(
                    [
                        (i, p)
                        for i, p in points
                        if p.kind == kind and _inside(p.position, bounds, lengths)
                    ]
                )
            if report.truncated:
                status[frame, j] = "detection_truncated"
            elif any(len(group) > 1 for group in selected):
                status[frame, j] = "ambiguous_selection"
            elif all(len(group) == 1 for group in selected):
                (oi, op), (xi, xp) = selected[0][0], selected[1][0]
                o_ids[frame, j], x_ids[frame, j] = oi, xi
                if oi in ambiguous_ids or xi in ambiguous_ids:
                    status[frame, j] = "ambiguous_tracking"
                else:
                    flux[frame, j], valid[frame, j] = op.psi - xp.psi, True
                    status[frame, j] = "valid"
                    if electric is not None:
                        electric_rate[frame, j] = (
                            electric_at_points[xi] - electric_at_points[oi]
                            if electric_at_points is not None
                            else electric.evaluate(xp.position)[0]
                            - electric.evaluate(op.position)[0]
                        )
            pair = (o_ids[frame, j], x_ids[frame, j])
            if status[frame, j] != "valid" or (
                previous_pairs is not None and pair != previous_pairs[j]
            ):
                events.append(
                    dict(
                        frame=tracker.frame,
                        time=float(t),
                        pair=region.name,
                        event=str(status[frame, j]) if not valid[frame, j] else "pair_switch",
                    )
                )
        previous_pairs = list(zip(o_ids[frame], x_ids[frame], strict=True))
        if progress is not None:
            progress(frame, float(t), report)
    rate, rate_valid = continuous_flux_rates(time, flux, valid, o_ids, x_ids)
    arrays = dict(
        time=time,
        pair_names=np.asarray([region.name for region in regions], dtype="U64"),
        paired_flux=flux,
        flux_valid=valid,
        o_track_id=o_ids,
        x_track_id=x_ids,
        pair_status=status,
        reconnection_rate=rate,
        rate_valid=rate_valid,
        electric_field_rate=electric_rate,
        rate_electric_difference=rate - electric_rate,
        detection_truncated=np.asarray([r.truncated for r in reports]),
        candidate_count=np.asarray([r.candidate_count for r in reports]),
        rejected_count=np.asarray([len(r.rejected) for r in reports]),
    )
    for name, dtype in (
        ("frame", int),
        ("time", float),
        ("track_id", int),
        ("kind", "U1"),
        ("psi", float),
        ("residual_norm", float),
        ("hessian_condition", float),
        ("hessian_determinant", float),
        ("hessian_trace", float),
        ("quality", "U24"),
        ("accepted", bool),
    ):
        arrays[f"point_{name}"] = np.asarray([row[name] for row in rows], dtype=dtype)
    arrays["point_position"] = np.asarray([row["position"] for row in rows]).reshape((-1, 2))
    diagnostics = dict(
        schema=PLANAR_FLUX_SCHEMA,
        method="fourier_newton",
        flux_sign="psi(O)-psi(X)",
        electric_rate_sign="E_z(X)-E_z(O)",
        pairing_rule="Exactly one accepted O and X within each declared box.",
        regions=[asdict(region) for region in regions],
        rate_stencil="Nonuniform three-point central; same O/X IDs in all three frames.",
        endpoint_policy="Invalid (NaN); no one-sided or proxy rates.",
        valid_flux_samples=int(valid.sum()),
        valid_rate_samples=int(rate_valid.sum()),
        truncated_frames=int(arrays["detection_truncated"].sum()),
        max_link_distance=tracker.max_link_distance,
        ambiguity_distance=tracker.ambiguity_distance,
        detector_options=options,
    )
    return PlanarFluxResult(
        arrays,
        dict(schema=PLANAR_FLUX_SCHEMA, events=events, snapshots=snapshots, settings=diagnostics),
        diagnostics,
    )


def _inside(position, bounds, lengths):
    return all(
        (coordinate - lo) % length < hi - lo or np.isclose(hi - lo, length)
        for coordinate, (lo, hi), length in zip(position, bounds, lengths, strict=True)
    )


def join_planar_flux(previous, current):
    """Join at a checkpoint and recompute temporal stencils across the boundary."""
    if previous is None:
        return current
    a, b = previous.arrays, current.arrays
    if not np.array_equal(a["pair_names"], b["pair_names"]):
        raise ValueError("paired-flux selection settings changed across restart")
    if previous.diagnostics["regions"] != current.diagnostics["regions"]:
        # JSON readers convert tuples to lists: compare a canonical representation.
        import json

        if json.dumps(previous.diagnostics["regions"]) != json.dumps(
            current.diagnostics["regions"]
        ):
            raise ValueError("paired-flux selection boxes changed across restart")
    matches = np.flatnonzero(np.isclose(a["time"], b["time"][0], rtol=1e-12, atol=1e-12))
    if not matches.size and b["time"][0] <= a["time"][-1]:
        raise ValueError("planar flux histories must share a checkpoint")
    boundary = int(matches[-1]) if matches.size else len(a["time"]) - 1
    start = 1 if matches.size else 0
    checkpoint = a["time"][boundary]
    arrays = {}
    for key in a:
        if key == "pair_names":
            arrays[key] = a[key]
        elif key.startswith("point_"):
            arrays[key] = np.concatenate(
                (a[key][a["point_time"] <= checkpoint], b[key][b["point_time"] > checkpoint])
            )
        else:
            arrays[key] = np.concatenate((a[key][: boundary + 1], b[key][start:]))
    arrays["reconnection_rate"], arrays["rate_valid"] = continuous_flux_rates(
        arrays["time"],
        arrays["paired_flux"],
        arrays["flux_valid"],
        arrays["o_track_id"],
        arrays["x_track_id"],
    )
    arrays["rate_electric_difference"] = arrays["reconnection_rate"] - arrays["electric_field_rate"]
    tracking = {**current.tracking}
    for key in ("events", "snapshots"):
        tracking[key] = [item for item in previous.tracking[key] if item["time"] <= checkpoint] + [
            item for item in current.tracking[key] if item["time"] > checkpoint
        ]
    diagnostics = {
        **current.diagnostics,
        "valid_flux_samples": int(arrays["flux_valid"].sum()),
        "valid_rate_samples": int(arrays["rate_valid"].sum()),
        "truncated_frames": int(arrays["detection_truncated"].sum()),
    }
    tracking["settings"] = diagnostics
    return PlanarFluxResult(arrays, tracking, diagnostics)
