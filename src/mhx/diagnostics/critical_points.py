"""Magnetic-flux critical-point diagnostics for two-dimensional reconnection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FluxCriticalPoint:
    """One detected critical point of the magnetic flux function."""

    kind: str
    index: tuple[int, int]
    position: tuple[float, float]
    psi: float
    gradient_norm: float
    hessian_determinant: float
    hessian_trace: float
    refined: bool = False
    newton_iterations: int = 0
    residual_norm: float | None = None


def detect_flux_critical_points(
    psi: np.ndarray,
    *,
    lengths: tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi),
    periodic: tuple[bool, bool] = (True, True),
    max_points: int = 16,
    min_separation: float | None = None,
    determinant_rtol: float = 1.0e-8,
    refine: bool = False,
    max_refinement_fraction: float = 0.75,
) -> tuple[FluxCriticalPoint, ...]:
    r"""Detect X/O points of a two-dimensional magnetic flux field.

    The detector looks for local minima of ``|∇ψ|`` and classifies each
    candidate by the Hessian determinant: saddles are returned as ``"X"`` and
    elliptic extrema as ``"O"``.  With ``refine=True``, a local quadratic
    Newton correction estimates a sub-cell critical-point position.
    """
    values = np.asarray(psi, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("psi must be a two-dimensional array")
    if values.shape[0] < 4 or values.shape[1] < 4:
        raise ValueError("psi must have at least four points in each direction")
    if len(lengths) != 2 or lengths[0] <= 0.0 or lengths[1] <= 0.0:
        raise ValueError("lengths must contain two positive domain lengths")
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    if determinant_rtol < 0.0:
        raise ValueError("determinant_rtol must be non-negative")
    if max_refinement_fraction <= 0.0:
        raise ValueError("max_refinement_fraction must be positive")
    if min_separation is None:
        min_separation = 1.5 * max(lengths[0] / values.shape[0], lengths[1] / values.shape[1])
    if min_separation < 0.0:
        raise ValueError("min_separation must be non-negative")

    dx = lengths[0] / values.shape[0]
    dy = lengths[1] / values.shape[1]
    fx = _first_derivative(values, axis=0, spacing=dx, periodic=periodic[0])
    fy = _first_derivative(values, axis=1, spacing=dy, periodic=periodic[1])
    fxx = _second_derivative(values, axis=0, spacing=dx, periodic=periodic[0])
    fyy = _second_derivative(values, axis=1, spacing=dy, periodic=periodic[1])
    fxy = _mixed_derivative(values, dx=dx, dy=dy, periodic=periodic)

    gradient_norm = np.hypot(fx, fy)
    determinant = fxx * fyy - fxy * fxy
    trace = fxx + fyy
    determinant_floor = determinant_rtol * max(
        float(np.max(np.abs(determinant))),
        np.finfo(np.float64).eps,
    )
    candidate_mask = _local_minima_mask(gradient_norm, periodic=periodic) & (
        np.abs(determinant) >= determinant_floor
    )
    candidates: list[FluxCriticalPoint] = []
    for i, j in np.argwhere(candidate_mask):
        kind = "O" if determinant[i, j] > 0.0 else "X"
        point = FluxCriticalPoint(
            kind=kind,
            index=(int(i), int(j)),
            position=(float(i * dx), float(j * dy)),
            psi=float(values[i, j]),
            gradient_norm=float(gradient_norm[i, j]),
            hessian_determinant=float(determinant[i, j]),
            hessian_trace=float(trace[i, j]),
        )
        if refine:
            point = _refine_critical_point(
                point,
                values=values,
                gradients=(fx, fy),
                hessians=(fxx, fxy, fyy),
                spacings=(dx, dy),
                lengths=lengths,
                periodic=periodic,
                max_refinement_fraction=max_refinement_fraction,
            )
        candidates.append(point)
    candidates.sort(key=lambda point: (point.gradient_norm, point.kind, point.index))
    selected: list[FluxCriticalPoint] = []
    for point in candidates:
        if all(
            _periodic_distance(point.position, other.position, lengths, periodic) >= min_separation
            for other in selected
        ):
            selected.append(point)
        if len(selected) >= max_points:
            break
    return tuple(selected)


def critical_points_by_kind(
    points: tuple[FluxCriticalPoint, ...],
    kind: str,
) -> tuple[FluxCriticalPoint, ...]:
    """Return detected critical points of one kind sorted by gradient norm."""
    if kind not in {"X", "O"}:
        raise ValueError("kind must be 'X' or 'O'")
    return tuple(point for point in points if point.kind == kind)


def track_critical_points(
    frames: tuple[tuple[FluxCriticalPoint, ...], ...],
    *,
    lengths: tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi),
    periodic: tuple[bool, bool] = (True, True),
    max_link_distance: float | None = None,
) -> tuple[tuple[int, FluxCriticalPoint], ...]:
    """Greedily associate critical points across frames by nearest neighbors."""
    if len(lengths) != 2 or lengths[0] <= 0.0 or lengths[1] <= 0.0:
        raise ValueError("lengths must contain two positive domain lengths")
    if max_link_distance is not None and max_link_distance < 0.0:
        raise ValueError("max_link_distance must be non-negative")
    if not frames:
        return ()
    if max_link_distance is None:
        max_link_distance = 0.25 * min(lengths)

    next_track_id = 0
    previous: list[tuple[int, FluxCriticalPoint]] = []
    tracked: list[tuple[int, FluxCriticalPoint]] = []
    for frame in frames:
        current: list[tuple[int, FluxCriticalPoint]] = []
        used_previous: set[int] = set()
        for point in frame:
            candidate_id: int | None = None
            candidate_index: int | None = None
            candidate_distance = float("inf")
            for previous_index, (track_id, previous_point) in enumerate(previous):
                if previous_index in used_previous or previous_point.kind != point.kind:
                    continue
                distance = _periodic_distance(
                    point.position,
                    previous_point.position,
                    lengths,
                    periodic,
                )
                if distance < candidate_distance:
                    candidate_id = track_id
                    candidate_index = previous_index
                    candidate_distance = distance
            if (
                candidate_id is not None
                and candidate_index is not None
                and candidate_distance <= max_link_distance
            ):
                used_previous.add(candidate_index)
                track_id = candidate_id
            else:
                track_id = next_track_id
                next_track_id += 1
            current.append((track_id, point))
            tracked.append((track_id, point))
        previous = current
    return tuple(tracked)


def _refine_critical_point(
    point: FluxCriticalPoint,
    *,
    values: np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray],
    hessians: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float],
    lengths: tuple[float, float],
    periodic: tuple[bool, bool],
    max_refinement_fraction: float,
) -> FluxCriticalPoint:
    i, j = point.index
    fx, fy = gradients
    fxx, fxy, fyy = hessians
    gradient = np.asarray([fx[i, j], fy[i, j]], dtype=np.float64)
    hessian = np.asarray(
        [[fxx[i, j], fxy[i, j]], [fxy[i, j], fyy[i, j]]],
        dtype=np.float64,
    )
    try:
        delta = np.linalg.solve(hessian, -gradient)
    except np.linalg.LinAlgError:
        return point
    max_delta = np.asarray(spacings, dtype=np.float64) * max_refinement_fraction
    delta = np.clip(delta, -max_delta, max_delta)
    refined_position = np.asarray(point.position, dtype=np.float64) + delta
    for axis, is_periodic in enumerate(periodic):
        if is_periodic:
            refined_position[axis] = refined_position[axis] % lengths[axis]
        else:
            refined_position[axis] = float(
                np.clip(refined_position[axis], 0.0, lengths[axis])
            )
    refined_gradient = gradient + hessian @ delta
    refined_psi = float(values[i, j] + gradient @ delta + 0.5 * delta @ hessian @ delta)
    residual = float(np.linalg.norm(refined_gradient))
    return FluxCriticalPoint(
        kind=point.kind,
        index=point.index,
        position=(float(refined_position[0]), float(refined_position[1])),
        psi=refined_psi,
        gradient_norm=residual,
        hessian_determinant=point.hessian_determinant,
        hessian_trace=point.hessian_trace,
        refined=True,
        newton_iterations=1,
        residual_norm=residual,
    )


def _first_derivative(
    values: np.ndarray,
    *,
    axis: int,
    spacing: float,
    periodic: bool,
) -> np.ndarray:
    if periodic:
        return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (
            2.0 * spacing
        )
    return np.gradient(values, spacing, axis=axis, edge_order=2)


def _second_derivative(
    values: np.ndarray,
    *,
    axis: int,
    spacing: float,
    periodic: bool,
) -> np.ndarray:
    if periodic:
        return (
            np.roll(values, -1, axis=axis)
            - 2.0 * values
            + np.roll(values, 1, axis=axis)
        ) / (spacing * spacing)
    first = np.gradient(values, spacing, axis=axis, edge_order=2)
    return np.gradient(first, spacing, axis=axis, edge_order=2)


def _mixed_derivative(
    values: np.ndarray,
    *,
    dx: float,
    dy: float,
    periodic: tuple[bool, bool],
) -> np.ndarray:
    first_x = _first_derivative(values, axis=0, spacing=dx, periodic=periodic[0])
    return _first_derivative(first_x, axis=1, spacing=dy, periodic=periodic[1])


def _local_minima_mask(values: np.ndarray, *, periodic: tuple[bool, bool]) -> np.ndarray:
    mask = np.ones_like(values, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = values
            if periodic[0]:
                shifted = np.roll(shifted, di, axis=0)
            elif di != 0:
                shifted = _shift_with_edge(shifted, di, axis=0)
            if periodic[1]:
                shifted = np.roll(shifted, dj, axis=1)
            elif dj != 0:
                shifted = _shift_with_edge(shifted, dj, axis=1)
            mask &= values <= shifted
    return mask


def _shift_with_edge(values: np.ndarray, offset: int, *, axis: int) -> np.ndarray:
    shifted = np.empty_like(values)
    if axis == 0:
        if offset < 0:
            shifted[:-1, :] = values[1:, :]
            shifted[-1, :] = values[-1, :]
        else:
            shifted[1:, :] = values[:-1, :]
            shifted[0, :] = values[0, :]
    else:
        if offset < 0:
            shifted[:, :-1] = values[:, 1:]
            shifted[:, -1] = values[:, -1]
        else:
            shifted[:, 1:] = values[:, :-1]
            shifted[:, 0] = values[:, 0]
    return shifted


def _periodic_distance(
    left: tuple[float, float],
    right: tuple[float, float],
    lengths: tuple[float, float],
    periodic: tuple[bool, bool],
) -> float:
    deltas = []
    for coordinate, other, length, is_periodic in zip(left, right, lengths, periodic, strict=True):
        delta = abs(coordinate - other)
        if is_periodic:
            delta = min(delta, length - delta)
        deltas.append(delta)
    return float(np.hypot(deltas[0], deltas[1]))
