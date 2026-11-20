"""Magnetic-flux critical-point diagnostics for two-dimensional reconnection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

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
    hessian_condition: float = 1.0
    accepted: bool = True
    quality: str = "unrefined"


@dataclass(frozen=True)
class CriticalPointDetection:
    """Accepted points and explicit rejection/cap diagnostics."""

    points: tuple[FluxCriticalPoint, ...]
    rejected: tuple[FluxCriticalPoint, ...]
    candidate_count: int
    truncated: bool
    method: str


class PeriodicFluxInterpolator:
    """Evaluate the real Fourier interpolant and its analytic derivatives on the host."""

    def __init__(self, values, lengths=(2.0 * np.pi,) * 2, lower=(0.0, 0.0)):
        values = np.asarray(values, dtype=np.float64)
        self.lengths, self.lower = np.asarray(lengths), np.asarray(lower)
        self.mean = float(np.mean(values))
        kx, ky = [
            2 * np.pi * np.fft.fftfreq(n, d=length / n)
            for n, length in zip(values.shape, lengths, strict=True)
        ]
        self.kx, self.ky = kx, ky
        coefficients = np.fft.fftn(values - self.mean) / values.size
        x, y = kx[:, None], ky[None, :]
        self.coefficients = np.asarray(
            [
                coefficients,
                1j * x * coefficients,
                1j * y * coefficients,
                -(x**2) * coefficients,
                -x * y * coefficients,
                -(y**2) * coefficients,
            ]
        )

    def evaluate(self, position):
        x, y = (np.asarray(position) - self.lower) % self.lengths
        basis = np.exp(1j * self.kx * x)[:, None] * np.exp(1j * self.ky * y)[None, :]
        value, gx, gy, hxx, hxy, hyy = np.einsum(
            "ij,kij->k", basis, self.coefficients, optimize=False
        ).real
        return value + self.mean, np.asarray([gx, gy]), np.asarray([[hxx, hxy], [hxy, hyy]])


def detect_flux_critical_points(
    psi: np.ndarray,
    *,
    lengths: tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi),
    periodic: tuple[bool, bool] = (True, True),
    max_points: int = 16,
    min_separation: float | None = None,
    determinant_rtol: float = 1.0e-8,
    refine: bool = True,
    max_refinement_fraction: float = 0.75,
    lower: tuple[float, float] = (0.0, 0.0),
    residual_atol: float = 1.0e-10,
    residual_rtol: float = 1.0e-7,
    max_iterations: int = 12,
    max_candidates: int | None = None,
    return_report: bool = False,
    backend: str = "numpy",
    seed_scale_cells: float = 0.0,
    min_flux_contrast_rtol: float = 0.0,
    min_island_cells: float = 4.0,
) -> tuple[FluxCriticalPoint, ...] | CriticalPointDetection:
    r"""Detect X/O points of a two-dimensional magnetic flux field.

    Gradient sign-crossing cells supplement local minima, important for weak
    islands on strong sheets. Periodic refinement evaluates the Fourier field
    again at every Newton iterate; its reported residual is never the solved
    linearized residual. ``return_report`` exposes rejected roots and caps.
    Optional Gaussian seed discovery suppresses grid-scale oscillations without
    changing the field used for Newton refinement or flux measurements. Excursion
    basins can then select islands with a declared minimum flux contrast and size.
    """
    values = np.asarray(psi, dtype=np.float64)
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be numpy or jax")
    if backend == "jax" and (not all(periodic) or not refine):
        raise ValueError("jax backend requires periodic Newton refinement")
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
    if residual_atol < 0 or residual_rtol < 0 or max_iterations < 1:
        raise ValueError("residual tolerances must be non-negative and max_iterations positive")
    if len(lower) != 2 or len(periodic) != 2 or not np.isfinite(values).all():
        raise ValueError("finite psi and two lower/periodic coordinates are required")
    if not np.isfinite(lengths).all() or not np.isfinite(lower).all():
        raise ValueError("domain coordinates must be finite")
    if max_candidates is None:
        max_candidates = max(256, 16 * max_points)
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if (
        not np.isfinite([seed_scale_cells, min_flux_contrast_rtol, min_island_cells]).all()
        or min(seed_scale_cells, min_flux_contrast_rtol) < 0
        or min_island_cells < 1
    ):
        raise ValueError("selection scales must be finite and non-negative; island cells >= 1")
    if (seed_scale_cells or min_flux_contrast_rtol) and not all(periodic):
        raise ValueError("resolved feature selection requires a periodic field")
    if min_separation is None:
        min_separation = 0.5 * min(lengths[0] / values.shape[0], lengths[1] / values.shape[1])
    if min_separation < 0.0:
        raise ValueError("min_separation must be non-negative")

    dx, dy = [
        length / (n if wrap else n - 1)
        for length, n, wrap in zip(lengths, values.shape, periodic, strict=True)
    ]
    if all(periodic):
        interpolator_type = PeriodicFluxInterpolator
        if backend == "jax":
            from mhx.diagnostics.critical_points_jax import JaxPeriodicFluxInterpolator

            interpolator_type = JaxPeriodicFluxInterpolator
        interpolator = interpolator_type(values, lengths, lower)
        # Spectral seed gradients avoid finite-difference displacement bias.
        spectral = interpolator.coefficients[0] * values.size
        fx = np.fft.ifftn(1j * interpolator.kx[:, None] * spectral).real
        fy = np.fft.ifftn(1j * interpolator.ky[None, :] * spectral).real
        gradient_norm = np.hypot(fx, fy)
        evaluate = interpolator.evaluate
    else:
        fx = _first_derivative(values, axis=0, spacing=dx, periodic=periodic[0])
        fy = _first_derivative(values, axis=1, spacing=dy, periodic=periodic[1])
        fxx = _second_derivative(values, axis=0, spacing=dx, periodic=periodic[0])
        fyy = _second_derivative(values, axis=1, spacing=dy, periodic=periodic[1])
        fxy = _mixed_derivative(values, dx=dx, dy=dy, periodic=periodic)
        gradient_norm = np.hypot(fx, fy)

        def evaluate(position):
            index = (np.asarray(position) - lower) / np.asarray([dx, dy])
            samples = [_bilinear(field, index) for field in (values, fx, fy, fxx, fxy, fyy)]
            value, gx, gy, hxx, hxy, hyy = samples
            return value, np.asarray([gx, gy]), np.asarray([[hxx, hxy], [hxy, hyy]])

    tolerance = residual_atol + residual_rtol * float(np.max(gradient_norm))
    seed_fx, seed_fy = fx, fy
    if seed_scale_cells:
        window = np.exp(
            -0.5
            * seed_scale_cells**2
            * ((interpolator.kx[:, None] * dx) ** 2 + (interpolator.ky[None, :] * dy) ** 2)
        )
        seed_fx = np.fft.ifftn(1j * interpolator.kx[:, None] * spectral * window).real
        seed_fy = np.fft.ifftn(1j * interpolator.ky[None, :] * spectral * window).real
        angle = np.arctan2(seed_fy, seed_fx)
        corners = (
            angle,
            np.roll(angle, -1, 0),
            np.roll(np.roll(angle, -1, 0), -1, 1),
            np.roll(angle, -1, 1),
        )
        winding = sum(
            (b - a + np.pi) % (2 * np.pi) - np.pi
            for a, b in zip(corners, (*corners[1:], corners[0]), strict=True)
        )
        candidate_mask = np.abs(winding) > np.pi
    else:
        candidate_mask = _local_minima_mask(gradient_norm, periodic=periodic)
    if refine and not seed_scale_cells:
        crosses = []
        for field in (fx, fy):
            corners = np.stack(
                [
                    field,
                    np.roll(field, -1, 0),
                    np.roll(field, -1, 1),
                    np.roll(np.roll(field, -1, 0), -1, 1),
                ]
            )
            crosses.append((corners.min(0) <= 0) & (corners.max(0) >= 0))
        candidate_mask |= crosses[0] & crosses[1]
    if not periodic[0]:
        candidate_mask[-1] = False
    if not periodic[1]:
        candidate_mask[:, -1] = False
    seeds = np.argwhere(candidate_mask)
    seeds = seeds[np.argsort(gradient_norm[tuple(seeds.T)], kind="stable")]
    candidate_count = len(seeds)
    if np.max(gradient_norm) == 0.0:
        seeds = seeds[:1]
        candidate_count = 1
    candidates: list[FluxCriticalPoint] = []
    rejected: list[FluxCriticalPoint] = []
    refined_batch = None
    if backend == "jax":
        refined_batch = interpolator.refine_many(
            np.asarray(lower) + seeds[:max_candidates] * np.asarray([dx, dy]),
            spacing=(dx, dy),
            fraction=max_refinement_fraction,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    for seed_number, (i, j) in enumerate(seeds[:max_candidates]):
        position = np.asarray(lower) + np.asarray([i * dx, j * dy])
        iterations = 0
        for iteration in range(max_iterations if refine and refined_batch is None else 0):
            _, gradient, hessian = evaluate(position)
            if np.linalg.norm(gradient) == 0.0:
                break
            try:
                delta = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(gradient) <= tolerance and np.max(np.abs(delta)) <= 1e-8 * min(
                dx, dy
            ):
                break
            delta = np.clip(
                delta,
                -max_refinement_fraction * np.asarray([dx, dy]),
                max_refinement_fraction * np.asarray([dx, dy]),
            )
            position += delta
            for axis, wrap in enumerate(periodic):
                if wrap:
                    position[axis] = lower[axis] + (position[axis] - lower[axis]) % lengths[axis]
                else:
                    position[axis] = np.clip(
                        position[axis], lower[axis], lower[axis] + lengths[axis]
                    )
            iterations = iteration + 1
        if refined_batch is not None:
            position = refined_batch[0][seed_number].copy()
            iterations = int(refined_batch[1][seed_number])
        for axis, wrap in enumerate(periodic):
            if wrap and abs(position[axis] - lower[axis] - lengths[axis]) < 1e-12 * lengths[axis]:
                position[axis] = lower[axis]
        if refined_batch is None:
            value, gradient, hessian = evaluate(position)
        else:
            value, gx, gy, hxx, hxy, hyy = refined_batch[2][seed_number]
            gradient = np.asarray([gx, gy])
            hessian = np.asarray([[hxx, hxy], [hxy, hyy]])
        eigenvalues = np.abs(np.linalg.eigvalsh(hessian))
        largest, smallest = float(eigenvalues.max()), float(eigenvalues.min())
        condition = largest / smallest if smallest > 0 else float("inf")
        determinant, trace = float(np.linalg.det(hessian)), float(np.trace(hessian))
        residual = float(np.linalg.norm(gradient))
        quality = "accepted" if refine else "unrefined"
        if largest == 0 or smallest <= determinant_rtol * largest:
            quality = "degenerate"
        elif refine and residual > tolerance:
            quality = "residual_rejected"
        elif refine and np.max(np.abs(np.linalg.solve(hessian, gradient))) > 1e-8 * min(dx, dy):
            quality = "not_converged"
        point = FluxCriticalPoint(
            "O" if determinant > 0 else "X",
            (int(i), int(j)),
            tuple(position),
            float(value),
            residual,
            determinant,
            trace,
            refine,
            iterations,
            residual,
            condition,
            quality in {"accepted", "unrefined"},
            quality,
        )
        (candidates if point.accepted else rejected).append(point)
    candidates.sort(key=lambda point: (point.gradient_norm, point.kind, point.index))
    selected: list[FluxCriticalPoint] = []
    for point in candidates:
        if all(
            point.kind != other.kind
            or _periodic_distance(point.position, other.position, lengths, periodic)
            > max(min_separation, 1e-10 * min(lengths))
            for other in selected
        ):
            selected.append(point)
        else:
            rejected.append(replace(point, accepted=False, quality="duplicate_root"))
    if min_flux_contrast_rtol:
        selected, insignificant = select_resolved_flux_points(
            selected,
            values,
            lengths=lengths,
            lower=lower,
            min_flux_contrast=min_flux_contrast_rtol * float(np.ptp(values)),
            min_island_cells=min_island_cells,
            flow_scale_cells=seed_scale_cells,
        )
        rejected.extend(insignificant)
    report = CriticalPointDetection(
        tuple(selected[:max_points]),
        tuple(rejected),
        candidate_count,
        candidate_count > max_candidates or len(selected) > max_points,
        "fourier_newton" if all(periodic) else "interpolated_newton",
    )
    return report if return_report else report.points


def _periodic_excursion_labels(mask):
    """Four-connected excursion components, including both periodic seams."""
    from scipy.ndimage import label

    labels, count = label(mask)
    parents = np.arange(count + 1)

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for side, opposite in ((labels[0], labels[-1]), (labels[:, 0], labels[:, -1])):
        for a, b in zip(side, opposite, strict=True):
            if a and b:
                parents[root(a)] = root(b)
    lookup = np.asarray([root(i) for i in range(count + 1)])
    return lookup[labels]


def _sample_periodic(field, positions, lower, spacing, *, order=3):
    from scipy.ndimage import map_coordinates

    coordinates = ((np.asarray(positions) - lower) / spacing).T
    return map_coordinates(field, coordinates, order=order, mode="grid-wrap", prefilter=False)


def _saddle_branches(point, sign, below, values, derivatives, *, lower, spacing, lengths):
    """Follow the two local descending branches into contrast-qualified basins.

    Flow derivatives use the declared discovery scale. Flux levels and component
    labels use the original mesh. A stalled branch never invents an association.
    """
    center = np.asarray(point.position).copy()
    # Start on the discovery-scale saddle, rather than starting both branches
    # on one side when the nearby original Fourier root is slightly displaced.
    for _ in range(24):
        gx, gy, hxx, hxy, hyy = [
            _sample_periodic(field, [center], lower, spacing)[0] for field in derivatives
        ]
        hessian = np.asarray([[hxx, hxy], [hxy, hyy]])
        try:
            delta = np.linalg.solve(hessian, -np.asarray([gx, gy]))
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(delta).all():
            return None
        if np.max(np.abs(delta) / spacing) < 1e-6:
            break
        center += np.clip(delta, -spacing, spacing)
        distance = np.abs(
            (center - point.position + 0.5 * np.asarray(lengths)) % lengths
            - 0.5 * np.asarray(lengths)
        )
        if np.max(distance / spacing) > 4:
            return None
    else:
        return None
    eigenvalues, vectors = np.linalg.eigh(sign * hessian)
    if not eigenvalues[0] < 0 < eigenvalues[1]:
        return None
    direction = vectors[:, 0]
    step_size = 0.5 * min(spacing)
    positions = center + np.asarray([-1, 1])[:, None] * step_size * direction
    pending, destinations = np.ones(2, bool), np.zeros(2, int)
    for _ in range(int(np.ceil(2 * sum(lengths) / step_size))):
        labels = _sample_periodic(below, positions, lower, spacing, order=0).astype(int)
        reached = pending & (labels != 0)
        destinations[reached], pending[reached] = labels[reached], False
        if not pending.any():
            return destinations, positions - point.position
        gradient = []
        for axis in (0, 1):
            offset = np.zeros(2)
            offset[axis] = 1e-3 * spacing[axis]
            gradient.append(
                (
                    _sample_periodic(values, positions + offset, lower, spacing)
                    - _sample_periodic(values, positions - offset, lower, spacing)
                )
                / (2 * offset[axis])
            )
        # Differentiate the same interpolant used for backtracking. Independently
        # interpolated spectral gradients can point uphill near a mesh boundary.
        gradient = sign * np.stack(gradient, axis=1)
        norms = np.linalg.norm(gradient, axis=1)
        delta = -step_size * gradient / np.maximum(norms[:, None], np.finfo(float).tiny)
        delta[~pending] = 0
        previous = sign * _sample_periodic(values, positions, lower, spacing)
        # Backtracking prevents a fixed step from hopping over an unresolved
        # extremum and falsely connecting it to a distant retained island.
        for _ in range(10):
            proposed = positions + delta
            current = sign * _sample_periodic(values, proposed, lower, spacing)
            uphill = pending & (current >= previous)
            if not uphill.any():
                break
            delta[uphill] *= 0.5
        if np.any(pending & (np.linalg.norm(delta, axis=1) < 1e-3 * step_size)):
            return None
        positions += delta
    return None


def select_resolved_flux_points(
    points,
    values,
    *,
    lengths,
    lower=(0.0, 0.0),
    min_flux_contrast,
    min_island_cells=4.0,
    flow_scale_cells=0.0,
):
    """Select resolved excursion basins and their connecting saddles on the saved mesh.

    This is a scale-dependent feature selection, not a claim to enumerate every
    subgrid root or to reconstruct a global separatrix. Measurements remain at
    the original Fourier roots. Small basins merge into their deeper extremum;
    an X survives only if its two local descending branches reach retained
    basins or close a periodic route. A merger elsewhere in the same flux band
    cannot validate that X. Equivalent local barriers keep their earliest saddle.
    """
    if not points:
        return [], []
    values = np.asarray(values)
    spacing = np.asarray(lengths) / values.shape
    kx, ky = [
        2 * np.pi * np.fft.fftfreq(n, d=d) for n, d in zip(values.shape, spacing, strict=True)
    ]
    spectral = np.fft.fftn(values - values.mean())
    window = np.exp(
        -0.5
        * flow_scale_cells**2
        * ((kx[:, None] * spacing[0]) ** 2 + (ky[None, :] * spacing[1]) ** 2)
    )
    spectral *= window
    from scipy.ndimage import spline_filter

    flow_values = spline_filter(np.fft.ifftn(spectral).real, mode="grid-wrap")
    x, y = kx[:, None], ky[None, :]
    derivatives = [
        spline_filter(np.fft.ifftn(multiplier * spectral).real, mode="grid-wrap")
        for multiplier in (1j * x, 1j * y, -x * x, -x * y, -y * y)
    ]
    indices = (
        np.rint((np.asarray([p.position for p in points]) - lower) / lengths * values.shape).astype(
            int
        )
        % values.shape
    )
    kept, qualities = [], {}
    for i, point in enumerate(points):
        if point.kind != "O":
            continue
        sign = np.sign(point.hessian_trace)
        curvature = 0.5 * (
            abs(point.hessian_trace)
            + np.sqrt(max(0.0, point.hessian_trace**2 - 4 * point.hessian_determinant))
        )
        radius = min_island_cells / np.sqrt(2) * max(np.asarray(lengths) / values.shape)
        # Test a contour large enough to span the requested cells: testing only
        # the minimum contrast would discard the tiny core of a large island.
        contrast = max(min_flux_contrast, 0.5 * curvature * radius**2)
        labels = _periodic_excursion_labels(sign * values <= sign * point.psi + contrast)
        component = labels[tuple(indices[i])]
        if not component:
            qualities[i] = "unresolved_island"
            continue
        others = [
            j
            for j, other in enumerate(points)
            if other.kind == "O"
            and sign * other.hessian_trace > 0
            and labels[tuple(indices[j])] == component
        ]
        best = min(others, key=lambda j: (sign * points[j].psi, points[j].position))
        basin = labels == component
        spans = [np.count_nonzero(basin.any(axis=axis)) for axis in (1, 0)]
        if best != i or any(span == size for span, size in zip(spans, values.shape, strict=True)):
            qualities[i] = "low_flux_contrast"
        elif min(spans) < min_island_cells or basin.sum() < min_island_cells**2:
            qualities[i] = "unresolved_island"
        else:
            kept.append(i)
    barriers = {}
    for i, point in enumerate(points):
        if point.kind != "X":
            continue
        for sign in (1, -1):
            extrema = [j for j in kept if points[j].hessian_trace * sign > 0]
            if not extrema:
                continue
            below = _periodic_excursion_labels(
                sign * values <= sign * point.psi - min_flux_contrast
            )
            branches = _saddle_branches(
                point,
                sign,
                below,
                flow_values,
                derivatives,
                lower=lower,
                spacing=spacing,
                lengths=lengths,
            )
            if branches is None:
                continue
            destinations, displacement = branches
            groups = {int(label): [] for label in destinations}
            for j in extrema:
                label = below[tuple(indices[j])]
                if label in groups:
                    groups[int(label)].append(j)
            if any(not group for group in groups.values()):
                continue
            opened = ()
            if len(groups) == 1:
                from scipy.ndimage import label

                # Cut the periodic component opposite this saddle. Distinct
                # local branches can reach the same O through opposite sides
                # of the periodic domain; an interior micro-saddle cannot.
                shift = np.asarray(values.shape) // 2 - indices[i]
                basin = np.roll(below == destinations[0], shift, axis=(0, 1))
                cut_labels, _ = label(basin)
                endpoints = (
                    np.rint((np.asarray(point.position) + displacement - lower) / spacing).astype(
                        int
                    )
                    + shift
                ) % values.shape
                a, b = [cut_labels[tuple(index)] for index in endpoints]
                if a and b and a != b:
                    opened = (int(np.argmax(np.abs(displacement[0] - displacement[1]))),)
            if len(groups) > 1 or opened:
                key = (sign, tuple(sorted(tuple(group) for group in groups.values())), opened)
                if key not in barriers or sign * point.psi < sign * points[barriers[key]].psi:
                    barriers[key] = i
        qualities[i] = "insignificant_saddle"
    kept = set(kept) | set(barriers.values())
    return (
        [point for i, point in enumerate(points) if i in kept],
        [
            replace(point, accepted=False, quality=qualities[i])
            for i, point in enumerate(points)
            if i not in kept
        ],
    )


def _bilinear(field, index):
    index = np.clip(index, 0, np.asarray(field.shape) - 1)
    lo = np.minimum(index.astype(int), np.asarray(field.shape) - 2)
    a, b = index - lo
    i, j = lo
    return (
        (1 - a) * (1 - b) * field[i, j]
        + a * (1 - b) * field[i + 1, j]
        + (1 - a) * b * field[i, j + 1]
        + a * b * field[i + 1, j + 1]
    )


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
    """Associate in canonical order; the stateful tracker also exposes events."""
    tracker = CriticalPointTracker(lengths, periodic, max_link_distance)
    tracked = []
    for frame in frames:
        tracked.extend(tracker.update(frame))
    return tuple(tracked)


class CriticalPointTracker:
    """Order-independent bounded links, explicit events, and JSON restart state."""

    def __init__(
        self,
        lengths=(2 * np.pi,) * 2,
        periodic=(True, True),
        max_link_distance=None,
        ambiguity_distance=None,
    ):
        if len(lengths) != 2 or min(lengths) <= 0 or not np.isfinite(lengths).all():
            raise ValueError("lengths must contain two positive domain lengths")
        if max_link_distance is not None and max_link_distance < 0:
            raise ValueError("max_link_distance must be non-negative")
        if ambiguity_distance is not None and ambiguity_distance < 0:
            raise ValueError("ambiguity_distance must be non-negative")
        self.lengths, self.periodic = tuple(lengths), tuple(periodic)
        self.max_link_distance = (
            0.25 * min(lengths) if max_link_distance is None else max_link_distance
        )
        self.ambiguity_distance = (
            1e-6 * min(lengths) if ambiguity_distance is None else ambiguity_distance
        )
        self.previous = ()
        self.next_track_id = 0
        self.frame = -1
        self.time = None
        self.events = []

    def update(self, points, *, time=None):
        self.frame += 1
        self.time = time
        self.events = []
        points = sorted(
            (point for point in points if point.accepted),
            key=lambda p: (p.kind, tuple(np.asarray(p.position) % self.lengths), p.hessian_trace),
        )
        edges = []
        for j, point in enumerate(points):
            for i, (track_id, old) in enumerate(self.previous):
                if point.kind != old.kind or (
                    point.kind == "O" and point.hessian_trace * old.hessian_trace < 0
                ):
                    continue
                distance = _periodic_distance(
                    point.position, old.position, self.lengths, self.periodic
                )
                if distance <= self.max_link_distance:
                    edges.append((distance, track_id, i, j))
        ambiguous = set()
        for axis in (2, 3):
            for index in {edge[axis] for edge in edges}:
                choices = sorted(edge for edge in edges if edge[axis] == index)
                if len(choices) > 1 and choices[1][0] - choices[0][0] <= self.ambiguity_distance:
                    ambiguous.update(edge[3] for edge in choices[:2])
        links, used = {}, set()
        n, m = len(self.previous), len(points)
        if n and m:
            # Dummy rows/columns permit births and deaths. Prefer the greatest
            # number of valid links, then minimum total displacement.
            penalty = float(n + m + 1)
            costs = np.zeros((n + m, n + m))
            costs[:n, :m] = 1e6 * penalty
            costs[:n, m:] = penalty
            costs[n:, :m] = penalty
            for distance, _, i, j in edges:
                if j not in ambiguous:
                    costs[i, j] = distance / max(self.max_link_distance, np.finfo(float).tiny)
            for i, j in enumerate(_minimum_assignment(costs)[:n]):
                if j < m and costs[i, j] <= 1.0:
                    links[j] = self.previous[i][0]
                    used.add(i)
        current = []
        for j, point in enumerate(points):
            if j in links:
                track_id = links[j]
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.events.append(
                    dict(
                        frame=self.frame,
                        time=time,
                        track_id=track_id,
                        kind=point.kind,
                        event="birth",
                    )
                )
            if j in ambiguous:
                self.events.append(
                    dict(
                        frame=self.frame,
                        time=time,
                        track_id=track_id,
                        kind=point.kind,
                        event="ambiguity",
                    )
                )
            current.append((track_id, point))
        for i, (track_id, point) in enumerate(self.previous):
            if i not in used:
                self.events.append(
                    dict(
                        frame=self.frame,
                        time=time,
                        track_id=track_id,
                        kind=point.kind,
                        event="disappearance",
                    )
                )
        self.previous = tuple(current)
        return self.previous

    def state(self):
        return dict(
            lengths=self.lengths,
            periodic=self.periodic,
            max_link_distance=self.max_link_distance,
            ambiguity_distance=self.ambiguity_distance,
            next_track_id=self.next_track_id,
            frame=self.frame,
            time=self.time,
            previous=[dict(track_id=i, point=asdict(p)) for i, p in self.previous],
        )

    @classmethod
    def from_state(cls, state):
        tracker = cls(
            state["lengths"],
            state["periodic"],
            state["max_link_distance"],
            state["ambiguity_distance"],
        )
        tracker.next_track_id, tracker.frame = state["next_track_id"], state["frame"]
        tracker.time = state.get("time")
        tracker.previous = tuple(
            (item["track_id"], FluxCriticalPoint(**item["point"])) for item in state["previous"]
        )
        return tracker


def _minimum_assignment(costs):
    """Square Hungarian assignment using NumPy, without a SciPy dependency."""
    n = len(costs)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)
    for row in range(1, n + 1):
        p[0], column = row, 0
        minimum, visited = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
        while True:
            visited[column] = True
            active = p[column]
            available = ~visited[1:]
            reduced = costs[active - 1] - u[active] - v[1:]
            improved = available & (reduced < minimum[1:])
            minimum[1:][improved] = reduced[improved]
            way[1:][improved] = column
            next_column = int(np.argmin(np.where(available, minimum[1:], np.inf))) + 1
            delta = minimum[next_column]
            u[p[visited]] += delta
            v[visited] -= delta
            minimum[~visited] -= delta
            column = next_column
            if p[column] == 0:
                break
        while column:
            previous = way[column]
            p[column] = p[previous]
            column = previous
    assignment = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def _first_derivative(
    values: np.ndarray,
    *,
    axis: int,
    spacing: float,
    periodic: bool,
) -> np.ndarray:
    if periodic:
        return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing)
    return np.gradient(values, spacing, axis=axis, edge_order=2)


def _second_derivative(
    values: np.ndarray,
    *,
    axis: int,
    spacing: float,
    periodic: bool,
) -> np.ndarray:
    if periodic:
        return (np.roll(values, -1, axis=axis) - 2.0 * values + np.roll(values, 1, axis=axis)) / (
            spacing * spacing
        )
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
            delta %= length
            delta = min(delta, length - delta)
        deltas.append(delta)
    return float(np.hypot(deltas[0], deltas[1]))
