r"""Sweet--Parker current-sheet diagnostics for two-dimensional reduced MHD.

Conventions follow :func:`mhx.equations.reduced_mhd.reduced_mhd_rhs`:
``j_z = -∇²ψ``, ``B = ẑ × ∇ψ`` (``B_x = -ψ_y``, ``B_y = ψ_x``) and
``v = ẑ × ∇φ`` with ``∇²φ = ω``. Density is one, so ``v_A = |B|``.

At an X-point ``∇ψ = 0``, so ``[φ, ψ] = 0`` and the reconnection rate is
exactly ``∂ψ_X/∂t = η∇²ψ_X = -η j_X``.

The measurement assumes the sheet is aligned with the ``y`` axis (inflow along
``x``), as for the periodic double-Harris equilibrium; the Hessian tilt is
reported so callers can gate on that assumption.

X-points and O-points are located on the sheet centreline (the ``|B_y|``
minimum near the sheet in each column): X-points are the extrema of ψ whose
curvature is opposite to the cross-sheet curvature, O-points the others.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SweetParkerSheetMeasurement:
    """Local geometry and flows of one reconnecting current sheet.

    Lengths are half-widths / half-lengths measured from the X-point.
    """

    x_point: tuple[float, float]
    current_x: float
    reconnection_rate: float
    delta: float
    length: float
    length_current: float
    b_upstream: float
    v_inflow: float
    v_outflow: float
    opening_angle_deg: float
    tilt_deg: float
    sheet_x_point_count: int
    island_width: float
    delta_over_dx: float

    @property
    def lundquist(self) -> float:
        """Return ``S_L = L B_up / η`` implied by the stored rate and current."""
        eta = self.reconnection_rate / max(abs(self.current_x), 1.0e-300)
        return self.length * self.b_upstream / max(eta, 1.0e-300)

    @property
    def normalized_rate(self) -> float:
        """Return ``E* = η |j_X| / B_up²``."""
        return self.reconnection_rate / max(self.b_upstream**2, 1.0e-300)

    def as_dict(self) -> dict[str, float]:
        """Return a flat JSON-friendly dictionary."""
        values = asdict(self)
        x_point = values.pop("x_point")
        values["x_point_x"], values["x_point_y"] = x_point
        values["lundquist"] = self.lundquist
        values["normalized_rate"] = self.normalized_rate
        return values


def sweet_parker_prediction(
    *,
    resistivity: float,
    viscosity: float,
    length: float,
    b_upstream: float,
) -> dict[str, float]:
    r"""Return Sweet--Parker predictions with the Park et al. (1984) Pm factors.

    ``δ/L = S_L^{-1/2}(1+Pm)^{1/4}``, ``v_in/v_A = S_L^{-1/2}(1+Pm)^{-1/4}``,
    ``v_out/v_A = (1+Pm)^{-1/2}`` and ``E/(B v_A) = S_L^{-1/2}(1+Pm)^{-1/4}``.
    """
    if resistivity <= 0.0 or viscosity < 0.0 or length <= 0.0 or b_upstream <= 0.0:
        raise ValueError("resistivity, length and b_upstream must be positive")
    prandtl = viscosity / resistivity
    lundquist = length * b_upstream / resistivity
    root_s = np.sqrt(lundquist)
    return {
        "lundquist": float(lundquist),
        "magnetic_prandtl": float(prandtl),
        "delta": float(length * (1.0 + prandtl) ** 0.25 / root_s),
        "v_inflow": float(b_upstream * (1.0 + prandtl) ** -0.25 / root_s),
        "v_outflow": float(b_upstream * (1.0 + prandtl) ** -0.5),
        "normalized_rate": float((1.0 + prandtl) ** -0.25 / root_s),
    }


def measure_sweet_parker_sheet(
    psi: np.ndarray,
    omega: np.ndarray,
    *,
    lengths: tuple[float, float],
    resistivity: float,
    sheet_x: float,
    sheet_half_window: float | None = None,
) -> SweetParkerSheetMeasurement:
    """Measure the current sheet whose centre lies near ``x = sheet_x``.

    The principal X-point is the largest centreline extremum of ``s ψ``, with
    ``s`` the sign of the cross-sheet curvature; ``sheet_x_point_count`` counts
    the prominent centreline X-points. ``island_width`` is the full ``x``
    extent, through the principal O-point, of the region inside the X-point
    separatrix.
    """
    psi = np.asarray(psi, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if psi.ndim != 2 or psi.shape != omega.shape:
        raise ValueError("psi and omega must be two-dimensional arrays of equal shape")
    nx, ny = psi.shape
    dx, dy = lengths[0] / nx, lengths[1] / ny
    if sheet_half_window is None:
        sheet_half_window = 0.125 * lengths[0]

    fields = _spectral_fields(psi, omega, lengths=lengths)
    current = fields["current"]

    in_reach = max(1, int(sheet_half_window / dx))
    rows = (round(sheet_x / dx) + np.arange(-in_reach, in_reach + 1)) % nx
    columns = np.arange(ny)
    centre_rows = rows[np.argmin(np.abs(fields["b_y"][rows, :]), axis=0)]
    cross_sign = np.sign(np.mean(fields["psi_xx"][centre_rows, columns]))
    signed_centre = cross_sign * psi[centre_rows, columns]
    iy = int(np.argmax(signed_centre))
    ix = int(centre_rows[iy])
    x_point = (ix * dx, iy * dy)
    sheet_x_point_count = _count_prominent_maxima(signed_centre)
    iy_o = int(np.argmin(signed_centre))
    signed_column = cross_sign * (psi[:, iy_o] - psi[ix, iy])
    island_width = dx * sum(
        _level_crossing_distance(signed_column, int(centre_rows[iy_o]), sign, nx // 4)
        for sign in (1, -1)
    )
    current_x = float(current[ix, iy])

    # Inflow line (vary x at fixed iy) and sheet centreline (vary y at fixed ix).
    current_in = np.abs(current[:, iy])
    b_y_in = np.abs(fields["b_y"][:, iy])
    v_x_in = np.abs(fields["v_x"][:, iy])
    current_along = np.abs(current[ix, :])
    v_y_along = np.abs(fields["v_y"][ix, :])

    along_reach = max(1, ny // 2 - 1)
    half = 0.5 * abs(current_x)
    delta_sides = [
        _half_level_distance(current_in, ix, sign, half, in_reach) * dx for sign in (1, -1)
    ]
    delta = float(np.mean(delta_sides))
    length_current_sides = [
        _half_level_distance(current_along, iy, sign, half, along_reach) * dy for sign in (1, -1)
    ]
    length_current = float(np.mean(length_current_sides))

    outflow_steps = [_argmax_distance(v_y_along, iy, sign, along_reach) for sign in (1, -1)]
    length = float(np.mean(outflow_steps) * dy)
    v_outflow = float(
        np.mean([v_y_along[(iy + s * k) % ny] for s, k in zip((1, -1), outflow_steps, strict=True)])
    )

    # Upstream state: average over a band 2δ–4δ from the X-point on both sides of
    # the sheet, so it follows the sheet width rather than hopping between
    # local maxima of |B_y|.
    b_upstream = v_inflow = float("nan")
    if np.isfinite(delta):
        inner = min(in_reach, max(1, round(2.0 * delta / dx)))
        outer = min(in_reach, max(inner + 1, round(4.0 * delta / dx)))
        steps = np.arange(inner, outer + 1)
        upstream = np.concatenate(((ix + steps) % nx, (ix - steps) % nx))
        b_upstream = float(np.mean(b_y_in[upstream]))
        v_inflow = float(np.mean(v_x_in[upstream]))

    hessian = np.array(
        [
            [fields["psi_xx"][ix, iy], fields["psi_xy"][ix, iy]],
            [fields["psi_xy"][ix, iy], fields["psi_yy"][ix, iy]],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(np.abs(eigenvalues))
    small, large = np.abs(eigenvalues[order[0]]), np.abs(eigenvalues[order[1]])
    opening_angle = float(2.0 * np.degrees(np.arctan(np.sqrt(small / max(large, 1.0e-300)))))
    inflow_axis = eigenvectors[:, order[1]]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(inflow_axis[0])))))

    return SweetParkerSheetMeasurement(
        x_point=(float(x_point[0]), float(x_point[1])),
        current_x=current_x,
        reconnection_rate=float(resistivity * abs(current_x)),
        delta=delta,
        length=length,
        length_current=length_current,
        b_upstream=b_upstream,
        v_inflow=v_inflow,
        v_outflow=v_outflow,
        opening_angle_deg=opening_angle,
        tilt_deg=tilt,
        sheet_x_point_count=sheet_x_point_count,
        island_width=float(island_width),
        delta_over_dx=float(delta / dx),
    )


def select_steady_window(
    times: np.ndarray,
    rate: np.ndarray,
    *,
    min_duration: float,
    max_cv: float = 0.1,
    constraints: tuple[np.ndarray, ...] = (),
    max_drift: float = 0.1,
) -> tuple[float, float, float] | None:
    """Return ``(t_start, t_end, cv)`` of the flattest quasi-steady window of ``rate``.

    Each candidate window is the shortest one starting at a saved sample that
    spans at least ``min_duration``. A candidate counts only if ``rate`` has a
    coefficient of variation of at most ``max_cv`` and every series in
    ``constraints`` (e.g. sheet width, length, upstream field) changes by at
    most ``max_drift`` in total, ``(max - min)/|mean|``, inside it. The total
    change (not the coefficient of variation, which is ~3.5x smaller for a
    linear drift) is what rejects a turning point of ``rate`` while the sheet
    is still evolving. Among the candidates, the window with the smallest
    ``rate`` variation is returned.
    """
    times = np.asarray(times, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    series = tuple(np.asarray(values, dtype=np.float64) for values in constraints)
    if times.ndim != 1 or times.shape != rate.shape:
        raise ValueError("times and rate must be one-dimensional arrays of equal length")
    if any(values.shape != times.shape for values in series):
        raise ValueError("constraints must match the shape of times")
    if min_duration <= 0.0:
        raise ValueError("min_duration must be positive")
    best: tuple[float, float, float] | None = None
    for start in range(times.size):
        stop = int(np.searchsorted(times, times[start] + min_duration))
        if stop >= times.size:
            break
        cv = _coefficient_of_variation(rate[start : stop + 1])
        if cv > max_cv or any(
            _relative_drift(values[start : stop + 1]) > max_drift for values in series
        ):
            continue
        if best is None or cv < best[2]:
            best = (float(times[start]), float(times[stop]), cv)
    return best


def _relative_drift(values: np.ndarray) -> float:
    """Return ``(max - min)/|mean|``, or infinity for non-finite or zero-mean samples."""
    mean = float(np.mean(values))
    if not np.all(np.isfinite(values)) or mean == 0.0:
        return float("inf")
    return float((np.max(values) - np.min(values)) / abs(mean))


def _coefficient_of_variation(values: np.ndarray) -> float:
    """Return ``std/|mean|``, or infinity for non-finite or zero-mean samples."""
    mean = float(np.mean(values))
    if not np.all(np.isfinite(values)) or mean == 0.0:
        return float("inf")
    return float(np.std(values) / abs(mean))


def _spectral_fields(
    psi: np.ndarray,
    omega: np.ndarray,
    *,
    lengths: tuple[float, float],
) -> dict[str, np.ndarray]:
    nx, ny = psi.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lengths[0] / nx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=lengths[1] / ny)[None, :]
    k_squared = kx**2 + ky**2
    inverse_k_squared = np.divide(1.0, k_squared, out=np.zeros_like(k_squared), where=k_squared > 0)
    psi_hat = np.fft.fft2(psi)
    phi_hat = -np.fft.fft2(omega) * inverse_k_squared

    def real(field_hat: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(field_hat).real

    return {
        "current": real(k_squared * psi_hat),
        "b_y": real(1j * kx * psi_hat),
        "v_x": real(-1j * ky * phi_hat),
        "v_y": real(1j * kx * phi_hat),
        "psi_xx": real(-(kx**2) * psi_hat),
        "psi_yy": real(-(ky**2) * psi_hat),
        "psi_xy": real(-kx * ky * psi_hat),
    }


def _periodic_offset(offset: float, length: float) -> float:
    return abs((offset + 0.5 * length) % length - 0.5 * length)


def _half_level_distance(
    profile: np.ndarray, start: int, sign: int, level: float, reach: int
) -> float:
    """Return the interpolated index distance where ``profile`` first drops below ``level``."""
    size = profile.size
    previous = profile[start]
    for step in range(1, reach + 1):
        value = profile[(start + sign * step) % size]
        if value < level:
            return step - 1 + (previous - level) / max(previous - value, 1.0e-300)
        previous = value
    return float("nan")


def _level_crossing_distance(profile: np.ndarray, start: int, sign: int, reach: int) -> float:
    """Return the interpolated index distance where a negative ``profile`` first reaches zero."""
    size = profile.size
    previous = profile[start]
    for step in range(1, reach + 1):
        value = profile[(start + sign * step) % size]
        if value >= 0.0:
            return step - 1 + (-previous) / max(value - previous, 1.0e-300)
        previous = value
    return float("nan")


def _count_prominent_maxima(values: np.ndarray, relative_tolerance: float = 1.0e-3) -> int:
    """Count local maxima of a periodic sequence, ignoring wiggles below the tolerance."""
    tolerance = relative_tolerance * max(float(np.ptp(values)), 1.0e-300)
    size = values.size
    extrema = [
        index
        for index in range(size)
        if (values[index] - values[index - 1]) * (values[(index + 1) % size] - values[index]) < 0.0
    ]
    changed = True
    while changed and len(extrema) > 2:
        changed = False
        for position in range(len(extrema)):
            here, following = extrema[position], extrema[(position + 1) % len(extrema)]
            if abs(values[here] - values[following]) < tolerance:
                for index in sorted({position, (position + 1) % len(extrema)}, reverse=True):
                    extrema.pop(index)
                changed = True
                break
    return max(1, sum(1 for index in extrema if values[index] > values[index - 1]))


def _argmax_distance(profile: np.ndarray, start: int, sign: int, reach: int) -> int:
    values = [profile[(start + sign * step) % profile.size] for step in range(reach + 1)]
    return int(np.argmax(values))
