"""Finite-difference Harris-sheet tearing eigenvalue validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mhx.io import write_manifest
from mhx.plotting import plot_linear_tearing_eigenvalue_validation

LINEAR_TEARING_EIGENVALUE_SCHEMA = "mhx.validation.linear_tearing_eigenvalue.v1"


@dataclass(frozen=True)
class LinearTearingEigenvalueResult:
    """Direct reduced-MHD Harris-sheet tearing eigenvalue validation artifacts."""

    grid_points: np.ndarray
    dx: np.ndarray
    growth_rates: np.ndarray
    fitted_growth_rates: np.ndarray
    extrapolated_growth_rate: float
    reference_growth_rate: float
    selected_grid_points: int
    selected_eigenvalue: complex
    selected_residual_norm: float
    selected_relative_growth_error: float
    extrapolated_relative_growth_error: float
    stable_control_wavenumber: float
    stable_control_max_real_part: float
    stable_control_residual_norm: float
    flux_even_correlation: float
    stream_odd_correlation: float
    coordinate: np.ndarray
    selected_spectrum: np.ndarray
    flux_eigenfunction: np.ndarray
    streamfunction_imag: np.ndarray
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class _EigenSolve:
    grid_points: int
    dx: float
    coordinate: np.ndarray
    eigenvalues: np.ndarray
    selected_eigenvalue: complex
    selected_eigenvector: np.ndarray
    residual_norm: float


def run_linear_tearing_eigenvalue_validation(
    *,
    grid_points: tuple[int, ...] = (192, 256, 320),
    half_width: float = 10.0,
    lundquist: float = 1000.0,
    wavenumber: float = 0.5,
    reference_growth_rate: float = 0.0131,
    stable_control_wavenumber: float = 1.2,
    stable_control_grid_points: int = 192,
    max_selected_relative_growth_error: float = 3.0e-2,
    max_extrapolated_relative_growth_error: float = 1.0e-2,
    max_stable_control_real_part: float = 0.0,
    max_residual_norm: float = 1.0e-10,
    max_imag_abs: float = 1.0e-10,
    min_parity_correlation: float = 0.995,
) -> LinearTearingEigenvalueResult:
    r"""Solve a direct 1D Harris-sheet tearing eigenproblem and gate it.

    The benchmark discretizes the inviscid linear reduced-MHD tearing equations
    for ``B_y=tanh(x)`` with conducting/no-slip Dirichlet perturbation
    boundaries on ``[-d, d]``:

    ``σ(Du) = i k B Db - i k B'' b`` and
    ``σ b = i k B u + S^{-1}Db``, where ``D=d²/dx²-k²``.

    It gates the unstable eigenvalue at ``S=1000``, ``k=0.5``, and ``d=10``
    against the published reduced-MHD reference value ``γ ≈ 0.0131`` while also
    checking second-order grid extrapolation, residuals, and tearing parity.
    """
    grid_point_values = np.asarray(grid_points, dtype=int)
    _validate_inputs(
        grid_point_values,
        half_width=half_width,
        lundquist=lundquist,
        wavenumber=wavenumber,
        reference_growth_rate=reference_growth_rate,
        stable_control_wavenumber=stable_control_wavenumber,
        stable_control_grid_points=stable_control_grid_points,
    )
    solves = tuple(
        _solve_harris_tearing_eigenproblem(
            int(points),
            half_width=half_width,
            lundquist=lundquist,
            wavenumber=wavenumber,
        )
        for points in grid_point_values
    )
    dx = np.asarray([solve.dx for solve in solves])
    growth_rates = np.asarray([solve.selected_eigenvalue.real for solve in solves])
    extrapolated_growth_rate, fitted_growth_rates = _fit_second_order_limit(dx, growth_rates)
    selected = solves[-1]
    selected_growth_rate = float(selected.selected_eigenvalue.real)
    selected_relative_growth_error = _relative_error(
        selected_growth_rate,
        reference_growth_rate,
    )
    extrapolated_relative_growth_error = _relative_error(
        extrapolated_growth_rate,
        reference_growth_rate,
    )
    flux, stream_imag, flux_even, stream_odd = _aligned_tearing_parity(
        selected.selected_eigenvector,
        selected.grid_points,
    )
    stable_control = _solve_harris_tearing_eigenproblem(
        stable_control_grid_points,
        half_width=half_width,
        lundquist=lundquist,
        wavenumber=stable_control_wavenumber,
    )
    stable_control_max_real_part = float(np.max(stable_control.eigenvalues.real))

    checks = {
        "finite_positive_growth_rates": bool(
            np.all(np.isfinite(growth_rates)) and np.all(growth_rates > 0.0)
        ),
        "grid_refinement_decreases_growth_rate": bool(np.all(np.diff(growth_rates) < 0.0)),
        "selected_growth_matches_reference": bool(
            selected_relative_growth_error <= max_selected_relative_growth_error
        ),
        "extrapolated_growth_matches_reference": bool(
            extrapolated_relative_growth_error <= max_extrapolated_relative_growth_error
        ),
        "stable_control_has_no_positive_growth": bool(
            stable_control_max_real_part <= max_stable_control_real_part
        ),
        "selected_eigenvalue_is_real": bool(abs(selected.selected_eigenvalue.imag) <= max_imag_abs),
        "selected_residual_small": bool(selected.residual_norm <= max_residual_norm),
        "stable_control_residual_small": bool(stable_control.residual_norm <= max_residual_norm),
        "flux_eigenfunction_is_even": bool(flux_even >= min_parity_correlation),
        "streamfunction_eigenfunction_is_odd": bool(stream_odd >= min_parity_correlation),
    }
    diagnostics = {
        "schema": LINEAR_TEARING_EIGENVALUE_SCHEMA,
        "equilibrium": "B_y = tanh(x/a), a = 1",
        "equations": [
            "sigma (d2/dx2 - k^2) u = i k B (d2/dx2 - k^2) b - i k B'' b",
            "sigma b = i k B u + S^{-1} (d2/dx2 - k^2) b",
        ],
        "boundary_conditions": "u=b=0 at x=+-d",
        "half_width": half_width,
        "lundquist": lundquist,
        "wavenumber": wavenumber,
        "grid_points": grid_point_values.tolist(),
        "dx": dx.tolist(),
        "growth_rates": growth_rates.tolist(),
        "fitted_growth_rates": fitted_growth_rates.tolist(),
        "extrapolated_growth_rate": extrapolated_growth_rate,
        "reference_growth_rate": reference_growth_rate,
        "stable_control_wavenumber": stable_control_wavenumber,
        "stable_control_grid_points": stable_control_grid_points,
        "stable_control_max_real_part": stable_control_max_real_part,
        "stable_control_selected_eigenvalue": {
            "real": float(stable_control.selected_eigenvalue.real),
            "imag": float(stable_control.selected_eigenvalue.imag),
        },
        "stable_control_residual_norm": stable_control.residual_norm,
        "selected_grid_points": selected.grid_points,
        "selected_eigenvalue": {
            "real": selected_growth_rate,
            "imag": float(selected.selected_eigenvalue.imag),
        },
        "selected_relative_growth_error": selected_relative_growth_error,
        "extrapolated_relative_growth_error": extrapolated_relative_growth_error,
        "selected_residual_norm": selected.residual_norm,
        "flux_even_correlation": flux_even,
        "streamfunction_odd_correlation": stream_odd,
        "references": {
            "fkr": (
                "Furth, Killeen & Rosenbluth 1963 established the classical "
                "resistive tearing scaling for large S."
            ),
            "mactaggart_2019": (
                "MacTaggart 2019, The tearing instability of resistive "
                "magnetohydrodynamics, solves the same reduced-MHD normal-mode "
                "problem and reports gamma approximately 0.0131 for S=1000, k=0.5, d=10."
            ),
            "mactaggart_stewart_2017": (
                "MacTaggart & Stewart 2017 use the same discrete generalized "
                "eigenvalue setup and identify a unique tearing eigenvalue near 0.0131."
            ),
        },
    }
    validation = {
        "schema": "mhx.validation.linear_tearing_eigenvalue.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_selected_relative_growth_error": max_selected_relative_growth_error,
            "max_extrapolated_relative_growth_error": max_extrapolated_relative_growth_error,
            "max_stable_control_real_part": max_stable_control_real_part,
            "max_residual_norm": max_residual_norm,
            "max_imag_abs": max_imag_abs,
            "min_parity_correlation": min_parity_correlation,
        },
        "diagnostics": diagnostics,
    }
    return LinearTearingEigenvalueResult(
        grid_points=grid_point_values,
        dx=dx,
        growth_rates=growth_rates,
        fitted_growth_rates=fitted_growth_rates,
        extrapolated_growth_rate=extrapolated_growth_rate,
        reference_growth_rate=reference_growth_rate,
        selected_grid_points=selected.grid_points,
        selected_eigenvalue=selected.selected_eigenvalue,
        selected_residual_norm=selected.residual_norm,
        selected_relative_growth_error=selected_relative_growth_error,
        extrapolated_relative_growth_error=extrapolated_relative_growth_error,
        stable_control_wavenumber=stable_control_wavenumber,
        stable_control_max_real_part=stable_control_max_real_part,
        stable_control_residual_norm=stable_control.residual_norm,
        flux_even_correlation=flux_even,
        stream_odd_correlation=stream_odd,
        coordinate=selected.coordinate,
        selected_spectrum=selected.eigenvalues,
        flux_eigenfunction=flux,
        streamfunction_imag=stream_imag,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_linear_tearing_eigenvalue_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Write direct Harris-sheet tearing eigenvalue artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_linear_tearing_eigenvalue_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "linear_tearing_eigenvalue.npz"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        history_path,
        schema=LINEAR_TEARING_EIGENVALUE_SCHEMA,
        grid_points=result.grid_points,
        dx=result.dx,
        growth_rates=result.growth_rates,
        fitted_growth_rates=result.fitted_growth_rates,
        extrapolated_growth_rate=result.extrapolated_growth_rate,
        reference_growth_rate=result.reference_growth_rate,
        selected_grid_points=result.selected_grid_points,
        selected_eigenvalue_real=result.selected_eigenvalue.real,
        selected_eigenvalue_imag=result.selected_eigenvalue.imag,
        selected_residual_norm=result.selected_residual_norm,
        selected_relative_growth_error=result.selected_relative_growth_error,
        extrapolated_relative_growth_error=result.extrapolated_relative_growth_error,
        stable_control_wavenumber=result.stable_control_wavenumber,
        stable_control_max_real_part=result.stable_control_max_real_part,
        stable_control_residual_norm=result.stable_control_residual_norm,
        flux_even_correlation=result.flux_even_correlation,
        stream_odd_correlation=result.stream_odd_correlation,
        coordinate=result.coordinate,
        spectrum_real=result.selected_spectrum.real,
        spectrum_imag=result.selected_spectrum.imag,
        flux_eigenfunction=result.flux_eigenfunction,
        streamfunction_imag=result.streamfunction_imag,
    )

    figure_path = plot_linear_tearing_eigenvalue_validation(
        result.dx,
        result.growth_rates,
        result.fitted_growth_rates,
        reference_growth_rate=result.reference_growth_rate,
        extrapolated_growth_rate=result.extrapolated_growth_rate,
        spectrum=result.selected_spectrum,
        selected_eigenvalue=result.selected_eigenvalue,
        coordinate=result.coordinate,
        flux_eigenfunction=result.flux_eigenfunction,
        streamfunction_imag=result.streamfunction_imag,
        path=output_dir / "figures" / "linear_tearing_eigenvalue.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "linear_tearing_eigenvalue": str(figure_path.relative_to(output_dir)),
        },
    )
    return manifest_path, result.validation


def _validate_inputs(
    grid_points: np.ndarray,
    *,
    half_width: float,
    lundquist: float,
    wavenumber: float,
    reference_growth_rate: float,
    stable_control_wavenumber: float,
    stable_control_grid_points: int,
) -> None:
    if grid_points.ndim != 1 or grid_points.shape[0] < 3:
        raise ValueError("at least three grid-point samples are required")
    if np.any(grid_points < 64):
        raise ValueError("grid-point samples must be at least 64")
    if np.any(np.diff(grid_points) <= 0):
        raise ValueError("grid-point samples must be strictly increasing")
    if half_width <= 1.0:
        raise ValueError("half_width must be greater than 1")
    if lundquist <= 0.0:
        raise ValueError("lundquist must be positive")
    if wavenumber <= 0.0 or wavenumber >= 1.0:
        raise ValueError("wavenumber must satisfy 0 < k < 1 for this unstable Harris gate")
    if reference_growth_rate <= 0.0:
        raise ValueError("reference_growth_rate must be positive")
    if stable_control_wavenumber <= 1.0:
        raise ValueError("stable_control_wavenumber must be greater than 1")
    if stable_control_grid_points < 64:
        raise ValueError("stable_control_grid_points must be at least 64")


def _solve_harris_tearing_eigenproblem(
    grid_points: int,
    *,
    half_width: float,
    lundquist: float,
    wavenumber: float,
) -> _EigenSolve:
    coordinate = np.linspace(-half_width, half_width, grid_points + 2, dtype=float)[1:-1]
    dx = float(coordinate[1] - coordinate[0])
    derivative = _second_derivative_minus_k_squared(grid_points, dx, wavenumber)
    magnetic_field = np.tanh(coordinate)
    magnetic_field_second_derivative = -2.0 * np.tanh(coordinate) / np.cosh(coordinate) ** 2
    coupling = 1j * wavenumber * (
        np.diag(magnetic_field) @ derivative - np.diag(magnetic_field_second_derivative)
    )
    operator = np.zeros((2 * grid_points, 2 * grid_points), dtype=np.complex128)
    operator[:grid_points, grid_points:] = np.linalg.solve(derivative, coupling)
    operator[grid_points:, :grid_points] = 1j * wavenumber * np.diag(magnetic_field)
    operator[grid_points:, grid_points:] = (1.0 / lundquist) * derivative
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    selected_index = int(np.argmax(eigenvalues.real))
    selected_eigenvalue = complex(eigenvalues[selected_index])
    selected_eigenvector = np.asarray(eigenvectors[:, selected_index])
    residual_norm = float(
        np.linalg.norm(operator @ selected_eigenvector - selected_eigenvalue * selected_eigenvector)
        / np.linalg.norm(selected_eigenvector)
    )
    return _EigenSolve(
        grid_points=grid_points,
        dx=dx,
        coordinate=coordinate,
        eigenvalues=eigenvalues,
        selected_eigenvalue=selected_eigenvalue,
        selected_eigenvector=selected_eigenvector,
        residual_norm=residual_norm,
    )


def _second_derivative_minus_k_squared(
    grid_points: int,
    dx: float,
    wavenumber: float,
) -> np.ndarray:
    main = (-2.0 / dx**2 - wavenumber**2) * np.ones(grid_points)
    off = (1.0 / dx**2) * np.ones(grid_points - 1)
    return np.diag(main) + np.diag(off, 1) + np.diag(off, -1)


def _fit_second_order_limit(dx: np.ndarray, growth_rates: np.ndarray) -> tuple[float, np.ndarray]:
    coefficients = np.polyfit(dx**2, growth_rates, deg=1)
    fitted = np.polyval(coefficients, dx**2)
    return float(coefficients[1]), np.asarray(fitted)


def _relative_error(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(abs(reference), 1.0e-300))


def _aligned_tearing_parity(eigenvector: np.ndarray, grid_points: int) -> tuple[np.ndarray, ...]:
    streamfunction = np.asarray(eigenvector[:grid_points])
    flux = np.asarray(eigenvector[grid_points:])
    alignment_index = int(np.argmax(np.abs(flux)))
    phase = np.exp(-1j * np.angle(flux[alignment_index]))
    aligned_flux = flux * phase
    aligned_stream = streamfunction * phase
    flux_real = np.real(aligned_flux)
    stream_imag = np.imag(aligned_stream)
    flux_real = _normalize_for_plotting(flux_real)
    stream_imag = _normalize_for_plotting(stream_imag)
    flux_even = _parity_correlation(flux_real, sign=1.0)
    stream_odd = _parity_correlation(stream_imag, sign=-1.0)
    return flux_real, stream_imag, flux_even, stream_odd


def _normalize_for_plotting(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return values
    return values / scale


def _parity_correlation(values: np.ndarray, *, sign: float) -> float:
    reflected = sign * values[::-1]
    denominator = np.linalg.norm(values) * np.linalg.norm(reflected)
    if denominator == 0.0:
        return 0.0
    return float(np.vdot(values, reflected).real / denominator)
