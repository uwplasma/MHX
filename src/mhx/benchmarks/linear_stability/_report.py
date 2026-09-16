"""Shared helpers for the linear-stability validation bundles."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.io import write_manifest


def _loglog_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    coefficients = np.polyfit(np.log(np.asarray(x_values)), np.log(np.asarray(y_values)), deg=1)
    return float(coefficients[0])


def _relative_error(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(abs(reference), 1.0e-300))


def _l2_norm(values) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.asarray(values) ** 2)))


def _relative_l2_error(actual, expected) -> float:
    return _l2_norm(jnp.asarray(actual) - jnp.asarray(expected)) / max(
        _l2_norm(expected),
        1.0e-300,
    )


def _fit_exponential_growth_rate(times: np.ndarray, amplitudes: np.ndarray) -> float:
    if times.size < 2:
        raise ValueError("at least two time samples are required for growth fitting")
    coefficients = np.polyfit(times, np.log(np.maximum(amplitudes, 1.0e-300)), deg=1)
    return float(coefficients[0])


def _second_derivative_minus_k_squared(
    grid_points: int,
    dx: float,
    wavenumber: float,
) -> np.ndarray:
    main = (-2.0 / dx**2 - wavenumber**2) * np.ones(grid_points)
    off = (1.0 / dx**2) * np.ones(grid_points - 1)
    return np.diag(main) + np.diag(off, 1) + np.diag(off, -1)


def _half_max_width(coordinate: np.ndarray, values: np.ndarray) -> float:
    magnitudes = np.abs(values)
    peak = float(np.max(magnitudes))
    if peak <= 0.0:
        return 0.0
    threshold = 0.5 * peak
    indices = np.flatnonzero(magnitudes >= threshold)
    if indices.size == 0:
        return 0.0
    left_index = int(indices[0])
    right_index = int(indices[-1])
    left = _threshold_crossing(
        coordinate,
        magnitudes,
        left_index - 1,
        left_index,
        threshold,
    )
    right = _threshold_crossing(
        coordinate,
        magnitudes,
        right_index,
        right_index + 1,
        threshold,
    )
    return float(max(right - left, 0.0))


def _threshold_crossing(
    coordinate: np.ndarray,
    values: np.ndarray,
    left_index: int,
    right_index: int,
    threshold: float,
) -> float:
    if left_index < 0:
        return float(coordinate[0])
    if right_index >= coordinate.size:
        return float(coordinate[-1])
    x0 = float(coordinate[left_index])
    x1 = float(coordinate[right_index])
    y0 = float(values[left_index])
    y1 = float(values[right_index])
    if y1 == y0:
        return x0
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def _matching_index(values: np.ndarray, target: float) -> int:
    matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1.0e-12))
    if matches.size != 1:
        raise ValueError("reference_wavenumber must appear exactly once in wavenumber samples")
    return int(matches[0])


def write_validation_bundle(
    outdir: str | Path,
    schema: str,
    diagnostics: dict[str, Any],
    validation: dict[str, Any],
    arrays: dict[str, Any],
    plot_fn: Callable[[Path], dict[str, str]] | None = None,
    *,
    history_filename: str = "history.npz",
    claim_level: str = "validation",
    claim_scope: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Write the standard 5-step validation bundle (JSON, NPZ, figure, manifest)."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / history_filename
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(history_path, schema=schema, **arrays)
    outputs = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "history": history_path.name,
    }
    if plot_fn is not None:
        figure_outputs = plot_fn(output_dir)
        outputs.update(figure_outputs)
    write_manifest(
        manifest_path,
        config=diagnostics,
        outputs=outputs,
        claim_level=claim_level,
        claim_scope=claim_scope,
    )
    return manifest_path, validation
