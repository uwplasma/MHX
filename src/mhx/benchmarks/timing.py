"""Deterministic FAST timing artifacts for reviewer-facing performance tracking."""

from __future__ import annotations

import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from mhx import _version
from mhx.benchmarks.decay import run_resistive_decay_validation
from mhx.benchmarks.scaling import run_reconnection_scaling_validation
from mhx.benchmarks.tearing import run_linear_tearing_smoke
from mhx.config import MeshConfig, RunConfig, TimeConfig
from mhx.io import write_manifest
from mhx.plotting import plot_timing_summary

TIMING_BENCHMARK_SCHEMA = "mhx.benchmark.timing.v1"


@dataclass(frozen=True)
class TimingCaseResult:
    """Timing summary for one benchmark case."""

    name: str
    durations_seconds: tuple[float, ...]
    peak_tracemalloc_bytes: tuple[int, ...]
    details: dict[str, Any]

    @property
    def summary(self) -> dict[str, float]:
        """Return stable scalar summary statistics for the measured repeats."""
        durations = list(self.durations_seconds)
        peaks = list(self.peak_tracemalloc_bytes)
        return {
            "duration_seconds_min": float(min(durations)),
            "duration_seconds_median": float(statistics.median(durations)),
            "duration_seconds_max": float(max(durations)),
            "peak_tracemalloc_mib_max": float(max(peaks) / (1024.0 * 1024.0)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the timing case to JSON-compatible builtins."""
        return {
            "name": self.name,
            "durations_seconds": list(self.durations_seconds),
            "peak_tracemalloc_bytes": list(self.peak_tracemalloc_bytes),
            "summary": self.summary,
            "details": self.details,
        }


@dataclass(frozen=True)
class TimingBenchmarkResult:
    """Timing benchmark results and environment metadata."""

    cases: tuple[TimingCaseResult, ...]
    diagnostics: dict[str, Any]


def _environment_metadata() -> dict[str, str]:
    return {
        "mhx_version": _version.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "jax_platform": jax.default_backend(),
    }


def _block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _case_linear_tearing_fast() -> dict[str, Any]:
    config = RunConfig(
        mesh=MeshConfig(shape=(16, 16)),
        time=TimeConfig(t1=0.03, dt=0.01, save_every=1),
    )
    trajectory, diagnostics = run_linear_tearing_smoke(config)
    _block_until_ready(trajectory.times)
    _block_until_ready(trajectory.states.psi)
    _block_until_ready(trajectory.states.omega)
    return {
        "grid_shape": list(config.mesh.shape),
        "t1": config.time.t1,
        "dt": config.time.dt,
        "n_saved": int(trajectory.times.shape[0]),
        "final_total_energy": diagnostics["final_total_energy"],
    }


def _case_resistive_decay_fast() -> dict[str, Any]:
    result = run_resistive_decay_validation(
        shape=(16, 16),
        t1=0.05,
        dt=0.01,
        max_relative_amplitude_error=1.0e-6,
        max_relative_energy_error=2.0e-6,
    )
    _block_until_ready(result.trajectory.states.psi)
    _block_until_ready(result.numerical_amplitude)
    return {
        "grid_shape": list(result.diagnostics["shape"]),
        "t1": result.diagnostics["t1"],
        "dt": result.diagnostics["dt"],
        "passed": bool(result.validation["passed"]),
        "max_relative_amplitude_error": result.diagnostics["max_relative_amplitude_error"],
    }


def _case_reconnection_scaling() -> dict[str, Any]:
    result = run_reconnection_scaling_validation()
    return {
        "n_lundquist": int(result.lundquist.shape[0]),
        "passed": bool(result.validation["passed"]),
        "fkr_gamma_slope": result.diagnostics["slopes"]["fkr_gamma"],
        "plasmoid_gamma_slope": result.diagnostics["slopes"]["plasmoid_gamma"],
    }


def _timing_cases() -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "linear_tearing_fast": _case_linear_tearing_fast,
        "resistive_decay_fast": _case_resistive_decay_fast,
        "reconnection_scaling": _case_reconnection_scaling,
    }


def _measure_case(
    name: str,
    function: Callable[[], dict[str, Any]],
    *,
    repeats: int,
    warmups: int,
) -> TimingCaseResult:
    for _ in range(warmups):
        function()

    durations: list[float] = []
    peaks: list[int] = []
    details: dict[str, Any] = {}
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        details = function()
        duration = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        durations.append(duration)
        peaks.append(int(peak))

    return TimingCaseResult(
        name=name,
        durations_seconds=tuple(durations),
        peak_tracemalloc_bytes=tuple(peaks),
        details=details,
    )


def run_timing_benchmark(
    *,
    repeats: int = 3,
    warmups: int = 1,
    cases: tuple[str, ...] = (
        "linear_tearing_fast",
        "resistive_decay_fast",
        "reconnection_scaling",
    ),
) -> TimingBenchmarkResult:
    """Run a small timing matrix without fragile absolute pass/fail thresholds."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmups < 0:
        raise ValueError("warmups must be >= 0")
    available_cases = _timing_cases()
    unknown = sorted(set(cases) - set(available_cases))
    if unknown:
        raise ValueError(f"unknown timing case(s): {', '.join(unknown)}")

    case_results = tuple(
        _measure_case(name, available_cases[name], repeats=repeats, warmups=warmups)
        for name in cases
    )
    diagnostics = {
        "schema": TIMING_BENCHMARK_SCHEMA,
        "repeats": repeats,
        "warmups": warmups,
        "environment": _environment_metadata(),
        "cases": [case.to_dict() for case in case_results],
        "measurement_notes": [
            "Durations are wall-clock measurements for FAST artifacts only.",
            "Memory uses Python tracemalloc peak memory; it does not include accelerator memory.",
            "CI gates only file presence and finite positive timings, not absolute speed.",
        ],
    }
    return TimingBenchmarkResult(cases=case_results, diagnostics=diagnostics)


def _timing_markdown(result: TimingBenchmarkResult) -> str:
    lines = [
        "# MHX FAST timing benchmark",
        "",
        f"Schema: `{TIMING_BENCHMARK_SCHEMA}`",
        "",
        "| Case | Median wall time (s) | Min (s) | Max (s) | Peak tracemalloc (MiB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in result.cases:
        summary = case.summary
        lines.append(
            "| "
            f"{case.name} | "
            f"{summary['duration_seconds_median']:.6f} | "
            f"{summary['duration_seconds_min']:.6f} | "
            f"{summary['duration_seconds_max']:.6f} | "
            f"{summary['peak_tracemalloc_mib_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These numbers are not hard validation tolerances. They are lightweight",
            "regression artifacts for comparing changes on the same machine or CI runner.",
            "",
            "Environment:",
            "",
        ]
    )
    for key, value in result.diagnostics["environment"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def write_timing_benchmark(
    outdir: str | Path,
    *,
    repeats: int = 3,
    warmups: int = 1,
    cases: tuple[str, ...] = (
        "linear_tearing_fast",
        "resistive_decay_fast",
        "reconnection_scaling",
    ),
) -> tuple[Path, dict[str, Any]]:
    """Write FAST timing JSON, Markdown, figure, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_timing_benchmark(repeats=repeats, warmups=warmups, cases=cases)

    diagnostics_path = output_dir / "timing.json"
    markdown_path = output_dir / "timing.md"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(_timing_markdown(result), encoding="utf-8")
    figure_path = plot_timing_summary(
        [case.name for case in result.cases],
        [case.summary["duration_seconds_median"] for case in result.cases],
        [case.summary["peak_tracemalloc_mib_max"] for case in result.cases],
        path=output_dir / "figures" / "timing_summary.png",
    )
    write_manifest(
        manifest_path,
        config={
            "schema": TIMING_BENCHMARK_SCHEMA,
            "repeats": repeats,
            "warmups": warmups,
            "cases": list(cases),
            "environment": result.diagnostics["environment"],
        },
        outputs={
            "timing_json": diagnostics_path.name,
            "timing_markdown": markdown_path.name,
            "timing_summary": str(figure_path.relative_to(output_dir)),
        },
    )
    return manifest_path, result.diagnostics
