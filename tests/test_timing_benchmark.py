from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mhx.benchmarks import (
    TIMING_BENCHMARK_SCHEMA,
    run_timing_benchmark,
    write_timing_benchmark,
)
from mhx.cli.main import app


def test_timing_benchmark_reports_positive_finite_measurements() -> None:
    result = run_timing_benchmark(
        repeats=1,
        warmups=0,
        cases=("reconnection_scaling",),
    )
    assert result.diagnostics["schema"] == TIMING_BENCHMARK_SCHEMA
    case = result.cases[0]
    summary = case.summary
    assert case.name == "reconnection_scaling"
    assert summary["duration_seconds_median"] > 0.0
    assert summary["duration_seconds_median"] == pytest.approx(
        summary["duration_seconds_min"]
    )
    assert summary["peak_tracemalloc_mib_max"] >= 0.0
    assert result.diagnostics["environment"]["jax_platform"]


def test_timing_benchmark_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="repeats"):
        run_timing_benchmark(repeats=0)
    with pytest.raises(ValueError, match="warmups"):
        run_timing_benchmark(warmups=-1)
    with pytest.raises(ValueError, match="unknown timing"):
        run_timing_benchmark(cases=("missing_case",))


def test_write_timing_benchmark_artifacts(tmp_path) -> None:
    manifest_path, diagnostics = write_timing_benchmark(
        tmp_path,
        repeats=1,
        warmups=0,
        cases=("reconnection_scaling",),
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert diagnostics["schema"] == TIMING_BENCHMARK_SCHEMA
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["cases"][0]["name"] == "reconnection_scaling"
    assert (tmp_path / "timing.md").read_text().startswith("# MHX FAST timing benchmark")
    assert (tmp_path / "figures" / "timing_summary.png").stat().st_size > 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["timing_summary"] == "figures/timing_summary.png"


def test_timing_cli(tmp_path) -> None:
    outdir = tmp_path / "timing"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "timing",
            "--outdir",
            str(outdir),
            "--repeats",
            "1",
            "--warmups",
            "0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    timing = json.loads((outdir / "timing.json").read_text())
    assert {case["name"] for case in timing["cases"]} == {
        "linear_tearing_fast",
        "resistive_decay_fast",
        "reconnection_scaling",
    }
