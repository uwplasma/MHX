"""Edge-case and error-path tests for the benchmark modules.

Covers run-report generation from hand-built run directories, input
validation of the FKR, Orszag-Tang and turbulence benchmarks, and the small
file-reading helpers used by the turbulence readiness checks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mhx.benchmarks import (
    loureiro_plasmoid_estimate,
    run_decaying_mhd_turbulence_validation,
    run_fkr_growth_rate_validation,
    run_fkr_window_validation,
    run_linear_tearing_smoke,
    run_orszag_tang_vortex_validation,
    write_orszag_tang_vortex_validation,
    write_run_report,
)
from mhx.benchmarks import orszag_tang as orszag_module
from mhx.benchmarks import tearing as tearing_module
from mhx.benchmarks import turbulence as turbulence_module
from mhx.config import MeshConfig, NumericsConfig, RunConfig, TimeConfig

# ---------------------------------------------------------------------------
# Run reports
# ---------------------------------------------------------------------------


def _hand_built_run(tmp_path, **extra):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    diagnostics = {
        "final_time": 0.1,
        "initial_total_energy": 2.0,
        "final_total_energy": 1.5,
        "diagnostic_mode": [1, 1],
        "initial_mode_amplitude": 1.0e-3,
        "final_mode_amplitude": 2.0e-3,
        "gamma_fit": 0.5,
    }
    diagnostics.update(extra)
    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest = {"schema": "mhx.manifest.v1", "hashes": {"diagnostics": "abc"}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_report_without_quantities_records_warnings(tmp_path) -> None:
    run_dir = _hand_built_run(tmp_path)
    json_path, markdown_path = write_run_report(run_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["diagnostic_metadata"] == []
    assert report["additional_scalar_diagnostics"] == {}
    assert any("diagnostic_quantities missing" in item for item in report["warnings"])
    text = markdown_path.read_text(encoding="utf-8")
    assert "## Warnings" in text
    assert "- diagnostic_quantities missing; cannot reconstruct registry metadata" in text
    assert "Additional scalar diagnostics" not in text
    assert "| gamma fit | `0.5` |" in text


def test_report_warns_when_effective_config_is_missing(tmp_path) -> None:
    run_dir = _hand_built_run(tmp_path, diagnostic_quantities=["energy"])
    json_path, _ = write_run_report(run_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert (
        "config_effective.json missing; cannot reconstruct registry metadata"
        in (report["warnings"])
    )
    assert report["diagnostic_metadata"] == []


def test_report_warns_for_unknown_diagnostic_name(tmp_path) -> None:
    run_dir = _hand_built_run(
        tmp_path, diagnostic_quantities=["definitely_not_registered"], extra_scalar=3
    )
    (run_dir / "config_effective.json").write_text(json.dumps({}), encoding="utf-8")
    json_path, markdown_path = write_run_report(run_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert any(
        item.startswith("could not reconstruct diagnostic registry metadata")
        for item in report["warnings"]
    )
    assert report["additional_scalar_diagnostics"] == {"extra_scalar": 3}
    assert "| `extra_scalar` | `3` |" in markdown_path.read_text(encoding="utf-8")


def test_report_lists_registry_metadata_for_known_diagnostics(tmp_path) -> None:
    run_dir = _hand_built_run(tmp_path, diagnostic_quantities=["energy", "mode_growth"])
    (run_dir / "config_effective.json").write_text(json.dumps({}), encoding="utf-8")
    json_path, markdown_path = write_run_report(run_dir)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in report["diagnostic_metadata"]] == ["energy", "mode_growth"]
    assert "| `energy` |" in markdown_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tearing smoke benchmark
# ---------------------------------------------------------------------------


def _implicit_config(**numerics) -> RunConfig:
    return RunConfig(
        mesh=MeshConfig(shape=(8, 8)),
        time=TimeConfig(t1=0.2, dt=0.1),
        numerics=NumericsConfig(time_integrator="backward_euler", **numerics),
    )


def test_implicit_smoke_run_reports_solver_diagnostics() -> None:
    _, diagnostics = run_linear_tearing_smoke(_implicit_config())
    assert diagnostics["implicit_converged"] is True
    assert diagnostics["implicit_linear_converged"] is True
    assert diagnostics["implicit_max_residual_norm"] < 1.0e-9
    assert diagnostics["time_integrator"] == "backward_euler"


def test_implicit_smoke_run_raises_when_newton_does_not_converge() -> None:
    config = _implicit_config(nonlinear_max_steps=1, rtol=1.0e-30, atol=1.0e-30)
    with pytest.raises(RuntimeError, match="nonlinear solve did not converge"):
        run_linear_tearing_smoke(config)


def test_implicit_smoke_run_raises_when_linear_solve_does_not_converge(monkeypatch) -> None:
    real = tearing_module.evolve_backward_euler

    def failing_linear(*args, **kwargs):
        result = real(*args, **kwargs)
        return result._replace(linear_converged=np.zeros_like(result.linear_converged))

    monkeypatch.setattr(tearing_module, "evolve_backward_euler", failing_linear)
    with pytest.raises(RuntimeError, match="linear solve did not converge"):
        run_linear_tearing_smoke(_implicit_config())


# ---------------------------------------------------------------------------
# Analytic theory and FKR validation inputs
# ---------------------------------------------------------------------------


def test_plasmoid_estimate_serializes_to_plain_floats() -> None:
    estimate = loureiro_plasmoid_estimate(1.0e6)
    payload = estimate.to_dict()
    assert set(payload) == {"lundquist", "gamma_tau_a", "fastest_mode_k_l"}
    assert payload["gamma_tau_a"] == pytest.approx(1.0e6**0.25)
    assert payload["fastest_mode_k_l"] == pytest.approx(1.0e6**0.375)
    with pytest.raises(ValueError, match="lundquist must be positive"):
        loureiro_plasmoid_estimate(0.0)


def test_fkr_window_requires_strictly_increasing_ka() -> None:
    with pytest.raises(ValueError, match="ka samples must be strictly increasing"):
        run_fkr_window_validation(ka=(0.2, 0.1, 0.3))


def test_fkr_growth_rate_validates_scan_parameters() -> None:
    with pytest.raises(ValueError, match="xmax_over_a must be greater than 1"):
        run_fkr_growth_rate_validation(xmax_over_a=1.0)
    with pytest.raises(ValueError, match="steps must be at least 100"):
        run_fkr_growth_rate_validation(steps=50)
    with pytest.raises(ValueError, match="requires ka < 1"):
        run_fkr_growth_rate_validation(ka=(0.5, 0.9, 1.1))


# ---------------------------------------------------------------------------
# Orszag-Tang
# ---------------------------------------------------------------------------


def test_orszag_tang_requires_four_steps() -> None:
    with pytest.raises(ValueError, match="at least four RK4 steps"):
        run_orszag_tang_vortex_validation(shape=(8, 8), dt=0.01, t_end=0.02)


def test_orszag_tang_artifacts_without_movies(tmp_path) -> None:
    manifest_path, _ = write_orszag_tang_vortex_validation(
        tmp_path / "ot",
        shape=(8, 8),
        dt=0.005,
        t_end=0.02,
        save_every=1,
        movies=False,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {"diagnostics", "validation", "history", "summary"}
    assert (tmp_path / "ot" / manifest["outputs"]["summary"]).stat().st_size > 0


def test_orszag_tang_frame_sampling_and_spectral_fraction() -> None:
    np.testing.assert_array_equal(orszag_module._sample_indices(3, 5), [0, 1, 2])
    sampled = orszag_module._sample_indices(10, 4)
    assert sampled[0] == 0 and sampled[-1] == 9 and len(sampled) == 4
    assert orszag_module._high_k_fraction(np.zeros((8, 8))) == 0.0
    x = np.arange(8) * 2 * np.pi / 8
    low = np.cos(x)[:, None] * np.ones((1, 8))
    high = np.cos(4 * x)[:, None] * np.ones((1, 8))
    assert orszag_module._high_k_fraction(low) < 1.0e-12
    assert orszag_module._high_k_fraction(high) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Turbulence helpers
# ---------------------------------------------------------------------------


def test_relative_energy_growth_flags_empty_and_nonfinite_histories() -> None:
    assert turbulence_module._relative_energy_growth(np.array([])) == float("inf")
    assert turbulence_module._relative_energy_growth(np.array([1.0, np.nan])) == float("inf")
    assert turbulence_module._relative_energy_growth(np.array([2.0, 3.0, 1.0])) == pytest.approx(
        0.5
    )
    assert turbulence_module._relative_energy_growth(np.array([2.0, 1.0])) == 0.0


def test_read_json_if_exists_handles_missing_and_corrupt_files(tmp_path) -> None:
    assert turbulence_module._read_json_if_exists(tmp_path / "absent.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken", encoding="utf-8")
    assert turbulence_module._read_json_if_exists(corrupt) is None
    good = tmp_path / "good.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert turbulence_module._read_json_if_exists(good) == {"a": 1}


def test_load_npz_history_handles_missing_and_corrupt_files(tmp_path) -> None:
    assert turbulence_module._load_npz_history(tmp_path / "absent.npz") == ({}, None)
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"this is not an npz archive")
    assert turbulence_module._load_npz_history(corrupt) == ({}, None)
    valid = tmp_path / "valid.npz"
    np.savez(valid, times=np.arange(3.0))
    history, schema = turbulence_module._load_npz_history(valid)
    np.testing.assert_array_equal(history["times"], np.arange(3.0))
    assert schema is None


@pytest.mark.parametrize("validation", [{"checks": "oops"}, {"checks": {}}, {}])
def test_readiness_matrix_handles_absent_checks(tmp_path, validation) -> None:
    path = turbulence_module._write_readiness_matrix(
        tmp_path / "figs", validation=validation, title="Readiness"
    )
    assert path.name == "promotion_matrix.png"
    assert path.stat().st_size > 0


def test_readiness_matrix_with_checks_writes_figure(tmp_path) -> None:
    path = turbulence_module._write_readiness_matrix(
        tmp_path / "figs", validation={"checks": {"a_check": True, "b_check": False}}, title="T"
    )
    assert path.stat().st_size > 0


def test_turbulence_rejects_nonpositive_end_time() -> None:
    with pytest.raises(ValueError, match="t_end must be positive"):
        run_decaying_mhd_turbulence_validation(shape=(8, 8), t_end=0.0)
