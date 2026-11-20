"""Tests for the forced 2-D reduced-MHD turbulence spectrum benchmark."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from mhx.benchmarks import turbulent_spectrum as ts
from mhx.state import ReducedMHDState

pytestmark = pytest.mark.slow


TINY = {"shape": (32, 32), "dt": 5.0e-3, "t_end": 0.1, "save_every": 10}


@pytest.fixture(scope="module")
def tiny_run() -> ts.TurbulentSpectrumResult:
    """One shared tiny run, reused across tests to avoid recompiling."""
    return ts.run_turbulent_spectrum_validation(**TINY)


def test_dealiasing_removes_only_high_modes() -> None:
    n = 32
    x = jnp.arange(n) / n * 2.0 * jnp.pi
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    low = jnp.sin(2.0 * xx) + jnp.cos(3.0 * yy)  # |k| <= n/3 along each axis
    high = jnp.sin(14.0 * xx)  # 14 > 32/3
    state = ReducedMHDState(psi=low + high, omega=low - high)

    out = ts._apply_2_3_dealiasing(state)

    np.testing.assert_allclose(np.asarray(out.psi), np.asarray(low), atol=1e-10)
    np.testing.assert_allclose(np.asarray(out.omega), np.asarray(low), atol=1e-10)
    assert out.psi.shape == (n, n)
    assert not jnp.iscomplexobj(out.psi)
    assert not jnp.iscomplexobj(out.omega)

    again = ts._apply_2_3_dealiasing(out)
    np.testing.assert_allclose(np.asarray(again.psi), np.asarray(out.psi), atol=1e-10)


def test_result_fields_and_shapes(tiny_run: ts.TurbulentSpectrumResult) -> None:
    r = tiny_run
    t = r.turbulence
    n_frames = len(t.time)
    assert n_frames >= 3  # initial frame plus at least two saved samples
    assert t.time[0] == 0.0
    assert np.all(np.diff(t.time) > 0)
    assert t.time[-1] == pytest.approx(TINY["t_end"])
    assert t.psi.shape == (n_frames, 32, 32)
    assert t.omega.shape == t.psi.shape
    assert np.all(np.isfinite(t.total_energy))
    assert r.k_1d[0] == 1
    assert r.k_1d.shape == r.power_1d.shape
    assert np.all(np.diff(r.k_1d) > 0)
    assert np.all(r.power_1d >= 0.0)
    assert np.any(r.power_1d > 0.0)


def test_diagnostics_and_validation_schema(tiny_run: ts.TurbulentSpectrumResult) -> None:
    r = tiny_run
    d = r.diagnostics
    assert d["schema"] == ts.TURBULENT_SPECTRUM_SCHEMA
    assert d["shape"] == [32, 32]
    assert d["dt"] == TINY["dt"]
    assert d["t_end"] == TINY["t_end"]
    assert d["fit_k_range"] == list(ts.FIT_K_RANGE)
    assert d["spectral_slope"] == r.slope
    assert np.isfinite(d["initial_total_energy"]) and np.isfinite(d["final_total_energy"])
    assert d["max_vorticity"] > 0.0 and d["max_current"] > 0.0
    json.dumps(d)

    v = r.validation
    assert set(v["checks"]) == {"finite_histories", "spectral_slope_in_range"}
    assert v["passed"] == all(v["checks"].values())
    assert v["checks"]["finite_histories"] is True
    assert v["thresholds"] == {"min_slope": ts.SLOPE_RANGE[0], "max_slope": ts.SLOPE_RANGE[1]}
    assert v["diagnostics"] is d


def test_seed_determinism(tiny_run: ts.TurbulentSpectrumResult) -> None:
    same = ts.run_turbulent_spectrum_validation(**TINY)
    np.testing.assert_array_equal(same.power_1d, tiny_run.power_1d)
    other = ts.run_turbulent_spectrum_validation(**TINY, seed=12)
    assert not np.allclose(other.power_1d, tiny_run.power_1d)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"resistivity": 0.0}, "resistivity must be positive"),
        ({"shape": (4, 4)}, "at least 8 points"),
        ({"save_every": 50}, "at least two"),
    ],
)
def test_input_validation_propagates(override: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ts.run_turbulent_spectrum_validation(**{**TINY, **override})


@pytest.mark.parametrize(
    ("slope", "in_range"),
    [(-1.5, True), (ts.SLOPE_RANGE[0], True), (-3.0, False), (-0.5, False)],
)
def test_slope_gate(monkeypatch: pytest.MonkeyPatch, slope: float, in_range: bool) -> None:
    seen = {}

    def fake_fit(k, p, *, kmin, kmax):
        seen["range"] = (kmin, kmax)
        return slope, 1.0

    monkeypatch.setattr(ts, "fit_power_law", fake_fit)
    r = ts.run_turbulent_spectrum_validation(**TINY)

    assert seen["range"] == ts.FIT_K_RANGE
    assert r.slope == slope
    assert r.validation["checks"]["spectral_slope_in_range"] is in_range
    assert r.validation["checks"]["finite_histories"] is True
    assert r.validation["passed"] is in_range


def test_write_outputs(tmp_path) -> None:
    manifest_path, validation = ts.write_turbulent_spectrum_validation(tmp_path / "out", **TINY)
    out = tmp_path / "out"
    assert manifest_path == out / "manifest.json"
    for name in (
        "manifest.json",
        "diagnostics.json",
        "validation.json",
        "turbulent_spectrum.npz",
        "figures/turbulent_spectrum_summary.png",
    ):
        assert (out / name).stat().st_size > 0, name

    with np.load(out / "turbulent_spectrum.npz") as data:
        for key in (
            "time",
            "psi",
            "omega",
            "current_density",
            "magnetic_energy",
            "kinetic_energy",
            "total_energy",
            "k_1d",
            "power_1d",
        ):
            assert key in data.files
        assert data["psi"].shape[1:] == (32, 32)
        assert data["psi"].shape[0] == data["time"].shape[0]
        assert data["k_1d"].shape == data["power_1d"].shape

    on_disk = json.loads((out / "validation.json").read_text())
    assert on_disk == validation
    diagnostics = json.loads((out / "diagnostics.json").read_text())
    assert diagnostics == validation["diagnostics"]

    manifest = json.loads(manifest_path.read_text())
    assert manifest["claim_level"] == "validation"
    assert "summary" in json.dumps(manifest)
