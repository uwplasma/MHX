"""Paper-comparison figures for 3-D MHD: files are written and metrics are sensible."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mhx.equations import mhd3d  # noqa: E402
from mhx.plotting import mhd3d as plotting  # noqa: E402
from mhx.state.mhd3d import MHD3DParams, MHD3DState, MHD3DTrajectory  # noqa: E402

LENGTHS = (2.0 * np.pi,) * 3


def _power_law_field(rng, shape, exponent):
    """Return a solenoidal real field whose Fourier amplitudes scale as ``|k|^-exponent``."""
    white = rng.normal(size=(3, *shape))
    k = np.asarray(mhd3d.wavevectors(shape, LENGTHS))
    k_mag = np.sqrt(np.sum(k**2, axis=0))
    scale = np.where(k_mag > 0, np.maximum(k_mag, 1.0) ** -exponent, 0.0)
    field_hat = mhd3d.to_spectral(jnp.asarray(white)) * scale
    return mhd3d.project(field_hat, jnp.asarray(k))


def _trajectory(states, times):
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
    return MHD3DTrajectory(times=jnp.asarray(times), states=stacked)


def _result(shape, *, seed, exponent=2.0, amplitudes=(1.0, 2.0, 1.5, 0.5)):
    rng = np.random.default_rng(seed)
    v0 = _power_law_field(rng, shape, exponent)
    b0 = _power_law_field(rng, shape, exponent)
    states = [MHD3DState(v_hat=a * v0, b_hat=0.5 * a * b0) for a in amplitudes]
    times = np.linspace(0.0, 0.75, len(amplitudes))
    return SimpleNamespace(
        trajectory=_trajectory(states, times),
        shape=shape,
        parameters=MHD3DParams(viscosity=1.0e-2, resistivity=2.0e-2),
    )


def test_power_law_slope_recovers_exponent_and_handles_sparse_data() -> None:
    k = np.arange(1.0, 11.0)
    assert plotting._power_law_slope(k, 3.0 * k**-1.5, 2.0, 8.0) == pytest.approx(-1.5)
    # Non-positive and non-finite samples are excluded from the fit.
    spectrum = 3.0 * k**-1.5
    spectrum[3] = np.nan
    spectrum[4] = -1.0
    assert plotting._power_law_slope(k, spectrum, 2.0, 8.0) == pytest.approx(-1.5)
    assert np.isnan(plotting._power_law_slope(k, spectrum, 20.0, 30.0))
    assert np.isnan(plotting._power_law_slope(k, k**-1.0, 3.0, 3.0))


def test_taylor_green_comparison_writes_figure_and_metrics(tmp_path) -> None:
    shape = (16, 16, 16)
    results = {
        "insulating": _result(shape, seed=0, exponent=2.0),
        "alternative": _result(shape, seed=1, exponent=1.7),
        "conducting": _result(shape, seed=2, exponent=1.5),
    }
    path = tmp_path / "nested" / "tg.png"
    out, metrics = plotting.plot_taylor_green_paper_comparison(
        results, path, peak_half_width=0.3, note="Runs I, A, C"
    )
    assert out == path and path.is_file() and path.stat().st_size > 1000
    assert set(metrics) == set(results)
    targets = {"insulating": -2.0, "alternative": -5.0 / 3.0, "conducting": -1.5}
    for name, item in metrics.items():
        assert item["target_slope"] == pytest.approx(targets[name])
        assert item["fit_k_min"] == 3.0 and item["fit_k_max"] == 4.0
        assert item["display_k_max"] == 5.0  # N/3
        assert item["peak_samples"] >= 1.0
        # Amplitudes peak (and so does dissipation) at the second sample, t = 0.25.
        assert item["peak_time"] == pytest.approx(0.25)
        assert item["peak_dissipation"] > 0.0
        assert np.isfinite(item["spectral_slope"])
    # Fields were built with energy per shell ~ k^(2 - 2 exponent): steeper input, steeper fit.
    assert metrics["insulating"]["spectral_slope"] < metrics["conducting"]["spectral_slope"]
    assert metrics["insulating"]["spectral_slope"] < -0.3  # a two-point fit is noisy


def _collision_result(shape, times, *, magnitude):
    rng = np.random.default_rng(5)
    base_v = _power_law_field(rng, shape, 0.0)
    base_b = _power_law_field(rng, shape, 0.0)
    states = [
        MHD3DState(v_hat=magnitude * (1.0 + t) * base_v, b_hat=magnitude * (1.0 + t**2) * base_b)
        for t in times
    ]
    return SimpleNamespace(trajectory=_trajectory(states, times), shape=shape)


def test_alfven_collision_comparison_writes_figure_and_metrics(tmp_path) -> None:
    shape = (8, 8, 8)
    times = np.linspace(0.0, 10.0, 11)
    result = _collision_result(shape, times, magnitude=1.0e-2)
    path = tmp_path / "collision.png"
    out, metrics = plotting.plot_alfven_collision_paper_comparison(
        result, path, amplitude_plus=0.1, amplitude_minus=0.1
    )
    assert out == path and path.is_file() and path.stat().st_size > 1000
    assert metrics["target_tertiary_energy_slope"] == 2.0
    assert set(metrics) == {
        "tertiary_energy_slope_after_2pi",
        "target_tertiary_energy_slope",
        "tertiary_energy_relative_l2_vs_equation_40",
        "secondary_relative_l2_vs_equation_36",
        "max_secondary_velocity_to_magnetic",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["max_secondary_velocity_to_magnetic"] > 0.0
    assert metrics["tertiary_energy_relative_l2_vs_equation_40"] > 0.0
    # Both mode histories grow like (1 + t^2)^2 once the field is stationary in shape, so
    # the late-time tertiary power exponent is a positive finite number.
    assert metrics["tertiary_energy_slope_after_2pi"] > 0.0


def test_alfven_collision_comparison_with_empty_modes(tmp_path) -> None:
    shape = (8, 8, 8)
    times = np.linspace(0.0, 4.0, 5)
    zero = jnp.zeros((3, 8, 8, 5), dtype=complex)
    states = [MHD3DState(v_hat=zero, b_hat=zero) for _ in times]
    result = SimpleNamespace(trajectory=_trajectory(states, times), shape=shape)
    path = tmp_path / "empty.png"
    _, metrics = plotting.plot_alfven_collision_paper_comparison(
        result, path, amplitude_plus=0.1, amplitude_minus=0.1, alfven_speed=1.0
    )
    assert path.is_file() and path.stat().st_size > 0
    assert metrics["max_secondary_velocity_to_magnetic"] == 0.0
    assert np.isnan(metrics["tertiary_energy_slope_after_2pi"])
    assert metrics["secondary_relative_l2_vs_equation_36"] == pytest.approx(1.0)
