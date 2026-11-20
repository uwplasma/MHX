"""Tests for the isotropic magnetic spectrum, power-law fit and frame sampling."""

from __future__ import annotations

import numpy as np
import pytest

from mhx.diagnostics.spectra import fit_power_law, magnetic_power_spectrum_2d
from mhx.plotting.frames import sample_indices

N = 64
LENGTH = 3.0


def _x_mesh() -> np.ndarray:
    x = np.arange(N) * LENGTH / N
    return np.meshgrid(x, x, indexing="ij")[0]


def test_single_mode_power_lands_in_its_shell_with_analytic_value() -> None:
    amplitude, mode = 0.7, 5
    psi = amplitude * np.cos(2.0 * np.pi * mode * _x_mesh() / LENGTH)

    k, power = magnetic_power_spectrum_2d(psi, LENGTH)

    assert k[0] == 1 and k[-1] == N // 2 - 1
    assert power.shape == k.shape
    expected = np.pi * mode**2 * amplitude**2 * LENGTH
    assert power[mode - 1] == pytest.approx(expected, rel=1e-10)
    others = np.delete(power, mode - 1)
    assert np.max(np.abs(others)) < 1e-12 * expected


def test_shell_sum_matches_gradient_energy_of_resolved_shells() -> None:
    x = _x_mesh()
    y = x.T
    q = 2.0 * np.pi / LENGTH
    psi = 0.3 * np.cos(2 * q * x) + 0.2 * np.sin(3 * q * y) + 0.1 * np.cos(q * (x + 2 * y))

    k, power = magnetic_power_spectrum_2d(psi, LENGTH)

    dk = 2.0 * np.pi / LENGTH
    # mean |grad psi|^2 evaluated analytically from the three modes.
    mean_grad_sq = 0.5 * (
        (0.3 * 2 * q) ** 2 + (0.2 * 3 * q) ** 2 + (0.1**2) * (q**2 + (2 * q) ** 2)
    )
    # The spectrum integrates to the box-integrated |grad psi|^2 (= L^2 <|grad psi|^2>).
    assert np.sum(power) * dk == pytest.approx(LENGTH**2 * mean_grad_sq, rel=1e-10)
    assert np.all(power >= 0.0)
    # The oblique mode (1, 2) has radius sqrt(5) ~ 2.24 and falls in shell 2.
    assert power[k == 2][0] > power[k == 1][0]


def test_constant_offset_carries_no_power() -> None:
    k, power = magnetic_power_spectrum_2d(np.full((16, 16), 4.2), 1.0)
    assert k.tolist() == list(range(1, 8))
    assert np.all(power == 0.0)


def test_fit_power_law_recovers_slope_from_synthetic_field() -> None:
    slope = -5.0 / 3.0
    x = _x_mesh()
    modes = np.arange(1, N // 2)
    psi = sum(
        m ** ((slope - 2.0) / 2.0) * np.cos(2.0 * np.pi * m * x / LENGTH + 0.3 * m)
        for m in modes
    )
    k, power = magnetic_power_spectrum_2d(psi, LENGTH)

    fitted, intercept = fit_power_law(k, power, kmin=2, kmax=30)

    assert fitted == pytest.approx(slope, abs=1e-9)
    assert intercept == pytest.approx(np.log10(np.pi * LENGTH), abs=1e-9)


def test_fit_power_law_band_selection_and_exact_line() -> None:
    k = np.arange(1, 11, dtype=float)
    power = 3.0 * k**-2.0
    power[0] = 1e6  # outside the band; must not influence the fit
    slope, intercept = fit_power_law(k, power, kmin=2, kmax=10)
    assert slope == pytest.approx(-2.0)
    assert intercept == pytest.approx(np.log10(3.0))


def test_fit_power_law_ignores_non_positive_samples() -> None:
    k = np.array([1.0, 2.0, 3.0, 4.0])
    power = np.array([1.0, 0.0, -5.0, 0.25])
    slope, _ = fit_power_law(k, power, kmin=1, kmax=4)
    assert slope == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("power", "kmin", "kmax"),
    [
        (np.array([1.0, 2.0, 3.0]), 10, 20),  # empty band
        (np.array([1.0, 2.0, 3.0]), 2, 2),  # single sample
        (np.array([1.0, 0.0, -1.0]), 1, 3),  # only one positive sample
    ],
)
def test_fit_power_law_needs_two_positive_samples(power, kmin, kmax) -> None:
    with pytest.raises(ValueError, match="at least two positive samples"):
        fit_power_law(np.array([1.0, 2.0, 3.0]), power, kmin=kmin, kmax=kmax)


def test_sample_indices_returns_everything_when_few_frames() -> None:
    assert sample_indices(5, 5) == [0, 1, 2, 3, 4]
    assert sample_indices(3, 10) == [0, 1, 2]
    assert sample_indices(1, 1) == [0]


@pytest.mark.parametrize(("count", "maximum"), [(100, 7), (101, 10), (10, 3), (1000, 2), (9, 8)])
def test_sample_indices_keeps_endpoints_and_is_increasing(count, maximum) -> None:
    indices = sample_indices(count, maximum)
    assert indices[0] == 0
    assert indices[-1] == count - 1
    assert len(indices) <= maximum
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))
    assert all(isinstance(i, int) for i in indices)


def test_sample_indices_is_uniform_when_evenly_divisible() -> None:
    assert sample_indices(101, 11) == list(range(0, 101, 10))


def test_sample_indices_validates_arguments() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        sample_indices(0, 3)
    with pytest.raises(ValueError, match="maximum"):
        sample_indices(5, 0)
