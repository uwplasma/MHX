"""Shell spectra, mode coefficients and trajectory diagnostics for periodic 3-D MHD."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.diagnostics import mhd3d as diag
from mhx.equations import mhd3d
from mhx.state.mhd3d import MHD3DState, MHD3DTrajectory

LENGTHS = (2.0 * np.pi,) * 3


def _grid(shape):
    axes = [np.arange(n) * 2.0 * np.pi / n for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def _state(v, b) -> MHD3DState:
    return MHD3DState(
        v_hat=mhd3d.to_spectral(jnp.asarray(v)),
        b_hat=mhd3d.to_spectral(jnp.asarray(b)),
    )


def _trajectory(states, times) -> MHD3DTrajectory:
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
    return MHD3DTrajectory(times=jnp.asarray(times), states=stacked)


def _single_mode_state(shape, v_amp, b_amp) -> MHD3DState:
    """``v = (0, A cos 2x, 0)`` (shell 2) and ``b = (B sin 3z, 0, 0)`` (shell 3)."""
    x, _, z = _grid(shape)
    v = np.stack([np.zeros(shape), v_amp * np.cos(2.0 * x), np.zeros(shape)])
    b = np.stack([b_amp * np.sin(3.0 * z), np.zeros(shape), np.zeros(shape)])
    return _state(v, b)


@pytest.mark.parametrize("shape", [(12, 12, 12), (12, 12, 11)])
def test_shell_spectra_single_modes_land_in_their_shells(shape) -> None:
    state = _single_mode_state(shape, v_amp=0.7, b_amp=0.3)
    spectra = diag.shell_spectra(state, shape=shape)
    kinetic, magnetic = spectra["kinetic"], spectra["magnetic"]
    assert spectra["k"][0] == 0 and spectra["k"].size == kinetic.size
    assert kinetic[2] == pytest.approx(0.7**2 / 4.0, rel=1e-12)
    assert magnetic[3] == pytest.approx(0.3**2 / 4.0, rel=1e-12)
    assert np.sum(np.abs(kinetic)) == pytest.approx(kinetic[2], rel=1e-12)
    assert np.sum(np.abs(magnetic)) == pytest.approx(magnetic[3], rel=1e-12)
    np.testing.assert_allclose(spectra["total"], kinetic + magnetic)


@pytest.mark.parametrize("shape", [(8, 8, 8), (8, 8, 9)])
def test_shell_spectra_sum_reproduces_mean_energy(shape) -> None:
    rng = np.random.default_rng(3)
    state = _state(rng.normal(size=(3, *shape)), rng.normal(size=(3, *shape)))
    spectra = diag.shell_spectra(state, shape=shape)
    energy = mhd3d.energies(state, shape=shape)
    assert np.sum(spectra["kinetic"]) == pytest.approx(float(energy["kinetic"]), rel=1e-12)
    assert np.sum(spectra["magnetic"]) == pytest.approx(float(energy["magnetic"]), rel=1e-12)
    assert np.sum(spectra["total"]) == pytest.approx(float(energy["total"]), rel=1e-12)


def test_signed_mode_coefficient_uses_conjugate_symmetry() -> None:
    shape = (8, 8, 8)
    x, _, z = _grid(shape)
    amp = 0.4
    v = np.stack([np.zeros(shape), amp * np.sin(x + z), np.zeros(shape)])
    state = _state(v, np.zeros_like(v))
    plus = diag.signed_mode_coefficient(state.v_hat, (1, 0, 1), shape=shape)
    minus = diag.signed_mode_coefficient(state.v_hat, (-1, 0, -1), shape=shape)
    # A sin(x+z) = A/(2i) (e^{i(x+z)} - e^{-i(x+z)}).
    np.testing.assert_allclose(plus, [0.0, -0.5j * amp, 0.0], atol=1e-14)
    np.testing.assert_allclose(minus, np.conj(plus), atol=1e-14)
    # A mode absent from the field gives zero.
    absent = diag.signed_mode_coefficient(state.v_hat, (2, 1, -1), shape=shape)
    np.testing.assert_allclose(absent, 0.0, atol=1e-14)


def test_signed_mode_coefficient_rejects_unrepresented_kz() -> None:
    shape = (8, 8, 8)
    state = _single_mode_state(shape, 1.0, 1.0)
    with pytest.raises(ValueError, match="outside the represented z spectrum"):
        diag.signed_mode_coefficient(state.v_hat, (0, 0, 5), shape=shape)


def test_trajectory_bulk_diagnostics_match_analytic_energy_and_dissipation() -> None:
    shape = (12, 12, 12)
    nu, eta = 0.02, 0.05
    amps = (1.0, 2.0)
    traj = _trajectory([_single_mode_state(shape, a * 0.7, a * 0.3) for a in amps], [0.0, 0.5])
    out = diag.trajectory_bulk_diagnostics(traj, shape=shape, viscosity=nu, resistivity=eta)
    np.testing.assert_allclose(out["time"], [0.0, 0.5])
    e_k = np.array([(a * 0.7) ** 2 / 4.0 for a in amps])
    e_m = np.array([(a * 0.3) ** 2 / 4.0 for a in amps])
    np.testing.assert_allclose(out["kinetic"], e_k, rtol=1e-12)
    np.testing.assert_allclose(out["magnetic"], e_m, rtol=1e-12)
    np.testing.assert_allclose(out["total"], e_k + e_m, rtol=1e-12)
    # eps = 2 nu k^2 E_K + 2 eta k^2 E_M with |k| = 2 (velocity) and 3 (field).
    expected = 2.0 * nu * 4.0 * e_k + 2.0 * eta * 9.0 * e_m
    np.testing.assert_allclose(out["dissipation"], expected, rtol=1e-12)


def test_peak_window_spectra_averages_samples_around_peak() -> None:
    shape = (8, 8, 8)
    amps = [1.0, 2.0, 3.0, 4.0]
    times = [0.0, 0.5, 1.0, 1.5]
    traj = _trajectory([_single_mode_state(shape, a, 0.0) for a in amps], times)
    dissipation = [a**2 for a in amps]

    narrow = diag.peak_window_spectra(traj, shape=shape, dissipation=dissipation)
    assert narrow["peak_index"] == 3 and narrow["peak_time"] == pytest.approx(1.5)
    assert narrow["sample_count"] == 1
    assert narrow["kinetic"][2] == pytest.approx(16.0 / 4.0)

    wide = diag.peak_window_spectra(traj, shape=shape, dissipation=dissipation, half_width=0.6)
    assert wide["sample_count"] == 2
    assert wide["kinetic"][2] == pytest.approx(0.5 * (9.0 + 16.0) / 4.0)
    np.testing.assert_allclose(wide["total"], wide["kinetic"] + wide["magnetic"])
    np.testing.assert_array_equal(wide["k"], np.arange(wide["kinetic"].size))

    # A negative half width selects nothing; the peak sample is used as a fallback.
    fallback = diag.peak_window_spectra(
        traj, shape=shape, dissipation=dissipation, half_width=-1.0
    )
    assert fallback["sample_count"] == 1 and fallback["peak_index"] == 3


def test_collision_mode_histories_recover_known_coefficients() -> None:
    shape = (8, 8, 8)
    x, y, z = _grid(shape)
    zero = np.zeros(shape)

    def make(scale):
        v = np.stack([zero, zero, scale * 0.6 * np.cos(x + y)])
        b = np.stack(
            [scale * 0.2 * np.cos(2.0 * x + y - z), zero, scale * 0.2 * np.cos(x + y)]
        )
        return _state(v, b)

    traj = _trajectory([make(1.0), make(2.0)], [0.0, 1.0])
    hist = diag.collision_mode_histories(traj, shape=shape)
    np.testing.assert_allclose(hist["time"], [0.0, 1.0])
    # (1,1,0): cos amplitude a gives coefficient a/2 for v and b.
    np.testing.assert_allclose(hist["secondary_velocity"], [0.3, 0.6], rtol=1e-12)
    np.testing.assert_allclose(hist["secondary_magnetic"], [0.1, 0.2], rtol=1e-12)
    np.testing.assert_allclose(hist["secondary_z_plus"], [0.4, 0.8], rtol=1e-12)
    np.testing.assert_allclose(hist["secondary_z_minus"], [0.2, 0.4], rtol=1e-12)
    # (2,1,-1) uses the conjugate branch for negative kz.
    np.testing.assert_allclose(hist["tertiary_plus_magnetic"], [0.1, 0.2], rtol=1e-12)
    np.testing.assert_allclose(hist["tertiary_plus_velocity"], 0.0, atol=1e-14)
    for name in ("primary_plus", "primary_minus", "tertiary_minus"):
        for kind in ("velocity", "magnetic", "z_plus", "z_minus"):
            np.testing.assert_allclose(hist[f"{name}_{kind}"], 0.0, atol=1e-14)


def test_alfven_collision_reference_closed_forms() -> None:
    ap, am = 0.3, 0.2
    t = np.array([0.0, np.pi / 2.0])
    ref = diag.alfven_collision_reference(t, amplitude_plus=ap, amplitude_minus=am)
    # All amplitudes vanish at t = 0.
    for key in ref:
        if key != "time":
            assert ref[key][0] == pytest.approx(0.0, abs=1e-15)
    # |exp(-2 i omega t) - 1| = 2 at omega t = π/2.
    assert ref["secondary_magnetic"][1] == pytest.approx(np.sqrt(2.0) * ap * am / 16.0 * 2.0)
    plus_pref = ap**2 * am / 640.0
    minus_pref = ap * am**2 / 640.0
    assert ref["tertiary_plus_secular_magnetic"][1] == pytest.approx(
        np.sqrt(5.0) * 4.0 * (np.pi / 2.0) * plus_pref
    )
    assert ref["tertiary_minus_secular_magnetic"][1] == pytest.approx(
        np.sqrt(5.0) * 4.0 * (np.pi / 2.0) * minus_pref
    )


def test_alfven_collision_reference_secular_term_dominates_late_times() -> None:
    t = np.array([2000.0])
    ref = diag.alfven_collision_reference(
        t, amplitude_plus=0.1, amplitude_minus=0.1, alfven_speed=1.5, k_parallel=2.0
    )
    for side in ("plus", "minus"):
        full = ref[f"tertiary_{side}_magnetic"][0]
        secular = ref[f"tertiary_{side}_secular_magnetic"][0]
        assert full == pytest.approx(secular, rel=5e-3)
    # The bounded secondary mode stays O(epsilon^2) at late times.
    assert ref["secondary_magnetic"][0] <= 2.0 * np.sqrt(2.0) * 0.01 / (16.0 * 1.5**2 * 2.0)
