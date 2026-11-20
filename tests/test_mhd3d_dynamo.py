"""Gate G5: the ABC kinematic dynamo windows, and the ETDRK4 cross-check."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.equations import mhd3d
from mhx.physics.equilibria3d import (
    ABCFlowEquilibrium,
    CircularlyPolarizedAlfvenEquilibrium,
)
from mhx.state.mhd3d import MHD3DParams, MHD3DState
from mhx.time_integrators.exponential import evolve_etdrk4
from mhx.time_integrators.low_storage import evolve_if_rk3

LENGTHS = (2.0 * jnp.pi,) * 3


def abc_growth_rate(magnetic_reynolds: float, *, shape=(32, 32, 32)) -> float:
    """Kinematic growth rate of the 1:1:1 ABC dynamo at ``Rm = 1/eta``."""
    velocity, seed = ABCFlowEquilibrium(seed_amplitude=1.0e-8).initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    mask = mhd3d.two_thirds_mask_rfft(shape)
    v_hat = mhd3d.project(mhd3d.to_spectral(velocity), k)
    u_real = mhd3d.to_physical(v_hat, shape=shape)

    params = MHD3DParams(viscosity=0.0, resistivity=1.0 / magnetic_reynolds)

    def kinematic(state: MHD3DState) -> MHD3DState:
        b = mhd3d.to_physical(state.b_hat, shape=shape)
        electric = jnp.cross(u_real, b, axis=0)
        e_hat = mhd3d.to_spectral(electric) * mask[None]
        return MHD3DState(
            v_hat=jnp.zeros_like(state.v_hat),
            b_hat=mhd3d.curl_hat(e_hat, k),
        )

    state0 = MHD3DState(
        v_hat=jnp.zeros_like(v_hat),
        b_hat=mhd3d.project(mhd3d.to_spectral(seed), k),
    )
    rates = mhd3d.decay_rates(params, k)
    decay = MHD3DState(v_hat=jnp.zeros_like(rates.v_hat), b_hat=rates.b_hat)

    dt, t_end = 2.0e-2, 240.0
    trajectory = evolve_if_rk3(
        state0, kinematic, decay, dt=dt, steps=int(t_end / dt), save_every=500
    )
    times = np.asarray(trajectory.times)
    energy = np.array(
        [
            float(
                mhd3d.energies(
                    jax.tree.map(lambda leaf, i=i: leaf[i], trajectory.states),
                    shape=shape,
                )["magnetic"]
            )
            for i in range(len(times))
        ]
    )
    window = times > 120.0
    return 0.5 * float(np.polyfit(times[window], np.log(energy[window]), 1)[0])


@pytest.mark.slow
def test_g5_abc_dynamo_windows_match_galloway_frisch() -> None:
    """Growth inside both windows, none in the gap (Bouya--Dormy 2013).

    The first window spans about Rm 8.9 to 17.5 and the second opens near
    Rm 27, with ``Rm = 1/eta`` for the unit-amplitude 1:1:1 flow on the
    2 pi box. Near-edge rates are small, so the gap assertion is one-sided.
    """
    sigma_window_one = abc_growth_rate(12.0)
    sigma_gap = abc_growth_rate(20.0)
    sigma_window_two = abc_growth_rate(30.0)

    assert sigma_window_one > 1.0e-3
    assert sigma_window_two > 8.0e-3
    assert sigma_gap < 1.0e-3
    assert sigma_gap < sigma_window_one < sigma_window_two


def test_etdrk4_cross_checks_the_production_stepper() -> None:
    """Two independent steppers must converge to the same exact solution.

    On the CP Alfvén wave the exact answer is a phase rotation with an
    exact decay envelope. ETDRK4 at fourth order must beat IF-RK3 at the
    same step, and both must land on the analytic state.
    """
    shape = (4, 4, 32)
    amplitude = 0.3
    equilibrium = CircularlyPolarizedAlfvenEquilibrium(amplitude=amplitude, mode=1)
    velocity, magnetic = equilibrium.initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    mask = mhd3d.two_thirds_mask_rfft(shape)
    state0 = MHD3DState(
        v_hat=mhd3d.project(mhd3d.to_spectral(velocity), k),
        b_hat=mhd3d.project(mhd3d.to_spectral(magnetic), k),
    )
    eta = 1.0e-3
    params = MHD3DParams(viscosity=eta, resistivity=eta, guide_field=(0.0, 0.0, 1.0))
    decay = mhd3d.decay_rates(params, k)

    def nonlinear(state: MHD3DState) -> MHD3DState:
        return mhd3d.mhd3d_nonlinear(state, params, shape=shape, k=k, mask=mask)

    dt, steps = 2.0e-2, 100
    t_end = dt * steps
    expected_b = state0.b_hat * jnp.exp(-1j * k[2] * t_end) * jnp.exp(-eta * t_end)
    scale = amplitude * shape[0] ** 3

    def final_error(evolve) -> float:
        trajectory = evolve(
            state0, nonlinear, decay, dt=dt, steps=steps, save_every=steps
        )
        final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
        return float(jnp.max(jnp.abs(final.b_hat - expected_b)) / scale)

    error_rk3 = final_error(evolve_if_rk3)
    error_etdrk4 = final_error(evolve_etdrk4)

    assert error_rk3 < 1.0e-4
    assert error_etdrk4 < error_rk3 / 5.0


def test_strong_guide_field_keeps_exact_dispersion() -> None:
    """Accuracy gate at B0/b = 33: the wave CFL costs step size, not physics.

    The guide field enters through the real-space products rather than an
    exact Elsässer rotation, so the stepper must resolve the fast Alfvén
    oscillation. This gate pins that the dispersion stays exact at strong
    B0 once the step resolves the wave, and that halving the step converges
    at third order. The exact phase rotation stays an optimization item.
    """
    shape = (4, 4, 32)
    amplitude = 0.3
    equilibrium = CircularlyPolarizedAlfvenEquilibrium(amplitude=amplitude, mode=1)
    velocity, magnetic = equilibrium.initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    mask = mhd3d.two_thirds_mask_rfft(shape)
    state0 = MHD3DState(
        v_hat=mhd3d.project(mhd3d.to_spectral(velocity), k),
        b_hat=mhd3d.project(mhd3d.to_spectral(magnetic), k),
    )
    eta = 1.0e-3
    guide = 10.0
    params = MHD3DParams(
        viscosity=eta, resistivity=eta, guide_field=(0.0, 0.0, guide)
    )
    decay = mhd3d.decay_rates(params, k)

    def nonlinear(state: MHD3DState) -> MHD3DState:
        return mhd3d.mhd3d_nonlinear(state, params, shape=shape, k=k, mask=mask)

    def final_error(dt: float, steps: int) -> float:
        trajectory = evolve_if_rk3(
            state0, nonlinear, decay, dt=dt, steps=steps, save_every=steps
        )
        final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
        t_end = dt * steps
        phase = jnp.exp(-1j * k[2] * guide * t_end)
        expected_b = state0.b_hat * phase * jnp.exp(-eta * t_end)
        return float(
            jnp.max(jnp.abs(final.b_hat - expected_b)) / (amplitude * shape[0] ** 3)
        )

    coarse = final_error(2.0e-3, 1000)
    fine = final_error(1.0e-3, 2000)
    assert coarse < 1.0e-3
    assert coarse / fine > 5.0
