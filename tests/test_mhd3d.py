"""Gates G1 to G4 for the 3D incompressible MHD core, plus gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.equations import mhd3d
from mhx.physics.equilibria3d import (
    CircularlyPolarizedAlfvenEquilibrium,
    OrszagTang3DEquilibrium,
    SingleModeEquilibrium,
)
from mhx.state.mhd3d import MHD3DParams, MHD3DState
from mhx.time_integrators.low_storage import evolve_if_rk3, if_rk3_step

LENGTHS = (2.0 * jnp.pi,) * 3


def make_state(shape, equilibrium):
    velocity, magnetic = equilibrium.initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    state = MHD3DState(
        v_hat=mhd3d.project(mhd3d.to_spectral(velocity), k),
        b_hat=mhd3d.project(mhd3d.to_spectral(magnetic), k),
    )
    return state, k


def make_nonlinear(shape, k, params):
    mask = mhd3d.two_thirds_mask_rfft(shape)

    def nonlinear(state: MHD3DState) -> MHD3DState:
        return mhd3d.mhd3d_nonlinear(state, params, shape=shape, k=k, mask=mask)

    return nonlinear


def test_g1_single_mode_resistive_decay_is_exact() -> None:
    shape = (16, 16, 16)
    equilibrium = SingleModeEquilibrium(amplitude=1.0e-3, mode=(1, 0, 1))
    state0, k = make_state(shape, equilibrium)
    params = MHD3DParams(viscosity=2.0e-2, resistivity=1.0e-2)
    decay = mhd3d.decay_rates(params, k)

    def zero_nonlinear(state: MHD3DState) -> MHD3DState:
        return jax.tree.map(jnp.zeros_like, state)

    dt, steps = 5.0e-2, 40
    trajectory = evolve_if_rk3(
        state0, zero_nonlinear, decay, dt=dt, steps=steps, save_every=steps
    )
    final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)

    k_squared = 2.0  # mode (1, 0, 1)
    expected = state0.b_hat * jnp.exp(-params.resistivity * k_squared * dt * steps)
    error = jnp.max(jnp.abs(final.b_hat - expected)) / jnp.max(jnp.abs(expected))
    assert float(error) < 1.0e-12


def test_projector_is_idempotent_and_kills_gradients() -> None:
    shape = (8, 8, 8)
    k = mhd3d.wavevectors(shape, LENGTHS)
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (3, *shape))
    field_hat = mhd3d.to_spectral(field)

    once = mhd3d.project(field_hat, k)
    twice = mhd3d.project(once, k)
    assert float(jnp.max(jnp.abs(twice - once))) < 1.0e-12
    assert float(mhd3d.divergence_linf(once, k)) < 1.0e-10 * float(
        jnp.max(jnp.abs(once))
    )

    # A pure gradient field projects to zero.
    scalar_hat = mhd3d.to_spectral(jax.random.normal(key, (1, *shape)))[0]
    gradient_hat = 1j * k * scalar_hat[None]
    projected = mhd3d.project(gradient_hat, k)
    assert float(jnp.max(jnp.abs(projected))) < 1.0e-10 * float(
        jnp.max(jnp.abs(gradient_hat))
    )


def test_divergence_stays_at_round_off_through_nonlinear_steps() -> None:
    shape = (16, 16, 16)
    state, k = make_state(shape, OrszagTang3DEquilibrium())
    params = MHD3DParams(viscosity=5.0e-3, resistivity=5.0e-3)
    nonlinear = make_nonlinear(shape, k, params)
    decay = mhd3d.decay_rates(params, k)

    for _ in range(10):
        state = if_rk3_step(state, nonlinear, decay, 1.0e-2)

    scale_v = float(jnp.max(jnp.abs(state.v_hat)))
    scale_b = float(jnp.max(jnp.abs(state.b_hat)))
    assert float(mhd3d.divergence_linf(state.v_hat, k)) < 1.0e-10 * scale_v
    assert float(mhd3d.divergence_linf(state.b_hat, k)) < 1.0e-10 * scale_b


def _mode_history(trajectory, mode):
    return trajectory.states.b_hat[:, 1, mode[0], mode[1], mode[2]]


def test_g2_damped_alfven_wave_matches_exact_dispersion() -> None:
    shape = (8, 8, 16)
    amplitude = 1.0e-6
    nu, eta = 4.0e-2, 1.0e-2

    # Exact dispersion for k_parallel = |k| = 1 and unit guide field.
    k_par, k_sq = 1.0, 1.0
    omega_exact = np.sqrt(k_par**2 - 0.25 * (nu - eta) ** 2 * k_sq**2)
    gamma_exact = 0.5 * (nu + eta) * k_sq

    # Build the exact damped eigenvector so one branch evolves alone:
    # v_hat / b_hat = omega + i (nu - eta) / 2 for this branch.
    x, y, z = np.meshgrid(
        *[np.linspace(0.0, 2.0 * np.pi, points, endpoint=False) for points in shape],
        indexing="ij",
    )
    del x, y
    ratio_real, ratio_imag = omega_exact, 0.5 * (nu - eta)
    zero = np.zeros(shape)
    magnetic = amplitude * np.stack((np.cos(z), zero, zero))
    velocity = amplitude * np.stack(
        (ratio_real * np.cos(z) - ratio_imag * np.sin(z), zero, zero)
    )

    k = mhd3d.wavevectors(shape, LENGTHS)
    state0 = MHD3DState(
        v_hat=mhd3d.to_spectral(jnp.asarray(velocity)),
        b_hat=mhd3d.to_spectral(jnp.asarray(magnetic)),
    )
    params = MHD3DParams(viscosity=nu, resistivity=eta, guide_field=(0.0, 0.0, 1.0))
    nonlinear = make_nonlinear(shape, k, params)
    decay = mhd3d.decay_rates(params, k)

    dt, steps, save_every = 2.0e-3, 4000, 40
    trajectory = evolve_if_rk3(
        state0, nonlinear, decay, dt=dt, steps=steps, save_every=save_every
    )
    times = np.asarray(trajectory.times)
    history = np.asarray(trajectory.states.b_hat[:, 0, 0, 0, 1])

    envelope = np.abs(history)
    gamma_fit = -np.polyfit(times, np.log(envelope), 1)[0]
    phase = np.unwrap(np.angle(history))
    omega_fit = np.abs(np.polyfit(times, phase, 1)[0])

    assert abs(gamma_fit - gamma_exact) / gamma_exact < 1.0e-3
    assert abs(omega_fit - omega_exact) / omega_exact < 1.0e-3


def test_g3_cp_alfven_wave_is_exact_up_to_time_discretization() -> None:
    shape = (4, 4, 32)
    amplitude = 0.3
    equilibrium = CircularlyPolarizedAlfvenEquilibrium(amplitude=amplitude, mode=1)
    state0, k = make_state(shape, equilibrium)
    eta = 1.0e-3
    params = MHD3DParams(viscosity=eta, resistivity=eta, guide_field=(0.0, 0.0, 1.0))
    nonlinear = make_nonlinear(shape, k, params)
    decay = mhd3d.decay_rates(params, k)

    def final_error(dt: float, steps: int) -> float:
        trajectory = evolve_if_rk3(
            state0, nonlinear, decay, dt=dt, steps=steps, save_every=steps
        )
        final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
        t_end = dt * steps
        # Walen state with v = -b propagates in +z: b(z, t) = b0(z - t),
        # damped at exactly eta k^2 with matched diffusivities.
        phase = jnp.exp(-1j * k[2] * t_end)
        expected_b = state0.b_hat * phase * jnp.exp(-eta * t_end)
        return float(
            jnp.max(jnp.abs(final.b_hat - expected_b)) / (amplitude * shape[0] ** 3)
        )

    coarse = final_error(2.0e-2, 100)
    fine = final_error(1.0e-2, 200)
    assert coarse < 1.0e-4
    # Third-order stepper: halving dt cuts the error by about eight.
    assert coarse / fine > 5.0


def test_g4_ideal_invariants_drift_at_time_discretization_only() -> None:
    shape = (16, 16, 16)
    state0, k = make_state(shape, OrszagTang3DEquilibrium())
    params = MHD3DParams(viscosity=0.0, resistivity=0.0)
    nonlinear = make_nonlinear(shape, k, params)
    decay = mhd3d.decay_rates(params, k)

    def drift(dt: float, steps: int) -> tuple[float, float]:
        trajectory = evolve_if_rk3(
            state0, nonlinear, decay, dt=dt, steps=steps, save_every=steps
        )
        final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
        start = mhd3d.energies(state0, shape=shape)
        end = mhd3d.energies(final, shape=shape)
        energy_drift = abs(float(end["total"] - start["total"])) / float(
            start["total"]
        )
        cross_drift = abs(float(end["cross_helicity"] - start["cross_helicity"]))
        return energy_drift, cross_drift

    energy_coarse, cross_coarse = drift(2.0e-3, 50)
    energy_fine, _ = drift(1.0e-3, 100)
    assert energy_coarse < 1.0e-6
    assert cross_coarse < 1.0e-6
    assert energy_coarse / energy_fine > 5.0


def test_orszag_tang_initial_energies_match_the_paper() -> None:
    shape = (32, 32, 32)
    state, k = make_state(shape, OrszagTang3DEquilibrium(beta=0.8))
    del k
    values = mhd3d.energies(state, shape=shape)
    # The exact means of the published beta = 0.8 fields: E_V = 2 and
    # E_M = 0.64 * 6 / 2 = 1.92, near equipartition.
    assert abs(float(values["kinetic"]) - 2.0) < 1.0e-10
    assert abs(float(values["magnetic"]) - 1.92) < 1.0e-10


def test_gradient_of_energy_with_respect_to_viscosity_matches_fd() -> None:
    if not jax.config.jax_enable_x64:
        pytest.skip("gradient gate requires x64")
    shape = (8, 8, 8)
    state0, k = make_state(shape, OrszagTang3DEquilibrium())
    mask = mhd3d.two_thirds_mask_rfft(shape)

    def loss(viscosity):
        params = MHD3DParams(viscosity=viscosity, resistivity=5.0e-3)
        decay = mhd3d.decay_rates(params, k)

        def nonlinear(state: MHD3DState) -> MHD3DState:
            return mhd3d.mhd3d_nonlinear(
                state, params, shape=shape, k=k, mask=mask
            )

        trajectory = evolve_if_rk3(
            state0, nonlinear, decay, dt=5.0e-3, steps=10, save_every=10
        )
        final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
        return mhd3d.energies(final, shape=shape)["total"]

    nu = jnp.asarray(2.0e-2)
    value, gradient = jax.value_and_grad(loss)(nu)
    del value
    eps = 1.0e-6
    finite_difference = (loss(nu + eps) - loss(nu - eps)) / (2.0 * eps)
    relative = abs(float(finite_difference - gradient)) / abs(float(gradient))
    assert relative < 1.0e-6
