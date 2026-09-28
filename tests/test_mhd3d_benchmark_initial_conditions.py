"""Published checks for the Taylor--Green and Alfvén-collision benchmarks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.diagnostics.mhd3d import (
    alfven_collision_reference,
    shell_spectra,
    signed_mode_coefficient,
)
from mhx.equations import mhd3d
from mhx.physics.equilibria3d import (
    AlfvenWaveCollisionEquilibrium,
    TaylorGreenAlternativeEquilibrium,
    TaylorGreenConductingEquilibrium,
)
from mhx.state.mhd3d import MHD3DParams, MHD3DState
from mhx.time_integrators.low_storage import evolve_if_rk3

LENGTHS = (2.0 * jnp.pi,) * 3


def _collision_state(shape, amplitude_plus=0.1, amplitude_minus=0.1):
    velocity, magnetic = AlfvenWaveCollisionEquilibrium(
        amplitude_plus=amplitude_plus,
        amplitude_minus=amplitude_minus,
    ).initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    return (
        MHD3DState(
            v_hat=mhd3d.project(mhd3d.to_spectral(velocity), k),
            b_hat=mhd3d.project(mhd3d.to_spectral(magnetic), k),
        ),
        k,
    )


@pytest.mark.parametrize(
    "equilibrium",
    [
        TaylorGreenAlternativeEquilibrium(),
        TaylorGreenConductingEquilibrium(b0=1.0 / jnp.sqrt(3.0)),
    ],
)
def test_taylor_green_a_and_c_start_in_equipartition(equilibrium) -> None:
    """Lee et al. (2010) A/C normalizations give E_V = E_M = 1/8."""
    shape = (16, 16, 16)
    velocity, magnetic = equilibrium.initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    state = MHD3DState(
        v_hat=mhd3d.to_spectral(velocity),
        b_hat=mhd3d.to_spectral(magnetic),
    )
    values = mhd3d.energies(state, shape=shape)
    assert abs(float(values["kinetic"]) - 0.125) < 1.0e-12
    assert abs(float(values["magnetic"]) - 0.125) < 1.0e-12
    assert float(mhd3d.divergence_linf(state.v_hat, k)) < 1.0e-10
    assert float(mhd3d.divergence_linf(state.b_hat, k)) < 1.0e-10
    assert abs(float(values["cross_helicity"])) < 1.0e-12
    assert abs(float(mhd3d.magnetic_helicity(state, shape=shape, k=k))) < 1.0e-12


def test_shell_spectra_close_parseval_energy() -> None:
    shape = (8, 10, 12)
    key_v, key_b = jax.random.split(jax.random.PRNGKey(7))
    k = mhd3d.wavevectors(shape, LENGTHS)
    state = MHD3DState(
        v_hat=mhd3d.project(
            mhd3d.to_spectral(jax.random.normal(key_v, (3, *shape))), k
        ),
        b_hat=mhd3d.project(
            mhd3d.to_spectral(jax.random.normal(key_b, (3, *shape))), k
        ),
    )
    spectra = shell_spectra(state, shape=shape)
    energy = mhd3d.energies(state, shape=shape)
    assert abs(float(np.sum(spectra["kinetic"])) - float(energy["kinetic"])) < 1.0e-12
    assert abs(float(np.sum(spectra["magnetic"])) - float(energy["magnetic"])) < 1.0e-12


def test_signed_mode_coefficient_recovers_negative_kz() -> None:
    shape = (8, 8, 8)
    state, _ = _collision_state(shape)
    plus = signed_mode_coefficient(state.v_hat, (1, 0, -1), shape=shape)
    conjugate_partner = signed_mode_coefficient(state.v_hat, (-1, 0, 1), shape=shape)
    assert np.max(np.abs(plus - np.conj(conjugate_partner))) < 1.0e-12
    assert np.linalg.norm(plus) > 0.0


def test_alfven_collision_generates_published_magnetic_secondary() -> None:
    """Howes--Nielson primaries generate the purely magnetic (1, 1, 0) mode."""
    shape = (8, 8, 8)
    state, k = _collision_state(shape)
    params = MHD3DParams(
        viscosity=0.0,
        resistivity=0.0,
        guide_field=(0.0, 0.0, 1.0),
    )
    rhs = mhd3d.mhd3d_nonlinear(
        state,
        params,
        shape=shape,
        k=k,
        mask=mhd3d.two_thirds_mask_rfft(shape),
    )
    assert float(jnp.linalg.norm(rhs.b_hat[:, 1, 1, 0])) > 1.0e-3
    assert float(jnp.linalg.norm(rhs.v_hat[:, 1, 1, 0])) < 1.0e-10


def test_single_wave_and_copropagating_waves_do_not_interact() -> None:
    shape = (8, 8, 8)
    mask = mhd3d.two_thirds_mask_rfft(shape)
    zero_guide = MHD3DParams(viscosity=0.0, resistivity=0.0)

    single, k = _collision_state(shape, amplitude_minus=0.0)
    single_rhs = mhd3d.mhd3d_nonlinear(
        single, zero_guide, shape=shape, k=k, mask=mask
    )
    assert float(jnp.max(jnp.abs(single_rhs.b_hat))) < 1.0e-10
    assert float(jnp.max(jnp.abs(single_rhs.v_hat))) < 1.0e-10

    x = jnp.linspace(0.0, 2.0 * jnp.pi, shape[0], endpoint=False)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, shape[1], endpoint=False)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, shape[2], endpoint=False)
    x, y, z = jnp.meshgrid(x, y, z, indexing="ij")
    zero = jnp.zeros(shape)
    z_plus = jnp.stack((jnp.cos(y - z), jnp.cos(x - z), zero))
    copropagating = MHD3DState(
        v_hat=mhd3d.to_spectral(0.5 * z_plus),
        b_hat=mhd3d.to_spectral(0.5 * z_plus),
    )
    copropagating_rhs = mhd3d.mhd3d_nonlinear(
        copropagating, zero_guide, shape=shape, k=k, mask=mask
    )
    assert float(jnp.max(jnp.abs(copropagating_rhs.b_hat))) < 1.0e-10
    assert float(jnp.max(jnp.abs(copropagating_rhs.v_hat))) < 1.0e-10


def test_collision_amplitude_scaling_matches_three_and_four_wave_orders() -> None:
    shape = (8, 8, 8)
    params = MHD3DParams(viscosity=0.0, resistivity=0.0)
    secondary = []
    for amplitude in (0.05, 0.1):
        state, k = _collision_state(
            shape, amplitude_plus=amplitude, amplitude_minus=amplitude
        )
        rhs = mhd3d.mhd3d_nonlinear(
            state,
            params,
            shape=shape,
            k=k,
            mask=mhd3d.two_thirds_mask_rfft(shape),
        )
        secondary.append(float(jnp.linalg.norm(rhs.b_hat[:, 1, 1, 0])))
    assert abs(secondary[1] / secondary[0] - 4.0) < 1.0e-10

    times = np.asarray([0.7])
    weak = alfven_collision_reference(
        times, amplitude_plus=0.05, amplitude_minus=0.05
    )
    strong = alfven_collision_reference(
        times, amplitude_plus=0.1, amplitude_minus=0.1
    )
    ratio = strong["tertiary_plus_magnetic"][0] / weak["tertiary_plus_magnetic"][0]
    assert abs(ratio - 8.0) < 1.0e-12


def test_collision_reference_has_paper_periodicity_and_zero_initial_data() -> None:
    times = np.asarray([0.0, 0.5 * np.pi, np.pi])
    reference = alfven_collision_reference(
        times, amplitude_plus=0.1, amplitude_minus=0.1
    )
    assert reference["secondary_magnetic"][0] < 1.0e-15
    assert reference["secondary_magnetic"][1] > 1.0e-3
    assert reference["secondary_magnetic"][2] < 1.0e-15
    assert reference["tertiary_plus_magnetic"][0] < 1.0e-15
    assert reference["tertiary_minus_magnetic"][0] < 1.0e-15
    secular_power = (
        reference["tertiary_plus_secular_magnetic"] ** 2
        + reference["tertiary_minus_secular_magnetic"] ** 2
    )
    assert secular_power[0] == 0.0
    assert abs(secular_power[2] / secular_power[1] - 4.0) < 1.0e-12


def test_weak_collision_matches_equation_36_at_short_time() -> None:
    """A weak full-MHD run follows the O(epsilon^2) magnetic solution."""
    shape = (8, 8, 8)
    amplitude = 1.0e-2
    state0, k = _collision_state(
        shape, amplitude_plus=amplitude, amplitude_minus=amplitude
    )
    params = MHD3DParams(
        viscosity=0.0,
        resistivity=0.0,
        guide_field=(0.0, 0.0, 1.0),
    )
    mask = mhd3d.two_thirds_mask_rfft(shape)

    def nonlinear(state):
        return mhd3d.mhd3d_nonlinear(
            state, params, shape=shape, k=k, mask=mask
        )

    decay = mhd3d.decay_rates(params, k)
    trajectory = evolve_if_rk3(
        state0,
        nonlinear,
        decay,
        dt=1.0e-3,
        steps=500,
        save_every=500,
    )
    final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
    measured = np.linalg.norm(
        signed_mode_coefficient(final.b_hat, (1, 1, 0), shape=shape)
    )
    expected = alfven_collision_reference(
        [0.5], amplitude_plus=amplitude, amplitude_minus=amplitude
    )["secondary_magnetic"][0]
    assert abs(measured - expected) / expected < 2.0e-2
