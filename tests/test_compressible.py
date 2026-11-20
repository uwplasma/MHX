"""Track C gates C1 and C2 for the subsonic compressible spectral model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mhx.equations import mhd3d
from mhx.equations.compressible import (
    CompressibleParams,
    CompressibleState,
    compressible_nonlinear,
    linear_block,
)
from mhx.physics.equilibria3d import OrszagTang3DEquilibrium
from mhx.time_integrators.low_storage import evolve_if_rk3

LENGTHS = (2.0 * jnp.pi,) * 3


def make_tools(shape, params):
    k = mhd3d.wavevectors(shape, LENGTHS)
    mask = mhd3d.two_thirds_mask_rfft(shape)

    def nonlinear(state: CompressibleState) -> CompressibleState:
        return compressible_nonlinear(state, params, shape=shape, k=k, mask=mask)

    zero_decay = CompressibleState(
        lnrho_hat=jnp.zeros(mhd3d.spectral_shape(shape)),
        u_hat=jnp.zeros((3, *mhd3d.spectral_shape(shape))),
        b_hat=jnp.zeros((3, *mhd3d.spectral_shape(shape))),
    )
    return k, mask, nonlinear, zero_decay


def fit_complex_frequency(times, history):
    gamma = -np.polyfit(times, np.log(np.abs(history)), 1)[0]
    omega = np.abs(np.polyfit(times, np.unwrap(np.angle(history)), 1)[0])
    return omega, gamma


def test_c1_sound_wave_matches_stokes_kirchhoff() -> None:
    """Viscously damped sound: gamma = ((4/3) nu + nu_b) k^2 / 2 exactly."""
    shape = (32, 4, 4)
    amplitude = 1.0e-6
    params = CompressibleParams(
        sound_speed=1.0,
        viscosity=2.0e-2,
        bulk_viscosity=1.0e-2,
        resistivity=0.0,
    )
    gamma_exact = 0.5 * (4.0 / 3.0 * 2.0e-2 + 1.0e-2)
    omega_exact = float(np.sqrt(1.0 - gamma_exact**2))

    x = jnp.linspace(0.0, 2.0 * jnp.pi, shape[0], endpoint=False)[:, None, None]
    zero = jnp.zeros(shape)
    # Exact damped-acoustic eigenvector: u_x/lnrho = omega + i gamma.
    lnrho = amplitude * jnp.broadcast_to(jnp.cos(x), shape)
    u_x = amplitude * jnp.broadcast_to(
        omega_exact * jnp.cos(x) - gamma_exact * (-jnp.sin(x)), shape
    )
    state0 = CompressibleState(
        lnrho_hat=jnp.fft.rfftn(lnrho, axes=(-3, -2, -1)),
        u_hat=jnp.stack(
            [jnp.fft.rfftn(u_x, axes=(-3, -2, -1))]
            + [jnp.fft.rfftn(zero, axes=(-3, -2, -1))] * 2
        ),
        b_hat=jnp.stack([jnp.fft.rfftn(zero, axes=(-3, -2, -1))] * 3),
    )
    _, _, nonlinear, zero_decay = make_tools(shape, params)

    trajectory = evolve_if_rk3(
        state0, nonlinear, zero_decay, dt=2.0e-3, steps=4000, save_every=40
    )
    times = np.asarray(trajectory.times)
    history = np.asarray(trajectory.states.lnrho_hat[:, 1, 0, 0])
    omega_fit, gamma_fit = fit_complex_frequency(times, history)

    assert abs(gamma_fit - gamma_exact) / gamma_exact < 1.0e-3
    assert abs(omega_fit - omega_exact) / omega_exact < 1.0e-3


def test_c1_oblique_modes_match_the_exact_block_eigenvalues() -> None:
    """Fitted frequencies against eig of the exact 7x7 linearization."""
    shape = (16, 4, 16)
    amplitude = 1.0e-7
    params = CompressibleParams(
        sound_speed=0.7,
        viscosity=1.5e-2,
        bulk_viscosity=5.0e-3,
        resistivity=1.0e-2,
        guide_field=(0.0, 0.0, 1.0),
    )
    k_vector = (1.0, 0.0, 1.0)
    block = linear_block(k_vector, params)
    eigenvalues, eigenvectors = np.linalg.eig(block)

    # Take the least-damped oscillatory mode (a fast magnetosonic branch).
    oscillatory = [
        (val, eigenvectors[:, i])
        for i, val in enumerate(eigenvalues)
        if abs(val.imag) > 1.0e-6
    ]
    value, vector = max(oscillatory, key=lambda pair: pair[0].real)
    omega_exact, gamma_exact = abs(value.imag), -value.real
    assert gamma_exact > 0.0

    x = jnp.linspace(0.0, 2.0 * jnp.pi, shape[0], endpoint=False)[:, None, None]
    z = jnp.linspace(0.0, 2.0 * jnp.pi, shape[2], endpoint=False)[None, None, :]
    phase = x + z

    def field_from(component: complex) -> jnp.ndarray:
        return amplitude * jnp.broadcast_to(
            jnp.real(component) * jnp.cos(phase) - jnp.imag(component) * jnp.sin(phase),
            shape,
        )

    def to_hat(field: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.rfftn(field, axes=(-3, -2, -1))

    state0 = CompressibleState(
        lnrho_hat=to_hat(field_from(vector[0])),
        u_hat=jnp.stack([to_hat(field_from(vector[1 + i])) for i in range(3)]),
        b_hat=jnp.stack([to_hat(field_from(vector[4 + i])) for i in range(3)]),
    )
    _, _, nonlinear, zero_decay = make_tools(shape, params)

    trajectory = evolve_if_rk3(
        state0, nonlinear, zero_decay, dt=2.0e-3, steps=4000, save_every=40
    )
    times = np.asarray(trajectory.times)
    history = np.asarray(trajectory.states.lnrho_hat[:, 1, 0, 1])
    omega_fit, gamma_fit = fit_complex_frequency(times, history)

    assert abs(omega_fit - omega_exact) / omega_exact < 2.0e-3
    assert abs(gamma_fit - gamma_exact) / gamma_exact < 2.0e-3


def test_solenoidality_of_b_is_preserved() -> None:
    shape = (16, 16, 16)
    params = CompressibleParams(
        sound_speed=1.0,
        viscosity=5.0e-3,
        bulk_viscosity=5.0e-3,
        resistivity=5.0e-3,
    )
    k, _, nonlinear, zero_decay = make_tools(shape, params)
    velocity, magnetic = OrszagTang3DEquilibrium(beta=0.2).initial_fields(shape)
    state = CompressibleState(
        lnrho_hat=jnp.fft.rfftn(jnp.zeros(shape), axes=(-3, -2, -1)),
        u_hat=mhd3d.project(mhd3d.to_spectral(0.2 * velocity), k),
        b_hat=mhd3d.project(mhd3d.to_spectral(magnetic), k),
    )
    trajectory = evolve_if_rk3(
        state, nonlinear, zero_decay, dt=2.0e-3, steps=100, save_every=100
    )
    final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
    scale = float(jnp.max(jnp.abs(final.b_hat)))
    assert float(mhd3d.divergence_linf(final.b_hat, k)) < 1.0e-10 * scale


def test_c2_pseudosound_density_scales_as_mach_squared() -> None:
    """Nearly incompressible theory: density fluctuations grow as M^2."""
    shape = (16, 16, 16)

    def density_rms(mach: float) -> float:
        params = CompressibleParams(
            sound_speed=1.0,
            viscosity=5.0e-3,
            bulk_viscosity=5.0e-3,
            resistivity=5.0e-3,
        )
        k, _, nonlinear, zero_decay = make_tools(shape, params)
        velocity, magnetic = OrszagTang3DEquilibrium(beta=0.8).initial_fields(shape)
        # The OT fields have unit-order amplitudes: rescale so u_rms ~ M c_s.
        scale_factor = mach / 2.0
        state = CompressibleState(
            lnrho_hat=jnp.fft.rfftn(jnp.zeros(shape), axes=(-3, -2, -1)),
            u_hat=mhd3d.project(mhd3d.to_spectral(scale_factor * velocity), k),
            b_hat=mhd3d.project(mhd3d.to_spectral(scale_factor * magnetic), k),
        )
        trajectory = evolve_if_rk3(
            state, nonlinear, zero_decay, dt=5.0e-3, steps=400, save_every=400
        )
        final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
        lnrho = jnp.fft.irfftn(final.lnrho_hat, s=shape, axes=(-3, -2, -1))
        return float(jnp.std(lnrho))

    low, high = density_rms(0.1), density_rms(0.2)
    exponent = np.log2(high / low)
    assert 1.7 < exponent < 2.3
