"""Gates C3 and C4: the incompressible limit and parametric decay."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.equations import mhd3d
from mhx.equations.compressible import (
    CompressibleParams,
    CompressibleState,
    compressible_nonlinear,
)
from mhx.physics.equilibria3d import OrszagTang3DEquilibrium
from mhx.state.mhd3d import MHD3DParams, MHD3DState
from mhx.time_integrators.low_storage import evolve_if_rk3

LENGTHS = (2.0 * jnp.pi,) * 3


def goldstein_derby_growth(beta: float, eta: float, daughters: np.ndarray) -> float:
    """Maximum growth rate of the parametric decay instability.

    Solves the Goldstein (1978) and Derby (1978) dispersion relation in
    pump units (``v_A = k_0 = omega_0 = 1``):

    ``((w + k)^2 - 4)(w - k)(w^2 - beta k^2)
      = eta^2 k^2 (w^3 + k w^2 - 3 w + k)``

    over the given daughter wavenumbers, returning the largest positive
    imaginary part.
    """
    growth = 0.0
    for k in daughters:
        # Left side expanded minus right side, degree five in w.
        left = np.polymul(
            np.polymul([1.0, 2.0 * k, k * k - 4.0], [1.0, -k]),
            [1.0, 0.0, -beta * k * k],
        )
        right = eta * eta * k * k * np.asarray([1.0, k, -3.0, k])
        polynomial = np.polysub(left, np.concatenate(([0.0], right)))
        roots = np.roots(polynomial)
        growth = max(growth, float(np.max(roots.imag)))
    return growth


@pytest.mark.slow
def test_c4_parametric_decay_matches_goldstein_derby() -> None:
    """Compressible-only physics: the CP Alfvén pump decays at the
    predicted rate, and the same pump is stable in incompressible MHD."""
    shape = (256, 4, 4)
    beta, pump_k = 0.1, 4
    sound_speed = float(np.sqrt(beta))
    params = CompressibleParams(
        sound_speed=sound_speed,
        viscosity=2.0e-4,
        bulk_viscosity=0.0,
        resistivity=2.0e-4,
        guide_field=(1.0, 0.0, 0.0),
    )
    k = mhd3d.wavevectors(shape, LENGTHS)
    mask = mhd3d.two_thirds_mask_rfft(shape)

    x = jnp.linspace(0.0, 2.0 * jnp.pi, shape[0], endpoint=False)[:, None, None]
    zero = jnp.zeros(shape)
    b_y = jnp.broadcast_to(jnp.cos(pump_k * x), shape)
    b_z = jnp.broadcast_to(jnp.sin(pump_k * x), shape)
    noise = 1.0e-5 * jax.random.normal(jax.random.PRNGKey(0), shape)

    state0 = CompressibleState(
        lnrho_hat=jnp.fft.rfftn(noise, axes=(-3, -2, -1)) * mask,
        u_hat=jnp.stack(
            [
                jnp.fft.rfftn(zero, axes=(-3, -2, -1)),
                jnp.fft.rfftn(-b_y, axes=(-3, -2, -1)),
                jnp.fft.rfftn(-b_z, axes=(-3, -2, -1)),
            ]
        ),
        b_hat=jnp.stack(
            [
                jnp.fft.rfftn(zero, axes=(-3, -2, -1)),
                jnp.fft.rfftn(b_y, axes=(-3, -2, -1)),
                jnp.fft.rfftn(b_z, axes=(-3, -2, -1)),
            ]
        ),
    )
    spectral = mhd3d.spectral_shape(shape)
    zero_decay = CompressibleState(
        lnrho_hat=jnp.zeros(spectral),
        u_hat=jnp.zeros((3, *spectral)),
        b_hat=jnp.zeros((3, *spectral)),
    )

    def nonlinear(state: CompressibleState) -> CompressibleState:
        return compressible_nonlinear(state, params, shape=shape, k=k, mask=mask)

    dt, steps, save_every = 1.0e-3, 7000, 100
    trajectory = evolve_if_rk3(
        state0, nonlinear, zero_decay, dt=dt, steps=steps, save_every=save_every
    )
    times = np.asarray(trajectory.times)
    spreads = np.array(
        [
            float(
                jnp.std(
                    jnp.fft.irfftn(
                        trajectory.states.lnrho_hat[i], s=shape, axes=(-3, -2, -1)
                    )
                )
            )
            for i in range(len(times))
        ]
    )

    # Fit after the seed mixture has converged onto the fastest daughter
    # and before saturation.
    window = (times > 3.0) & (spreads < 2.0e-2)
    measured = float(np.polyfit(times[window], np.log(spreads[window]), 1)[0])

    daughters = np.arange(1, 3 * pump_k) / pump_k
    predicted = goldstein_derby_growth(beta, 1.0, daughters) * pump_k
    assert predicted > 0.0
    assert abs(measured - predicted) / predicted < 0.1

    # The same pump in incompressible MHD is an exact stable solution.
    inc_params = MHD3DParams(
        viscosity=2.0e-4, resistivity=2.0e-4, guide_field=(1.0, 0.0, 0.0)
    )
    inc_state = MHD3DState(v_hat=state0.u_hat, b_hat=state0.b_hat)
    inc_decay = mhd3d.decay_rates(inc_params, k)

    def inc_nonlinear(state: MHD3DState) -> MHD3DState:
        return mhd3d.mhd3d_nonlinear(
            state, inc_params, shape=shape, k=k, mask=mask
        )

    inc_trajectory = evolve_if_rk3(
        inc_state, inc_nonlinear, inc_decay, dt=dt, steps=steps, save_every=steps
    )
    final = jax.tree.map(lambda leaf: leaf[-1], inc_trajectory.states)
    pump_energy0 = float(mhd3d.energies(inc_state, shape=shape)["magnetic"])
    pump_energy1 = float(mhd3d.energies(final, shape=shape)["magnetic"])
    resistive = float(np.exp(-2.0 * 2.0e-4 * pump_k**2 * dt * steps))

    assert abs(pump_energy1 / pump_energy0 - resistive) < 5.0e-3


def test_c3_compressible_converges_to_incompressible_as_mach_falls() -> None:
    shape = (16, 16, 16)
    k = mhd3d.wavevectors(shape, LENGTHS)
    mask = mhd3d.two_thirds_mask_rfft(shape)
    velocity, magnetic = OrszagTang3DEquilibrium(beta=0.8).initial_fields(shape)
    t_end, dt = 0.5, 2.0e-3

    def incompressible_final():
        params = MHD3DParams(viscosity=5.0e-3, resistivity=5.0e-3)
        state = MHD3DState(
            v_hat=mhd3d.project(mhd3d.to_spectral(0.1 * velocity), k),
            b_hat=mhd3d.project(mhd3d.to_spectral(0.1 * magnetic), k),
        )

        def nonlinear(s):
            return mhd3d.mhd3d_nonlinear(s, params, shape=shape, k=k, mask=mask)

        trajectory = evolve_if_rk3(
            state,
            nonlinear,
            mhd3d.decay_rates(params, k),
            dt=dt,
            steps=int(t_end / dt),
            save_every=int(t_end / dt),
        )
        return jax.tree.map(lambda leaf: leaf[-1], trajectory.states)

    def compressible_final(sound_speed):
        params = CompressibleParams(
            sound_speed=sound_speed,
            viscosity=5.0e-3,
            bulk_viscosity=0.0,
            resistivity=5.0e-3,
        )
        spectral = mhd3d.spectral_shape(shape)
        state = CompressibleState(
            lnrho_hat=jnp.zeros(spectral, dtype=complex),
            u_hat=mhd3d.project(mhd3d.to_spectral(0.1 * velocity), k),
            b_hat=mhd3d.project(mhd3d.to_spectral(0.1 * magnetic), k),
        )

        def nonlinear(s):
            return compressible_nonlinear(s, params, shape=shape, k=k, mask=mask)

        zero_decay = CompressibleState(
            lnrho_hat=jnp.zeros(spectral),
            u_hat=jnp.zeros((3, *spectral)),
            b_hat=jnp.zeros((3, *spectral)),
        )
        trajectory = evolve_if_rk3(
            state,
            nonlinear,
            zero_decay,
            dt=dt,
            steps=int(t_end / dt),
            save_every=int(t_end / dt),
        )
        return jax.tree.map(lambda leaf: leaf[-1], trajectory.states)

    reference = incompressible_final()
    scale = float(jnp.max(jnp.abs(reference.v_hat)))

    solenoidal_errors, compressive_errors = [], []
    for sound_speed in (1.0, 2.0, 4.0):  # Mach ~ 0.3, 0.15, 0.07
        final = compressible_final(sound_speed)
        difference = final.u_hat - reference.v_hat
        solenoidal = mhd3d.project(difference, k)
        solenoidal_errors.append(float(jnp.max(jnp.abs(solenoidal))) / scale)
        compressive_errors.append(
            float(jnp.max(jnp.abs(difference - solenoidal))) / scale
        )

    # The incompressible content converges: the solenoidal velocity of the
    # compressible run matches the incompressible module below one percent
    # at every Mach, decreasing monotonically. The compressive residual is
    # the free acoustic transient of the unbalanced start: it shrinks with
    # Mach but carries an oscillation phase, so it gets a monotonicity
    # gate, not a power law. A pseudosound-balanced initialization would
    # recover the clean Mach-squared law and is recorded as a refinement.
    assert all(error < 5.0e-3 for error in solenoidal_errors)
    assert solenoidal_errors[0] > solenoidal_errors[2]
    assert compressive_errors[0] > compressive_errors[1] > compressive_errors[2]
