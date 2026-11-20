"""Physics-anchored tests for the 3D paths the gate suite leaves open."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import mhx
from mhx.equations import mhd3d
from mhx.physics.equilibria3d import ABCFlowEquilibrium, TaylorGreenEquilibrium
from mhx.state.mhd3d import MHD3DState
from mhx.time_integrators.exponential import evolve_etdrk4
from mhx.time_integrators.low_storage import evolve_if_rk3

LENGTHS = (2.0 * jnp.pi,) * 3


def test_taylor_green_energies_match_the_exact_means() -> None:
    """Lee et al. (2010) class-I fields have exact mean energies.

    The Taylor--Green velocity gives ``E_V = v0^2 / 8`` and the insulating
    magnetic field gives ``E_M = 3 b0^2 / 8``, so the paper's equal-energy
    start ``E_V = E_M = 0.125`` requires ``b0 = 1/sqrt(3)``.
    """
    shape = (16, 16, 16)
    velocity, magnetic = TaylorGreenEquilibrium(
        v0=1.0, b0=1.0 / jnp.sqrt(3.0)
    ).initial_fields(shape)
    state = MHD3DState(
        v_hat=mhd3d.to_spectral(velocity),
        b_hat=mhd3d.to_spectral(magnetic),
    )
    values = mhd3d.energies(state, shape=shape)
    assert abs(float(values["kinetic"]) - 0.125) < 1.0e-12
    assert abs(float(values["magnetic"]) - 0.125) < 1.0e-12


def test_abc_beltrami_magnetic_helicity_identity() -> None:
    """A unit-wavenumber Beltrami field has ``<a . b> = 2 E_M`` exactly.

    The 1:1:1 ABC field satisfies ``curl b = b``, so its Coulomb-gauge
    vector potential equals the field itself and the helicity saturates
    its realizability bound at ``k = 1``.
    """
    shape = (16, 16, 16)
    abc_velocity, _ = ABCFlowEquilibrium().initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    state = MHD3DState(
        v_hat=jnp.zeros((3, *mhd3d.spectral_shape(shape)), dtype=complex),
        b_hat=mhd3d.to_spectral(abc_velocity),
    )
    curl = mhd3d.curl_hat(state.b_hat, k)
    assert float(jnp.max(jnp.abs(curl - state.b_hat))) < 1.0e-10

    helicity = float(mhd3d.magnetic_helicity(state, shape=shape, k=k))
    energy = float(mhd3d.energies(state, shape=shape)["magnetic"])
    assert abs(helicity - 2.0 * energy) < 1.0e-10


def test_parseval_weight_handles_odd_last_axis() -> None:
    shape = (8, 8, 9)
    field = jax.random.normal(jax.random.PRNGKey(0), (3, *shape))
    state = MHD3DState(
        v_hat=mhd3d.to_spectral(field),
        b_hat=mhd3d.to_spectral(field),
    )
    energy = mhd3d.energies(state, shape=shape)
    direct = 0.5 * float(jnp.mean(jnp.sum(field * field, axis=0)))
    assert abs(float(energy["kinetic"]) - direct) < 1.0e-10


@pytest.mark.parametrize("evolve", [evolve_if_rk3, evolve_etdrk4])
def test_evolvers_validate_and_keep_the_final_partial_chunk(evolve) -> None:
    shape = (8, 8, 8)
    velocity, magnetic = TaylorGreenEquilibrium().initial_fields(shape)
    k = mhd3d.wavevectors(shape, LENGTHS)
    state0 = MHD3DState(
        v_hat=mhd3d.project(mhd3d.to_spectral(velocity), k),
        b_hat=mhd3d.project(mhd3d.to_spectral(magnetic), k),
    )
    decay = MHD3DState(
        v_hat=jnp.zeros_like(jnp.real(state0.v_hat)),
        b_hat=jnp.zeros_like(jnp.real(state0.b_hat)),
    )

    def zero(state: MHD3DState) -> MHD3DState:
        return jax.tree.map(jnp.zeros_like, state)

    with pytest.raises(ValueError, match="steps"):
        evolve(state0, zero, decay, dt=1.0e-2, steps=0)
    with pytest.raises(ValueError, match="dt"):
        evolve(state0, zero, decay, dt=-1.0, steps=1)
    with pytest.raises(ValueError, match="save_every"):
        evolve(state0, zero, decay, dt=1.0e-2, steps=1, save_every=0)

    # Seven steps saved every three: two full chunks plus the final state.
    trajectory = evolve(state0, zero, decay, dt=1.0e-2, steps=7, save_every=3)
    assert trajectory.times.shape[0] == 3
    assert abs(float(trajectory.times[-1]) - 0.07) < 1.0e-12


def test_mhd3d_sharded_simulation_matches_single_device() -> None:
    """The device_count path through Simulation agrees with one device."""
    if jax.device_count() < 2:
        pytest.skip("needs the multi-device conftest environment")

    def run(count):
        return mhx.Simulation(
            shape=(8, 8, 8),
            equations="mhd3d",
            equilibrium=mhx.OrszagTang3DEquilibrium(),
            viscosity=5.0e-3,
            resistivity=5.0e-3,
            dt=1.0e-2,
            t_end=0.05,
            save_every=5,
            device_count=count,
            verbose=False,
        ).run()

    single = run(None)
    sharded = run(2)
    assert sharded.device_count == 2
    gap = abs(
        single.diagnostics["final_total_energy"]
        - sharded.diagnostics["final_total_energy"]
    )
    assert gap < 1.0e-12

    with pytest.raises(ValueError, match="divide"):
        mhx.Simulation(
            shape=(9, 8, 8),
            equations="mhd3d",
            equilibrium=mhx.OrszagTang3DEquilibrium(),
            dt=1.0e-2,
            t_end=0.05,
            device_count=2,
            verbose=False,
        ).run()


def test_mhd3d_result_print_summary(capsys) -> None:
    result = mhx.Simulation(
        shape=(8, 8, 8),
        equations="mhd3d",
        equilibrium=mhx.OrszagTang3DEquilibrium(),
        dt=1.0e-2,
        t_end=0.02,
        save_every=1,
        verbose=False,
    ).run()
    result.print_summary()
    captured = capsys.readouterr().out
    assert "MHX 3D result" in captured
    assert "div B" in captured