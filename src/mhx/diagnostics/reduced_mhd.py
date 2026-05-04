"""Diagnostics for reduced-MHD states and trajectories."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from mhx.equations.reduced_mhd import stream_function
from mhx.numerics.spectral import gradient
from mhx.state import ReducedMHDState, ReducedMHDTrajectory


def magnetic_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    r"""Return mean magnetic perturbation energy ``0.5 <|∇ψ|²>``."""
    grad_psi = gradient(state.psi, lengths=lengths)
    return 0.5 * sum(jnp.mean(component**2) for component in grad_psi)


def kinetic_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    r"""Return mean kinetic energy ``0.5 <|∇φ|²>``."""
    phi = stream_function(state.omega, lengths=lengths)
    grad_phi = gradient(phi, lengths=lengths)
    return 0.5 * sum(jnp.mean(component**2) for component in grad_phi)


def total_energy(state: ReducedMHDState, *, lengths: tuple[float, float]) -> Array:
    """Return reduced-MHD magnetic plus kinetic energy."""
    return magnetic_energy(state, lengths=lengths) + kinetic_energy(state, lengths=lengths)


def trajectory_energies(
    trajectory: ReducedMHDTrajectory,
    *,
    lengths: tuple[float, float],
) -> dict[str, Array]:
    """Return energy time series for a saved trajectory."""
    magnetic = jnp.asarray(
        [magnetic_energy(state, lengths=lengths) for state in _iter_states(trajectory.states)]
    )
    kinetic = jnp.asarray(
        [kinetic_energy(state, lengths=lengths) for state in _iter_states(trajectory.states)]
    )
    return {
        "time": trajectory.times,
        "magnetic": magnetic,
        "kinetic": kinetic,
        "total": magnetic + kinetic,
    }


def _iter_states(states: ReducedMHDState):
    for index in range(states.psi.shape[0]):
        yield ReducedMHDState(psi=states.psi[index], omega=states.omega[index])

