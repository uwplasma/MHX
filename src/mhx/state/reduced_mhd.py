"""PyTree-compatible reduced-MHD state containers."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array


class ReducedMHDState(NamedTuple):
    """Two fields that define a reduced-MHD state.

    Attributes:
        psi: Magnetic flux function. Its in-plane curl gives the magnetic field.
        omega: Out-of-plane vorticity. The stream function solves
            ``laplacian(phi) = omega``.
    """

    psi: Array
    omega: Array


class ReducedMHDParams(NamedTuple):
    """Physical diffusion coefficients.

    Attributes:
        resistivity: Magnetic diffusivity in the flux equation.
        viscosity: Viscous diffusivity in the vorticity equation.
    """

    resistivity: float
    viscosity: float


class ReducedMHDTrajectory(NamedTuple):
    """States retained during a simulation.

    Attributes:
        times: One-dimensional array of saved times.
        states: Fields with a leading saved-time axis.
    """

    times: Array
    states: ReducedMHDState


def reduced_mhd_state_size(shape: tuple[int, int]) -> int:
    """Return flattened vector size for one reduced-MHD state on ``shape``."""
    return 2 * shape[0] * shape[1]


def flatten_reduced_mhd_state(state: ReducedMHDState) -> Array:
    """Flatten ``(psi, omega)`` into one deterministic 1D vector."""
    return jnp.concatenate((jnp.ravel(state.psi), jnp.ravel(state.omega)))


def unflatten_reduced_mhd_state(vector: Array, shape: tuple[int, int]) -> ReducedMHDState:
    """Reconstruct ``ReducedMHDState`` from a flattened vector and grid shape."""
    expected_size = reduced_mhd_state_size(shape)
    if vector.size != expected_size:
        raise ValueError(f"expected vector size {expected_size}, got {vector.size}")
    split = shape[0] * shape[1]
    return ReducedMHDState(
        psi=jnp.reshape(vector[:split], shape),
        omega=jnp.reshape(vector[split:], shape),
    )
