"""PyTree-compatible reduced-MHD state containers."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array


class ReducedMHDState(NamedTuple):
    """2D reduced-MHD state using magnetic flux ``psi`` and vorticity ``omega``."""

    psi: Array
    omega: Array


class ReducedMHDParams(NamedTuple):
    """Diffusivities for resistive-viscous reduced MHD."""

    resistivity: float
    viscosity: float


class ReducedMHDTrajectory(NamedTuple):
    """Saved reduced-MHD trajectory samples."""

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
