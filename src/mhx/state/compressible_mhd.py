"""PyTree-compatible state containers for smooth compressible MHD examples."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array


class CompressibleMHDState(NamedTuple):
    """2D conservative compressible-MHD state with a passive dye scalar."""

    density: Array
    momentum_x: Array
    momentum_y: Array
    total_energy: Array
    magnetic_x: Array
    magnetic_y: Array
    dye_density: Array


class CompressibleMHDParams(NamedTuple):
    """Parameters for the smooth pedagogical compressible-MHD model."""

    gamma: float = 5.0 / 3.0
    density_floor: float = 1.0e-8
    pressure_floor: float = 1.0e-8


class CompressibleMHDPrimitive(NamedTuple):
    """Primitive variables for 2D compressible MHD."""

    density: Array
    velocity_x: Array
    velocity_y: Array
    pressure: Array
    magnetic_x: Array
    magnetic_y: Array
    dye: Array


class CompressibleMHDTrajectory(NamedTuple):
    """Saved compressible-MHD trajectory samples."""

    times: Array
    states: CompressibleMHDState


def conservative_from_primitive(
    primitive: CompressibleMHDPrimitive,
    params: CompressibleMHDParams,
) -> CompressibleMHDState:
    """Convert primitive variables to conservative variables."""
    density = jnp.maximum(primitive.density, params.density_floor)
    kinetic = 0.5 * density * (primitive.velocity_x**2 + primitive.velocity_y**2)
    magnetic = 0.5 * (primitive.magnetic_x**2 + primitive.magnetic_y**2)
    pressure = jnp.maximum(primitive.pressure, params.pressure_floor)
    total_energy = pressure / (params.gamma - 1.0) + kinetic + magnetic
    return CompressibleMHDState(
        density=density,
        momentum_x=density * primitive.velocity_x,
        momentum_y=density * primitive.velocity_y,
        total_energy=total_energy,
        magnetic_x=primitive.magnetic_x,
        magnetic_y=primitive.magnetic_y,
        dye_density=density * primitive.dye,
    )


def primitive_from_conservative(
    state: CompressibleMHDState,
    params: CompressibleMHDParams,
) -> CompressibleMHDPrimitive:
    """Convert conservative variables to primitive variables with floors."""
    density = jnp.maximum(state.density, params.density_floor)
    velocity_x = state.momentum_x / density
    velocity_y = state.momentum_y / density
    kinetic = 0.5 * density * (velocity_x**2 + velocity_y**2)
    magnetic = 0.5 * (state.magnetic_x**2 + state.magnetic_y**2)
    pressure = (params.gamma - 1.0) * (state.total_energy - kinetic - magnetic)
    pressure = jnp.maximum(pressure, params.pressure_floor)
    dye = state.dye_density / density
    return CompressibleMHDPrimitive(
        density=density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=pressure,
        magnetic_x=state.magnetic_x,
        magnetic_y=state.magnetic_y,
        dye=dye,
    )
