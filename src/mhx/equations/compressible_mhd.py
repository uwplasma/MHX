"""Smooth periodic compressible-MHD equations for pedagogical examples.

The implementation uses conservative ideal-MHD fluxes with Fourier
derivatives.  It is intended for smooth, low-Mach tutorial runs and gradient
experiments.  It is not a shock-capturing production MHD solver.
"""

from __future__ import annotations

import jax.numpy as jnp

from mhx.numerics.spectral import fft_derivative
from mhx.state import (
    CompressibleMHDParams,
    CompressibleMHDPrimitive,
    CompressibleMHDState,
    conservative_from_primitive,
    primitive_from_conservative,
)


def compressible_mhd_rhs(
    state: CompressibleMHDState,
    params: CompressibleMHDParams,
    *,
    lengths: tuple[float, float],
) -> CompressibleMHDState:
    """Return the smooth conservative ideal-MHD RHS on a periodic domain."""
    primitive = primitive_from_conservative(state, params)
    flux_x, flux_y = compressible_mhd_fluxes(primitive, state)

    def divergence(x_flux, y_flux):
        return fft_derivative(x_flux, axis=0, length=lengths[0]) + fft_derivative(
            y_flux,
            axis=1,
            length=lengths[1],
        )

    return CompressibleMHDState(
        density=-divergence(flux_x.density, flux_y.density),
        momentum_x=-divergence(flux_x.momentum_x, flux_y.momentum_x),
        momentum_y=-divergence(flux_x.momentum_y, flux_y.momentum_y),
        total_energy=-divergence(flux_x.total_energy, flux_y.total_energy),
        magnetic_x=-divergence(flux_x.magnetic_x, flux_y.magnetic_x),
        magnetic_y=-divergence(flux_x.magnetic_y, flux_y.magnetic_y),
        dye_density=-divergence(flux_x.dye_density, flux_y.dye_density),
    )


def compressible_mhd_fluxes(
    primitive: CompressibleMHDPrimitive,
    state: CompressibleMHDState,
) -> tuple[CompressibleMHDState, CompressibleMHDState]:
    """Return ideal-MHD conservative fluxes in x and y."""
    velocity_dot_magnetic = (
        primitive.velocity_x * primitive.magnetic_x
        + primitive.velocity_y * primitive.magnetic_y
    )
    magnetic_pressure = 0.5 * (primitive.magnetic_x**2 + primitive.magnetic_y**2)
    total_pressure = primitive.pressure + magnetic_pressure
    total_enthalpy = state.total_energy + total_pressure
    induction_flux = primitive.velocity_x * primitive.magnetic_y - primitive.velocity_y * (
        primitive.magnetic_x
    )
    flux_x = CompressibleMHDState(
        density=state.momentum_x,
        momentum_x=state.momentum_x * primitive.velocity_x + total_pressure
        - primitive.magnetic_x**2,
        momentum_y=state.momentum_y * primitive.velocity_x
        - primitive.magnetic_x * primitive.magnetic_y,
        total_energy=total_enthalpy * primitive.velocity_x
        - velocity_dot_magnetic * primitive.magnetic_x,
        magnetic_x=jnp.zeros_like(primitive.magnetic_x),
        magnetic_y=induction_flux,
        dye_density=state.dye_density * primitive.velocity_x,
    )
    flux_y = CompressibleMHDState(
        density=state.momentum_y,
        momentum_x=state.momentum_x * primitive.velocity_y
        - primitive.magnetic_x * primitive.magnetic_y,
        momentum_y=state.momentum_y * primitive.velocity_y + total_pressure
        - primitive.magnetic_y**2,
        total_energy=total_enthalpy * primitive.velocity_y
        - velocity_dot_magnetic * primitive.magnetic_y,
        magnetic_x=-induction_flux,
        magnetic_y=jnp.zeros_like(primitive.magnetic_y),
        dye_density=state.dye_density * primitive.velocity_y,
    )
    return flux_x, flux_y


def uniform_compressible_mhd_state(
    *,
    shape: tuple[int, int],
    density: float = 1.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    pressure: float = 1.0,
    magnetic: tuple[float, float] = (0.0, 0.0),
    dye: float = 0.0,
    params: CompressibleMHDParams | None = None,
) -> CompressibleMHDState:
    """Return a uniform conservative state for tests and sanity checks."""
    active_params = params or CompressibleMHDParams()
    primitive = CompressibleMHDPrimitive(
        density=jnp.full(shape, density),
        velocity_x=jnp.full(shape, velocity[0]),
        velocity_y=jnp.full(shape, velocity[1]),
        pressure=jnp.full(shape, pressure),
        magnetic_x=jnp.full(shape, magnetic[0]),
        magnetic_y=jnp.full(shape, magnetic[1]),
        dye=jnp.full(shape, dye),
    )
    return conservative_from_primitive(primitive, active_params)
