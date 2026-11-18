"""MHX adapters and preconditioners for SOLVAX."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import solvax
from jaxtyping import Array

from mhx.numerics.linear_operator import MatrixFreeOperator
from mhx.numerics.spectral import spectral_wavenumbers
from mhx.state import ReducedMHDParams, ReducedMHDState


def as_solvax_operator(operator: MatrixFreeOperator) -> Any:
    """Adapt a flat MHX operator to :class:`solvax.MatrixFreeOperator`.

    Args:
        operator: MHX operator whose input and output are one-dimensional.

    Returns:
        A SOLVAX operator with a square matrix shape.
    """
    size = 1
    for extent in operator.shape:
        size *= extent
    if operator.shape != (size,):
        raise ValueError(
            "SOLVAX's algebraic operator adapter requires a flat MHX vector shape, "
            f"got {operator.shape}"
        )
    return solvax.MatrixFreeOperator(operator, shape=(size, size))


def complex_linear_extension(
    operator: Callable[[Array], Array],
) -> Callable[[Array], Array]:
    """Extend a real-linear JAX operator to complex trial vectors."""

    def apply(vector: Array) -> Array:
        return operator(jnp.real(vector)) + 1j * operator(jnp.imag(vector))

    return apply


def spectral_diffusion_preconditioner(
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    dt: float,
) -> Callable[[ReducedMHDState], ReducedMHDState]:
    """Build an exact spectral inverse for the linear diffusion terms.

    SOLVAX calls the returned function as a right preconditioner during each
    Newton--Krylov step. MHX supplies the Fourier symbol because it follows
    directly from the reduced-MHD equations.

    Args:
        params: Resistivity and viscosity.
        lengths: Periodic domain length in each direction.
        dt: Backward-Euler step size.

    Returns:
        A function that applies the inverse diffusion operator to a state.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if len(lengths) != 2 or any(length <= 0.0 for length in lengths):
        raise ValueError("lengths must contain two positive entries")

    def solve_field(field: Array, diffusivity: float) -> Array:
        array = jnp.asarray(field)
        wave_number_squared = jnp.zeros(array.shape, dtype=array.real.dtype)
        for axis, length in enumerate(lengths):
            wavenumbers = spectral_wavenumbers(array.shape[axis], length)
            shape = [1] * array.ndim
            shape[axis] = array.shape[axis]
            wave_number_squared = wave_number_squared + jnp.reshape(
                wavenumbers**2,
                shape,
            )
        denominator = 1.0 + dt * diffusivity * wave_number_squared
        result = jnp.fft.ifftn(jnp.fft.fftn(array) / denominator)
        if jnp.isrealobj(array):
            return jnp.real(result)
        return result

    @jax.jit
    def apply(state: ReducedMHDState) -> ReducedMHDState:
        return ReducedMHDState(
            psi=solve_field(state.psi, params.resistivity),
            omega=solve_field(state.omega, params.viscosity),
        )

    return apply
