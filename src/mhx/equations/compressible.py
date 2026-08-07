"""Subsonic compressible MHD, spectral and smooth, per plan_3d.md Track C.

Isothermal closure with log density, following the Pencil Code and
Shebalin precedents: positivity holds without floors, so the model stays
differentiable. In Alfvén units with sound speed ``c_s``:

.. math::

    \\partial_t \\ln\\rho = -\\mathbf{u}\\cdot\\nabla\\ln\\rho
        - \\nabla\\cdot\\mathbf{u},

.. math::

    \\partial_t \\mathbf{u} = -\\boldsymbol{\\omega}\\times\\mathbf{u}
        - \\nabla\\tfrac{u^2}{2} - c_s^2\\nabla\\ln\\rho
        + \\frac{\\mathbf{j}\\times\\mathbf{B}}{\\rho}
        + \\nu\\left(\\nabla^2\\mathbf{u}
        + \\tfrac{1}{3}\\nabla\\nabla\\cdot\\mathbf{u}\\right)
        + \\nu_b\\nabla\\nabla\\cdot\\mathbf{u},

.. math::

    \\partial_t \\mathbf{B} = \\nabla\\times(\\mathbf{u}\\times\\mathbf{B})
        + \\eta\\nabla^2\\mathbf{B}.

The validity boundary is stated, not hidden: smooth subsonic flows, Mach
below about one half, no shock capturing (Dahlburg--Picone 1990 lineage).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array

from mhx.equations.mhd3d import curl_hat, divergence_linf, wavevectors
from mhx.numerics.spectral.pfft import pfft3, pifft3


class CompressibleState(NamedTuple):
    """Half-spectrum log density, velocity, and magnetic field."""

    lnrho_hat: Array
    u_hat: Array
    b_hat: Array


class CompressibleParams(NamedTuple):
    """Differentiable parameters of the isothermal compressible model."""

    sound_speed: Array | float
    viscosity: Array | float
    bulk_viscosity: Array | float
    resistivity: Array | float
    guide_field: tuple[float, float, float] = (0.0, 0.0, 0.0)


def compressible_nonlinear(
    state: CompressibleState,
    params: CompressibleParams,
    *,
    shape: tuple[int, int, int],
    k: Array,
    mask: Array,
    mesh: Mesh | None = None,
) -> CompressibleState:
    """Return the full explicit right-hand side, dealiased."""
    lnrho = pifft3(state.lnrho_hat, shape=shape, mesh=mesh)
    fields_hat = jnp.stack((state.u_hat, state.b_hat, curl_hat(state.b_hat, k)))
    fields = pifft3(fields_hat, shape=shape, mesh=mesh)
    velocity, magnetic, current = fields

    grad_lnrho = pifft3(1j * k * state.lnrho_hat[None], shape=shape, mesh=mesh)
    vorticity = pifft3(curl_hat(state.u_hat, k), shape=shape, mesh=mesh)

    guide = jnp.asarray(params.guide_field, dtype=fields.dtype)
    total_b = magnetic + guide[:, None, None, None]
    inverse_rho = jnp.exp(-lnrho)

    advect_lnrho = -jnp.sum(velocity * grad_lnrho, axis=0)
    momentum = (
        -jnp.cross(vorticity, velocity, axis=0)
        + jnp.cross(current, total_b, axis=0) * inverse_rho[None]
    )
    electric = jnp.cross(velocity, total_b, axis=0)
    kinetic = 0.5 * jnp.sum(velocity * velocity, axis=0)

    products_hat = pfft3(
        jnp.concatenate(
            (advect_lnrho[None], kinetic[None], momentum, electric)
        ),
        mesh=mesh,
    )
    advect_lnrho_hat = products_hat[0] * mask
    kinetic_hat = products_hat[1] * mask
    momentum_hat = products_hat[2:5] * mask[None]
    electric_hat = products_hat[5:8] * mask[None]

    divergence_u = jnp.sum(1j * k * state.u_hat, axis=0)
    k_squared = jnp.sum(k * k, axis=0)

    rhs_lnrho = advect_lnrho_hat - divergence_u * mask
    rhs_u = (
        momentum_hat
        - 1j * k * kinetic_hat[None]
        - params.sound_speed**2 * 1j * k * (state.lnrho_hat * mask)[None]
        - params.viscosity * k_squared[None] * state.u_hat
        # The compressive stress is +(nu/3 + nu_b) grad div u, which in
        # spectral form is (i k)(i k . u) = -k (k . u): the plus sign here
        # composes with the two factors of i to give exactly that.
        + (params.viscosity / 3.0 + params.bulk_viscosity)
        * 1j
        * k
        * divergence_u[None]
    )
    rhs_b = curl_hat(electric_hat, k) - params.resistivity * k_squared[None] * state.b_hat
    return CompressibleState(lnrho_hat=rhs_lnrho, u_hat=rhs_u, b_hat=rhs_b)


def linear_block(
    k_vector: tuple[float, float, float],
    params: CompressibleParams,
):
    """Exact 7x7 linearization about rest for one Fourier mode.

    Row order: ``(lnrho, u_x, u_y, u_z, b_x, b_y, b_z)``. The eigenvalues
    of this block are the exact complex frequencies of the fast, slow,
    Alfvén, and damped compressive modes, which gate C1 fits against.
    """
    import numpy as np

    k_arr = np.asarray(k_vector, dtype=float)
    b0 = np.asarray(params.guide_field, dtype=float)
    nu = float(params.viscosity)
    nu_b = float(params.bulk_viscosity)
    eta = float(params.resistivity)
    cs2 = float(params.sound_speed) ** 2
    k2 = float(k_arr @ k_arr)

    matrix = np.zeros((7, 7), dtype=complex)
    # d lnrho/dt = -i k . u
    matrix[0, 1:4] = -1j * k_arr
    # d u/dt = -cs^2 i k lnrho + i (k x b) x B0 - nu k^2 u
    #          - (nu/3 + nu_b) k (k . u)
    matrix[1:4, 0] = -cs2 * 1j * k_arr
    for row in range(3):
        for col in range(3):
            matrix[1 + row, 1 + col] = (
                -nu * k2 * (row == col)
                - (nu / 3.0 + nu_b) * k_arr[row] * k_arr[col]
            )
    # (i k x b) x B0 acting on b components.
    for col in range(3):
        unit = np.zeros(3)
        unit[col] = 1.0
        force = np.cross(1j * k_arr, unit)
        matrix[1:4, 4 + col] = np.cross(force, b0)
    # d b/dt = i k x (u x B0) - eta k^2 b
    for col in range(3):
        unit = np.zeros(3)
        unit[col] = 1.0
        induction = np.cross(1j * k_arr, np.cross(unit, b0))
        matrix[4:7, 1 + col] = induction
    matrix[4:7, 4:7] -= eta * k2 * np.eye(3)
    return matrix


def mass_density(state: CompressibleState, *, shape, mesh=None) -> Array:
    """Return the real-space density field."""
    return jnp.exp(pifft3(state.lnrho_hat, shape=shape, mesh=mesh))


__all__ = [
    "CompressibleParams",
    "CompressibleState",
    "compressible_nonlinear",
    "divergence_linf",
    "linear_block",
    "mass_density",
    "wavevectors",
]
