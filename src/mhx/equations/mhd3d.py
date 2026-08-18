"""Incompressible visco-resistive 3D MHD in a periodic box.

The model, in Alfvén units with $\\nabla\\cdot\\mathbf{v} =
\\nabla\\cdot\\mathbf{B} = 0$:

.. math::

    \\partial_t \\mathbf{v} = \\mathcal{P}\\left[
        \\mathbf{v}\\times\\boldsymbol{\\omega}
        + \\mathbf{j}\\times\\mathbf{B}\\right] + \\nu\\nabla^2\\mathbf{v},
    \\qquad
    \\partial_t \\mathbf{B} = \\nabla\\times(\\mathbf{v}\\times\\mathbf{B})
        + \\eta\\nabla^2\\mathbf{B},

where :math:`\\mathcal{P}` is the spectral Leray projector. The dissipative
terms are not part of :func:`mhd3d_nonlinear`: the low-storage integrator
applies them exactly through integrating factors.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array

from mhx.numerics.spectral.pfft import pfft3, pifft3, spectral_shape
from mhx.state.mhd3d import MHD3DParams, MHD3DState


def wavevectors(
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
) -> Array:
    """Return the wavevector array ``k`` of shape ``(3, nx, ny, nzr)``."""
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(shape[0], d=1.0 / shape[0]) / lengths[0]
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(shape[1], d=1.0 / shape[1]) / lengths[1]
    kz = 2.0 * jnp.pi * jnp.fft.rfftfreq(shape[2], d=1.0 / shape[2]) / lengths[2]
    return jnp.stack(
        jnp.meshgrid(kx, ky, kz, indexing="ij"),
        axis=0,
    )


def two_thirds_mask_rfft(shape: tuple[int, int, int]) -> Array:
    """Two-thirds dealiasing mask on the half-spectrum grid."""
    keep = []
    for axis, points in enumerate(shape):
        limit = points // 3
        if axis < 2:
            modes = jnp.abs(jnp.fft.fftfreq(points, d=1.0 / points))
        else:
            modes = jnp.fft.rfftfreq(points, d=1.0 / points)
        keep.append(modes <= limit)
    return (
        keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]
    )


def parseval_weight(shape: tuple[int, int, int]) -> Array:
    """Mode weights so half-spectrum sums equal full-spectrum sums.

    Interior ``kz`` modes represent themselves and their conjugates, so
    they count twice. The ``kz = 0`` plane, and the Nyquist plane for even
    ``nz``, count once.
    """
    nzr = shape[2] // 2 + 1
    weight = 2.0 * jnp.ones(nzr)
    weight = weight.at[0].set(1.0)
    if shape[2] % 2 == 0:
        weight = weight.at[-1].set(1.0)
    return weight[None, None, :]


def curl_hat(field_hat: Array, k: Array) -> Array:
    """Spectral curl ``i k x f`` for component-first fields."""
    return 1j * jnp.cross(
        jnp.broadcast_to(k, field_hat.shape),
        field_hat,
        axis=0,
    )


def project(field_hat: Array, k: Array) -> Array:
    """Leray projection: remove the compressive part of a vector field."""
    k_squared = jnp.sum(k * k, axis=0)
    safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    divergence = jnp.sum(k * field_hat, axis=0)
    return field_hat - k * (divergence / safe)[None]


def divergence_linf(field_hat: Array, k: Array) -> Array:
    """Return the maximum spectral divergence magnitude."""
    return jnp.max(jnp.abs(jnp.sum(k * field_hat, axis=0)))


def mhd3d_nonlinear(
    state: MHD3DState,
    params: MHD3DParams,
    *,
    shape: tuple[int, int, int],
    k: Array,
    mask: Array,
    mesh: Mesh | None = None,
) -> MHD3DState:
    """Return the projected, dealiased nonlinear right-hand side.

    The guide field enters by addition to the real-space magnetic field,
    which reproduces both the ``j x B0`` force and the ``B0``-advection
    term in the induction equation without separate terms.
    """
    fields_hat = jnp.stack(
        (
            state.v_hat,
            state.b_hat,
            curl_hat(state.v_hat, k),
            curl_hat(state.b_hat, k),
        )
    )
    fields = pifft3(fields_hat, shape=shape, mesh=mesh)
    velocity, magnetic, vorticity, current = fields

    guide = jnp.asarray(params.guide_field, dtype=fields.dtype)
    total_b = magnetic + guide[:, None, None, None]

    momentum = jnp.cross(velocity, vorticity, axis=0) + jnp.cross(
        current, total_b, axis=0
    )
    electric = jnp.cross(velocity, total_b, axis=0)

    products_hat = pfft3(jnp.stack((momentum, electric)), mesh=mesh)
    momentum_hat = products_hat[0] * mask[None]
    electric_hat = products_hat[1] * mask[None]

    return MHD3DState(
        v_hat=project(momentum_hat, k),
        b_hat=curl_hat(electric_hat, k),
    )


def decay_rates(
    params: MHD3DParams,
    k: Array,
) -> MHD3DState:
    """Return the diagonal dissipation rates for the integrating factor."""
    k_squared = jnp.sum(k * k, axis=0)
    dissipation = k_squared ** params.dissipation_order
    return MHD3DState(
        v_hat=params.viscosity * dissipation[None],
        b_hat=params.resistivity * dissipation[None],
    )


def to_spectral(fields: Array, *, mesh: Mesh | None = None) -> Array:
    """Transform component-first real fields to the half spectrum."""
    return pfft3(fields, mesh=mesh)


def to_physical(
    fields_hat: Array,
    *,
    shape: tuple[int, int, int],
    mesh: Mesh | None = None,
) -> Array:
    """Transform component-first half-spectrum fields to real space."""
    return pifft3(fields_hat, shape=shape, mesh=mesh)


def energies(
    state: MHD3DState,
    *,
    shape: tuple[int, int, int],
) -> dict[str, Array]:
    """Mean kinetic and magnetic energy densities and helicities.

    Uses Parseval's identity on the half spectrum with conjugate weights.
    Magnetic helicity uses the Coulomb-gauge vector potential
    ``a_hat = i k x b_hat / k^2``.
    """
    weight = parseval_weight(shape)
    volume_sq = float(shape[0] * shape[1] * shape[2]) ** 2

    def mean_product(left: Array, right: Array) -> Array:
        return (
            jnp.sum(weight * jnp.real(left * jnp.conj(right))) / volume_sq
        )

    kinetic = 0.5 * mean_product(state.v_hat, state.v_hat)
    magnetic = 0.5 * mean_product(state.b_hat, state.b_hat)
    cross = mean_product(state.v_hat, state.b_hat)
    return {
        "kinetic": kinetic,
        "magnetic": magnetic,
        "total": kinetic + magnetic,
        "cross_helicity": 0.5 * cross,
    }


def magnetic_helicity(
    state: MHD3DState,
    *,
    shape: tuple[int, int, int],
    k: Array,
) -> Array:
    """Mean magnetic helicity ``<a . b>`` in the Coulomb gauge."""
    k_squared = jnp.sum(k * k, axis=0)
    safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    a_hat = curl_hat(state.b_hat, k) / safe[None]
    weight = parseval_weight(shape)
    volume_sq = float(shape[0] * shape[1] * shape[2]) ** 2
    return jnp.sum(weight * jnp.real(a_hat * jnp.conj(state.b_hat))) / volume_sq


__all__ = [
    "MHD3DParams",
    "MHD3DState",
    "curl_hat",
    "decay_rates",
    "divergence_linf",
    "energies",
    "magnetic_helicity",
    "mhd3d_nonlinear",
    "parseval_weight",
    "project",
    "spectral_shape",
    "to_physical",
    "to_spectral",
    "two_thirds_mask_rfft",
    "wavevectors",
]
