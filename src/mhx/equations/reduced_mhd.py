"""Periodic pseudo-spectral reduced-MHD equations."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from mhx.numerics.spectral import gradient, inverse_laplacian, laplacian
from mhx.state import ReducedMHDParams, ReducedMHDState


def poisson_bracket(a: Array, b: Array, *, lengths: tuple[float, float]) -> Array:
    r"""Return the 2D Poisson bracket ``[a,b] = a_x b_y - a_y b_x``."""
    da_dx, da_dy = gradient(a, lengths=lengths)
    db_dx, db_dy = gradient(b, lengths=lengths)
    return da_dx * db_dy - da_dy * db_dx


def stream_function(omega: Array, *, lengths: tuple[float, float]) -> Array:
    r"""Solve ``∇² φ = ω`` with zero mean."""
    return inverse_laplacian(omega, lengths=lengths)


def current_density(psi: Array, *, lengths: tuple[float, float]) -> Array:
    r"""Return ``j_z = -∇² ψ``."""
    return -laplacian(psi, lengths=lengths)


def reduced_mhd_rhs(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
) -> ReducedMHDState:
    r"""Return the resistive-viscous reduced-MHD right-hand side.

    The convention is

    ``ψ_t + [φ, ψ] = η ∇²ψ``

    ``ω_t + [φ, ω] = [ψ, ∇²ψ] + ν ∇²ω``

    with ``∇²φ = ω`` on a periodic domain.
    """
    phi = stream_function(state.omega, lengths=lengths)
    lap_psi = laplacian(state.psi, lengths=lengths)
    lap_omega = laplacian(state.omega, lengths=lengths)
    dpsi = -poisson_bracket(phi, state.psi, lengths=lengths) + params.resistivity * lap_psi
    domega = (
        -poisson_bracket(phi, state.omega, lengths=lengths)
        + poisson_bracket(state.psi, lap_psi, lengths=lengths)
        + params.viscosity * lap_omega
    )
    return ReducedMHDState(psi=dpsi, omega=domega)


def reduced_mhd_residual_norm(state: ReducedMHDState) -> Array:
    """Return a scalar finite-state sanity norm for debugging."""
    return jnp.sqrt(jnp.mean(state.psi**2) + jnp.mean(state.omega**2))

