"""Periodic pseudo-spectral reduced-MHD equations with Arakawa brackets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from mhx.numerics import MatrixFreeOperator
from mhx.numerics.spectral import fft_derivative, inverse_laplacian, laplacian
from mhx.physics import PhysicsTerm, apply_physics_terms
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    flatten_reduced_mhd_state,
    reduced_mhd_state_size,
    unflatten_reduced_mhd_state,
)


def arakawa_poisson_bracket(a: Array, b: Array, *, lengths: tuple[float, float]) -> Array:
    """Return the fully conservative 2D Arakawa Poisson bracket."""
    da_dx = fft_derivative(a, axis=0, length=lengths[0])
    da_dy = fft_derivative(a, axis=1, length=lengths[1])
    db_dx = fft_derivative(b, axis=0, length=lengths[0])
    db_dy = fft_derivative(b, axis=1, length=lengths[1])

    j1 = da_dx * db_dy - da_dy * db_dx
    j2 = fft_derivative(a * db_dy, axis=0, length=lengths[0]) - fft_derivative(
        a * db_dx, axis=1, length=lengths[1]
    )
    j3 = fft_derivative(b * da_dx, axis=1, length=lengths[1]) - fft_derivative(
        b * da_dy, axis=0, length=lengths[0]
    )

    return (j1 + j2 + j3) / 3.0


def stream_function(omega: Array, *, lengths: tuple[float, float]) -> Array:
    """Solve ``∇²φ = ω`` with zero mean."""
    return inverse_laplacian(omega, lengths=lengths)


def current_density(psi: Array, *, lengths: tuple[float, float]) -> Array:
    """Return ``j_z = -∇²ψ``."""
    return -laplacian(psi, lengths=lengths)


def arakawa_reduced_mhd_rhs(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
) -> ReducedMHDState:
    """Return the resistive-viscous reduced-MHD RHS with Arakawa brackets.

    The convention is

    ``ψ_t + [φ, ψ] = η ∇²ψ``

    ``ω_t + [φ, ω] = [ψ, ∇²ψ] + ν ∇²ω``

    with ``∇²φ = ω`` on a periodic domain.
    """
    phi = stream_function(state.omega, lengths=lengths)
    lap_psi = laplacian(state.psi, lengths=lengths)
    lap_omega = laplacian(state.omega, lengths=lengths)
    dpsi = (
        -arakawa_poisson_bracket(phi, state.psi, lengths=lengths)
        + params.resistivity * lap_psi
    )
    domega = (
        -arakawa_poisson_bracket(phi, state.omega, lengths=lengths)
        + arakawa_poisson_bracket(state.psi, lap_psi, lengths=lengths)
        + params.viscosity * lap_omega
    )
    base_rhs = ReducedMHDState(psi=dpsi, omega=domega)
    return apply_physics_terms(base_rhs, terms, state, params, lengths=lengths)


def linearized_arakawa_reduced_mhd_rhs(
    state: ReducedMHDState,
    perturbation: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
) -> ReducedMHDState:
    """Return the matrix-free Jacobian-vector product of the Arakawa reduced-MHD RHS."""

    def rhs_for_jvp(active_state: ReducedMHDState) -> ReducedMHDState:
        return arakawa_reduced_mhd_rhs(active_state, params, lengths=lengths, terms=terms)

    _, tangent = jax.jvp(rhs_for_jvp, (state,), (perturbation,))
    return tangent


def finite_difference_linearized_arakawa_reduced_mhd_rhs(
    state: ReducedMHDState,
    perturbation: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    epsilon: float = 1.0e-4,
    terms: tuple[PhysicsTerm, ...] = (),
) -> ReducedMHDState:
    """Return a centered finite-difference approximation to the linearized RHS."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    plus = ReducedMHDState(
        psi=state.psi + epsilon * perturbation.psi,
        omega=state.omega + epsilon * perturbation.omega,
    )
    minus = ReducedMHDState(
        psi=state.psi - epsilon * perturbation.psi,
        omega=state.omega - epsilon * perturbation.omega,
    )
    rhs_plus = arakawa_reduced_mhd_rhs(plus, params, lengths=lengths, terms=terms)
    rhs_minus = arakawa_reduced_mhd_rhs(minus, params, lengths=lengths, terms=terms)
    scale = 0.5 / epsilon
    return ReducedMHDState(
        psi=scale * (rhs_plus.psi - rhs_minus.psi),
        omega=scale * (rhs_plus.omega - rhs_minus.omega),
    )


def linearized_arakawa_reduced_mhd_operator(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
) -> MatrixFreeOperator:
    """Return a flattened matrix-free linearized Arakawa reduced-MHD operator."""
    shape = tuple(int(item) for item in state.psi.shape)
    operator_shape = (reduced_mhd_state_size(shape),)

    def matvec(vector: Array) -> Array:
        perturbation = unflatten_reduced_mhd_state(vector, shape)
        tangent = linearized_arakawa_reduced_mhd_rhs(
            state,
            perturbation,
            params,
            lengths=lengths,
            terms=terms,
        )
        return flatten_reduced_mhd_state(tangent)

    return MatrixFreeOperator(
        shape=operator_shape,
        matvec=matvec,
        name="linearized_arakawa_reduced_mhd",
    )


__all__ = [
    "arakawa_poisson_bracket",
    "arakawa_reduced_mhd_rhs",
    "current_density",
    "finite_difference_linearized_arakawa_reduced_mhd_rhs",
    "linearized_arakawa_reduced_mhd_operator",
    "linearized_arakawa_reduced_mhd_rhs",
    "stream_function",
]
