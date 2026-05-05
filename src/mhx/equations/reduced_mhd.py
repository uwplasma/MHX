"""Periodic pseudo-spectral reduced-MHD equations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from mhx.numerics import MatrixFreeOperator
from mhx.numerics.spectral import gradient, inverse_laplacian, laplacian
from mhx.physics import PhysicsTerm, apply_physics_terms
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    flatten_reduced_mhd_state,
    reduced_mhd_state_size,
    unflatten_reduced_mhd_state,
)


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
    terms: tuple[PhysicsTerm, ...] = (),
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
    base_rhs = ReducedMHDState(psi=dpsi, omega=domega)
    return apply_physics_terms(base_rhs, terms, state, params, lengths=lengths)


def linearized_reduced_mhd_rhs(
    state: ReducedMHDState,
    perturbation: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
) -> ReducedMHDState:
    """Return the matrix-free Jacobian-vector product of the reduced-MHD RHS.

    This computes ``dF(state)[perturbation]`` with JAX forward-mode automatic
    differentiation, where ``F`` is :func:`reduced_mhd_rhs`. It is the first
    building block for eigenvalue and adjoint tearing-mode benchmarks.
    """

    def rhs_for_jvp(active_state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(active_state, params, lengths=lengths, terms=terms)

    _, tangent = jax.jvp(rhs_for_jvp, (state,), (perturbation,))
    return tangent


def finite_difference_linearized_reduced_mhd_rhs(
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
    rhs_plus = reduced_mhd_rhs(plus, params, lengths=lengths, terms=terms)
    rhs_minus = reduced_mhd_rhs(minus, params, lengths=lengths, terms=terms)
    scale = 0.5 / epsilon
    return ReducedMHDState(
        psi=scale * (rhs_plus.psi - rhs_minus.psi),
        omega=scale * (rhs_plus.omega - rhs_minus.omega),
    )


def linearized_reduced_mhd_operator(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
) -> MatrixFreeOperator:
    """Return a flattened matrix-free linearized reduced-MHD operator."""
    shape = tuple(int(item) for item in state.psi.shape)
    operator_shape = (reduced_mhd_state_size(shape),)

    def matvec(vector: Array) -> Array:
        perturbation = unflatten_reduced_mhd_state(vector, shape)
        tangent = linearized_reduced_mhd_rhs(
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
        name="linearized_reduced_mhd",
    )


def reduced_mhd_residual_norm(state: ReducedMHDState) -> Array:
    """Return a scalar finite-state sanity norm for debugging."""
    return jnp.sqrt(jnp.mean(state.psi**2) + jnp.mean(state.omega**2))
