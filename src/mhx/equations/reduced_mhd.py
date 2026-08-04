"""Periodic pseudo-spectral reduced-MHD equations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from mhx.numerics import MatrixFreeOperator
from mhx.numerics.spectral import (
    dealiased_product,
    gradient,
    inverse_laplacian,
    laplacian,
    spectral_wavenumbers,
    two_thirds_mask,
)
from mhx.physics import PhysicsTerm, apply_physics_terms
from mhx.state import (
    ReducedMHDParams,
    ReducedMHDState,
    ReducedMHDTrajectory,
    flatten_reduced_mhd_state,
    reduced_mhd_state_size,
    unflatten_reduced_mhd_state,
)


def poisson_bracket(
    a: Array,
    b: Array,
    *,
    lengths: tuple[float, float],
    dealiasing: str = "none",
) -> Array:
    r"""Return the 2D Poisson bracket ``[a,b] = a_x b_y - a_y b_x``."""
    da_dx, da_dy = gradient(a, lengths=lengths)
    db_dx, db_dy = gradient(b, lengths=lengths)
    return dealiased_product(
        da_dx,
        db_dy,
        dealiasing=dealiasing,
    ) - dealiased_product(
        da_dy,
        db_dx,
        dealiasing=dealiasing,
    )


def stream_function(omega: Array, *, lengths: tuple[float, float]) -> Array:
    r"""Solve ``∇² φ = ω`` with zero mean."""
    return inverse_laplacian(omega, lengths=lengths)


def current_density(psi: Array, *, lengths: tuple[float, float]) -> Array:
    r"""Return ``j_z = -∇² ψ``."""
    return -laplacian(psi, lengths=lengths)


def to_spectral_state(state: ReducedMHDState) -> ReducedMHDState:
    """Transform a physical reduced-MHD state to Fourier coefficients.

    Both fields are transformed as one batch. On a device mesh this gives JAX
    one larger collective instead of two small collectives.
    """
    fields = jnp.stack((state.psi, state.omega))
    transformed = jnp.fft.fftn(fields, axes=(-2, -1))
    return ReducedMHDState(psi=transformed[0], omega=transformed[1])


def to_physical_state(state_hat: ReducedMHDState) -> ReducedMHDState:
    """Transform Fourier coefficients to real physical fields."""
    fields_hat = jnp.stack((state_hat.psi, state_hat.omega))
    fields = jnp.fft.ifftn(fields_hat, axes=(-2, -1)).real
    return ReducedMHDState(psi=fields[0], omega=fields[1])


def to_physical_trajectory(trajectory: ReducedMHDTrajectory) -> ReducedMHDTrajectory:
    """Transform a saved Fourier-space trajectory to physical fields.

    The field and saved-time dimensions are batch dimensions, so the conversion
    needs one distributed inverse transform rather than one per saved state.
    """
    fields_hat = jnp.stack((trajectory.states.psi, trajectory.states.omega))
    fields = jnp.fft.ifftn(fields_hat, axes=(-2, -1)).real
    return ReducedMHDTrajectory(
        times=trajectory.times,
        states=ReducedMHDState(psi=fields[0], omega=fields[1]),
    )


def reduced_mhd_rhs_spectral(
    state_hat: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
    dealiasing: str = "none",
) -> ReducedMHDState:
    r"""Return the reduced-MHD right-hand side in Fourier space.

    The implementation reuses the Fourier coefficients of ``ψ`` and ``ω``.
    Eight physical derivatives are evaluated in one inverse-FFT batch, and the
    three Poisson brackets are returned to Fourier space in one forward-FFT
    batch. This is algebraically equivalent to :func:`reduced_mhd_rhs`, but it
    removes repeated transforms and reduces collective-launch overhead when the
    fields are sharded.
    """
    if state_hat.psi.ndim != 2 or state_hat.omega.shape != state_hat.psi.shape:
        raise ValueError("spectral reduced-MHD fields must have the same 2D shape")
    if len(lengths) != 2:
        raise ValueError(f"expected 2 lengths, got {len(lengths)}")
    if dealiasing not in ("none", "two_thirds"):
        raise ValueError("dealiasing must be 'none' or 'two_thirds'")

    shape = state_hat.psi.shape
    kx = spectral_wavenumbers(shape[0], lengths[0])[:, None]
    ky = spectral_wavenumbers(shape[1], lengths[1])[None, :]
    k_squared = kx**2 + ky**2
    safe_k_squared = jnp.where(k_squared == 0.0, 1.0, k_squared)

    if dealiasing == "two_thirds":
        nonlinear_mask = two_thirds_mask(shape)
        psi_nonlinear_hat = state_hat.psi * nonlinear_mask
        omega_nonlinear_hat = state_hat.omega * nonlinear_mask
    else:
        nonlinear_mask = jnp.ones(shape, dtype=bool)
        psi_nonlinear_hat = state_hat.psi
        omega_nonlinear_hat = state_hat.omega

    phi_hat = jnp.where(
        k_squared == 0.0,
        0.0,
        -omega_nonlinear_hat / safe_k_squared,
    )
    lap_psi_nonlinear_hat = -k_squared * psi_nonlinear_hat
    derivative_hats = jnp.stack(
        (
            1j * kx * phi_hat,
            1j * ky * phi_hat,
            1j * kx * psi_nonlinear_hat,
            1j * ky * psi_nonlinear_hat,
            1j * kx * omega_nonlinear_hat,
            1j * ky * omega_nonlinear_hat,
            1j * kx * lap_psi_nonlinear_hat,
            1j * ky * lap_psi_nonlinear_hat,
        )
    )
    derivatives = jnp.fft.ifftn(derivative_hats, axes=(-2, -1)).real
    phi_x, phi_y, psi_x, psi_y, omega_x, omega_y, lap_psi_x, lap_psi_y = derivatives

    brackets = jnp.stack(
        (
            phi_x * psi_y - phi_y * psi_x,
            phi_x * omega_y - phi_y * omega_x,
            psi_x * lap_psi_y - psi_y * lap_psi_x,
        )
    )
    bracket_hats = jnp.fft.fftn(brackets, axes=(-2, -1)) * nonlinear_mask
    dpsi_hat = (
        -bracket_hats[0]
        - params.resistivity * k_squared * state_hat.psi
    )
    domega_hat = (
        -bracket_hats[1]
        + bracket_hats[2]
        - params.viscosity * k_squared * state_hat.omega
    )
    base_rhs_hat = ReducedMHDState(psi=dpsi_hat, omega=domega_hat)

    if not terms:
        return base_rhs_hat

    state = to_physical_state(state_hat)
    base_rhs = to_physical_state(base_rhs_hat)
    complete_rhs = apply_physics_terms(
        base_rhs,
        terms,
        state,
        params,
        lengths=lengths,
    )
    return to_spectral_state(complete_rhs)


def reduced_mhd_rhs(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
    dealiasing: str = "none",
) -> ReducedMHDState:
    r"""Return the resistive-viscous reduced-MHD right-hand side.

    The convention is

    ``ψ_t + [φ, ψ] = η ∇²ψ``

    ``ω_t + [φ, ω] = [ψ, ∇²ψ] + ν ∇²ω``

    with ``∇²φ = ω`` on a periodic domain.
    """
    rhs_hat = reduced_mhd_rhs_spectral(
        to_spectral_state(state),
        params,
        lengths=lengths,
        terms=terms,
        dealiasing=dealiasing,
    )
    return to_physical_state(rhs_hat)


def linearized_reduced_mhd_rhs(
    state: ReducedMHDState,
    perturbation: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    terms: tuple[PhysicsTerm, ...] = (),
    dealiasing: str = "none",
) -> ReducedMHDState:
    """Return the matrix-free Jacobian-vector product of the reduced-MHD RHS.

    This computes ``dF(state)[perturbation]`` with JAX forward-mode automatic
    differentiation, where ``F`` is :func:`reduced_mhd_rhs`. It is the first
    building block for eigenvalue and adjoint tearing-mode benchmarks.
    """

    def rhs_for_jvp(active_state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(
            active_state,
            params,
            lengths=lengths,
            terms=terms,
            dealiasing=dealiasing,
        )

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
    dealiasing: str = "none",
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
    rhs_plus = reduced_mhd_rhs(
        plus,
        params,
        lengths=lengths,
        terms=terms,
        dealiasing=dealiasing,
    )
    rhs_minus = reduced_mhd_rhs(
        minus,
        params,
        lengths=lengths,
        terms=terms,
        dealiasing=dealiasing,
    )
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
    dealiasing: str = "none",
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
            dealiasing=dealiasing,
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
