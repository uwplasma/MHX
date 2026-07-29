"""Optional SOLVAX-backed algebraic solvers and MHX adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array

from mhx.numerics.linear_operator import MatrixFreeOperator
from mhx.numerics.spectral import spectral_wavenumbers
from mhx.state import ReducedMHDParams, ReducedMHDState

StateT = TypeVar("StateT")


class LinearSolveResult(NamedTuple):
    """Backend-independent linear solution and convergence diagnostics."""

    x: Any
    residual_norm: Array
    iterations: Array
    converged: Array


class NonlinearSolveResult(NamedTuple):
    """Backend-independent nonlinear solution and convergence diagnostics."""

    x: Any
    residual_norm: Array
    nonlinear_iterations: Array
    linear_iterations: Array
    converged: Array
    linear_converged: Array


def _solvax() -> Any:
    try:
        import solvax
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise ImportError(
            "SOLVAX-backed methods require the optional solver dependencies; "
            "install MHX with the 'solvers' extra"
        ) from error
    return solvax


def as_solvax_operator(operator: MatrixFreeOperator) -> Any:
    """Adapt an MHX flat matrix-free operator to SOLVAX's algebraic shape."""
    size = 1
    for extent in operator.shape:
        size *= extent
    if operator.shape != (size,):
        raise ValueError(
            "SOLVAX's algebraic operator adapter requires a flat MHX vector shape, "
            f"got {operator.shape}"
        )
    return _solvax().MatrixFreeOperator(operator, shape=(size, size))


def complex_linear_extension(
    operator: Callable[[Array], Array],
) -> Callable[[Array], Array]:
    """Extend a real-linear JAX operator to complex trial vectors."""

    def apply(vector: Array) -> Array:
        return operator(jnp.real(vector)) + 1j * operator(jnp.imag(vector))

    return apply


def gmres_solve(
    matvec: Callable[[StateT], StateT],
    rhs: StateT,
    *,
    x0: StateT | None = None,
    preconditioner: Callable[[StateT], StateT] | None = None,
    restart: int = 30,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    max_restarts: int = 10,
) -> LinearSolveResult:
    """Solve a matrix-free system with SOLVAX FGMRES."""
    solution = _solvax().gmres(
        matvec,
        rhs,
        x0=x0,
        precond=preconditioner,
        restart=restart,
        rtol=rtol,
        atol=atol,
        max_restarts=max_restarts,
    )
    return LinearSolveResult(
        x=solution.x,
        residual_norm=solution.residual_norm,
        iterations=solution.iterations,
        converged=solution.converged,
    )


def newton_krylov_solve(
    residual: Callable[[StateT], StateT],
    x0: StateT,
    *,
    preconditioner: Callable[[StateT], StateT] | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    max_steps: int = 20,
    linear_restart: int = 30,
    linear_rtol: float = 0.1,
    linear_atol: float = 0.0,
    linear_max_restarts: int = 10,
) -> NonlinearSolveResult:
    """Solve a PyTree residual with SOLVAX Jacobian-free Newton–Krylov."""
    solution = _solvax().newton_krylov(
        residual,
        x0,
        precond=preconditioner,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        linear_restart=linear_restart,
        linear_rtol=linear_rtol,
        linear_atol=linear_atol,
        linear_max_restarts=linear_max_restarts,
    )
    return NonlinearSolveResult(
        x=solution.x,
        residual_norm=solution.residual_norm,
        nonlinear_iterations=solution.newton_iterations,
        linear_iterations=solution.linear_iterations,
        converged=solution.converged,
        linear_converged=solution.linear_converged,
    )


def spectral_diffusion_preconditioner(
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
    dt: float,
) -> Callable[[ReducedMHDState], ReducedMHDState]:
    """Return the exact inverse of the backward-Euler diffusion principal part."""
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
