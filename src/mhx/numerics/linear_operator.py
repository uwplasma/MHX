"""Small matrix-free linear-operator utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


@dataclass(frozen=True)
class MatrixFreeOperator:
    """Callable matrix-free operator with explicit input shape metadata."""

    shape: tuple[int, ...]
    matvec: Callable[[Array], Array]
    name: str = "matrix_free_operator"

    def __call__(self, vector: Array) -> Array:
        """Apply the operator to ``vector`` after checking the configured shape."""
        if tuple(vector.shape) != self.shape:
            raise ValueError(f"{self.name} expected shape {self.shape}, got {tuple(vector.shape)}")
        result = self.matvec(vector)
        if tuple(result.shape) != self.shape:
            raise ValueError(
                f"{self.name} returned shape {tuple(result.shape)}, expected {self.shape}"
            )
        return result


@dataclass(frozen=True)
class PowerIterationResult:
    """Dominant-eigenpair estimate and convergence history."""

    eigenvalue: Array
    eigenvector: Array
    residual_norm: Array
    rayleigh_history: Array
    residual_history: Array


@dataclass(frozen=True)
class ArnoldiResult:
    """Arnoldi Ritz spectrum and residual estimates for one Krylov run."""

    ritz_values: Array
    residual_estimates: Array
    hessenberg: Array
    basis: Array


def rayleigh_quotient(operator: MatrixFreeOperator, vector: Array) -> Array:
    """Return the real Rayleigh quotient ``<v,Lv>/<v,v>``."""
    denominator = jnp.vdot(vector, vector)
    return jnp.real(jnp.vdot(vector, operator(vector)) / denominator)


def eigen_residual_norm(
    operator: MatrixFreeOperator,
    vector: Array,
    eigenvalue: float | Array,
) -> Array:
    """Return relative residual ``||Lv-λv||₂ / ||v||₂``."""
    residual = operator(vector) - eigenvalue * vector
    return jnp.linalg.norm(jnp.ravel(residual)) / jnp.linalg.norm(jnp.ravel(vector))


def power_iteration(
    operator: MatrixFreeOperator,
    initial_vector: Array,
    *,
    iterations: int = 25,
) -> PowerIterationResult:
    """Estimate the dominant-magnitude eigenpair by normalized power iteration."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    vector = _normalize(initial_vector)
    rayleigh_values = []
    residual_values = []
    for _ in range(iterations):
        next_vector = _normalize(operator(vector))
        eigenvalue = rayleigh_quotient(operator, next_vector)
        residual = eigen_residual_norm(operator, next_vector, eigenvalue)
        rayleigh_values.append(eigenvalue)
        residual_values.append(residual)
        vector = next_vector
    return PowerIterationResult(
        eigenvalue=rayleigh_values[-1],
        eigenvector=vector,
        residual_norm=residual_values[-1],
        rayleigh_history=jnp.asarray(rayleigh_values),
        residual_history=jnp.asarray(residual_values),
    )


def arnoldi_iteration(
    operator: MatrixFreeOperator,
    initial_vector: Array,
    *,
    krylov_dim: int = 8,
    breakdown_tolerance: float = 1.0e-12,
) -> ArnoldiResult:
    """Return Ritz values from modified-Gram-Schmidt Arnoldi iteration."""
    if krylov_dim < 1:
        raise ValueError("krylov_dim must be >= 1")
    if tuple(initial_vector.shape) != operator.shape:
        raise ValueError(
            f"{operator.name} expected initial shape {operator.shape}, got "
            f"{tuple(initial_vector.shape)}"
        )
    vector_size = int(np.prod(operator.shape))
    steps = min(krylov_dim, vector_size)
    basis_vectors = [_normalize(jnp.ravel(initial_vector))]
    hessenberg = jnp.zeros((steps + 1, steps), dtype=initial_vector.dtype)
    actual_steps = steps
    for column in range(steps):
        vector = jnp.ravel(operator(jnp.reshape(basis_vectors[column], operator.shape)))
        for row in range(column + 1):
            projection = jnp.vdot(basis_vectors[row], vector)
            hessenberg = hessenberg.at[row, column].set(projection)
            vector = vector - projection * basis_vectors[row]
        next_norm = jnp.linalg.norm(vector)
        hessenberg = hessenberg.at[column + 1, column].set(next_norm)
        if float(next_norm) <= breakdown_tolerance:
            actual_steps = column + 1
            break
        if column + 1 < steps:
            basis_vectors.append(vector / next_norm)

    square_hessenberg = hessenberg[:actual_steps, :actual_steps]
    ritz_values, ritz_vectors = jnp.linalg.eig(square_hessenberg)
    residual_factor = hessenberg[actual_steps, actual_steps - 1]
    residual_estimates = jnp.abs(residual_factor * ritz_vectors[-1, :])
    basis = jnp.stack(
        [jnp.reshape(vector, operator.shape) for vector in basis_vectors[:actual_steps]]
    )
    return ArnoldiResult(
        ritz_values=ritz_values,
        residual_estimates=residual_estimates,
        hessenberg=square_hessenberg,
        basis=basis,
    )


def to_scipy_linear_operator(
    operator: MatrixFreeOperator,
    *,
    dtype: Any = np.float64,
) -> Any:
    """Return a lazy ``scipy.sparse.linalg.LinearOperator`` wrapper.

    SciPy is imported lazily so the core MHX package can expose matrix-free
    operators without requiring SciPy-only workflows at import time.
    """
    from scipy.sparse.linalg import LinearOperator

    vector_size = int(np.prod(operator.shape))

    def matvec(vector: np.ndarray) -> np.ndarray:
        values = jnp.asarray(vector).reshape(operator.shape)
        result = operator(values)
        return np.asarray(result).reshape(vector_size)

    return LinearOperator(
        shape=(vector_size, vector_size),
        matvec=matvec,
        dtype=np.dtype(dtype),
    )


def _normalize(vector: Array) -> Array:
    norm = jnp.linalg.norm(jnp.ravel(vector))
    if float(norm) == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm
