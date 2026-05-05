"""Small matrix-free linear-operator utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
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


def _normalize(vector: Array) -> Array:
    norm = jnp.linalg.norm(jnp.ravel(vector))
    if float(norm) == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm
