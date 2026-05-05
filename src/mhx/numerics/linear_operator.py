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
