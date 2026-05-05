"""Numerical methods."""

from mhx.numerics.linear_operator import (
    MatrixFreeOperator,
    PowerIterationResult,
    eigen_residual_norm,
    power_iteration,
    rayleigh_quotient,
)

__all__ = [
    "MatrixFreeOperator",
    "PowerIterationResult",
    "eigen_residual_norm",
    "power_iteration",
    "rayleigh_quotient",
]
