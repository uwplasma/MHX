"""Numerical methods."""

from mhx.numerics.linear_operator import (
    MatrixFreeOperator,
    eigen_residual_norm,
    rayleigh_quotient,
)

__all__ = ["MatrixFreeOperator", "eigen_residual_norm", "rayleigh_quotient"]
