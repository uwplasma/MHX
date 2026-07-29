"""Numerical methods."""

from mhx.numerics.linear_operator import (
    ArnoldiResult,
    MatrixFreeOperator,
    PowerIterationResult,
    arnoldi_iteration,
    eigen_residual_norm,
    power_iteration,
    rayleigh_quotient,
    to_scipy_linear_operator,
)
from mhx.numerics.solvers import (
    LinearSolveResult,
    NonlinearSolveResult,
    as_solvax_operator,
    complex_linear_extension,
    gmres_solve,
    newton_krylov_solve,
    spectral_diffusion_preconditioner,
)

__all__ = [
    "ArnoldiResult",
    "LinearSolveResult",
    "MatrixFreeOperator",
    "NonlinearSolveResult",
    "PowerIterationResult",
    "arnoldi_iteration",
    "as_solvax_operator",
    "complex_linear_extension",
    "eigen_residual_norm",
    "gmres_solve",
    "newton_krylov_solve",
    "power_iteration",
    "rayleigh_quotient",
    "spectral_diffusion_preconditioner",
    "to_scipy_linear_operator",
]
