"""Equation-set namespace for MHX models."""

from mhx.equations.compressible_mhd import (
    compressible_mhd_fluxes,
    compressible_mhd_rhs,
    uniform_compressible_mhd_state,
)
from mhx.equations.reduced_mhd import (
    current_density,
    finite_difference_linearized_reduced_mhd_rhs,
    linearized_reduced_mhd_operator,
    linearized_reduced_mhd_rhs,
    poisson_bracket,
    reduced_mhd_residual_norm,
    reduced_mhd_rhs,
    stream_function,
)

__all__ = [
    "compressible_mhd_fluxes",
    "compressible_mhd_rhs",
    "current_density",
    "finite_difference_linearized_reduced_mhd_rhs",
    "linearized_reduced_mhd_operator",
    "linearized_reduced_mhd_rhs",
    "poisson_bracket",
    "reduced_mhd_residual_norm",
    "reduced_mhd_rhs",
    "stream_function",
    "uniform_compressible_mhd_state",
]
