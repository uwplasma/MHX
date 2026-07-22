"""Simulation state containers."""

from mhx.state.compressible_mhd import (
    CompressibleMHDParams,
    CompressibleMHDPrimitive,
    CompressibleMHDState,
    CompressibleMHDTrajectory,
    conservative_from_primitive,
    primitive_from_conservative,
)
from mhx.state.reduced_mhd import (
    ReducedMHDParams,
    ReducedMHDState,
    ReducedMHDTrajectory,
    flatten_reduced_mhd_state,
    reduced_mhd_state_size,
    unflatten_reduced_mhd_state,
)

__all__ = [
    "CompressibleMHDParams",
    "CompressibleMHDPrimitive",
    "CompressibleMHDState",
    "CompressibleMHDTrajectory",
    "ReducedMHDParams",
    "ReducedMHDState",
    "ReducedMHDTrajectory",
    "conservative_from_primitive",
    "flatten_reduced_mhd_state",
    "primitive_from_conservative",
    "reduced_mhd_state_size",
    "unflatten_reduced_mhd_state",
]
