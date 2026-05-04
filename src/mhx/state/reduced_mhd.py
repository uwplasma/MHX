"""PyTree-compatible reduced-MHD state containers."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Array


class ReducedMHDState(NamedTuple):
    """2D reduced-MHD state using magnetic flux ``psi`` and vorticity ``omega``."""

    psi: Array
    omega: Array


class ReducedMHDParams(NamedTuple):
    """Diffusivities for resistive-viscous reduced MHD."""

    resistivity: float
    viscosity: float


class ReducedMHDTrajectory(NamedTuple):
    """Saved reduced-MHD trajectory samples."""

    times: Array
    states: ReducedMHDState

