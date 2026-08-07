"""State containers for three-dimensional incompressible MHD."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Array


class MHD3DState(NamedTuple):
    """Velocity and magnetic field in the half-spectrum representation.

    Both leaves are complex arrays of shape ``(3, nx, ny, nz // 2 + 1)``:
    component first, then the three periodic directions with the real FFT
    convention on the last axis.
    """

    v_hat: Array
    b_hat: Array


class MHD3DParams(NamedTuple):
    """Differentiable physical parameters of the 3D model.

    Args:
        viscosity: Kinematic viscosity, the inverse Reynolds number.
        resistivity: Magnetic diffusivity, the inverse Lundquist number.
        guide_field: Uniform background field added inside the nonlinear
            terms. The default is no guide field.
    """

    viscosity: Array | float
    resistivity: Array | float
    guide_field: tuple[float, float, float] = (0.0, 0.0, 0.0)


class MHD3DTrajectory(NamedTuple):
    """Saved times and states from a 3D evolution."""

    times: Array
    states: MHD3DState
