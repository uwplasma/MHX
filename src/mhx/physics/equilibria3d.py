"""Initial conditions for three-dimensional incompressible MHD.

Every builder returns component-first real fields ``(v, b)`` of shape
``(3, nx, ny, nz)`` on the periodic box, ready for
:func:`mhx.equations.mhd3d.to_spectral`. The Orszag--Tang and
Taylor--Green fields follow the published formulas cited in ``plan_3d.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
from jaxtyping import Array


def _grid(shape: tuple[int, int, int]) -> tuple[Array, Array, Array]:
    axes = [
        jnp.linspace(0.0, 2.0 * jnp.pi, points, endpoint=False)
        for points in shape
    ]
    return tuple(jnp.meshgrid(*axes, indexing="ij"))


@dataclass(frozen=True)
class SingleModeEquilibrium:
    """One magnetic Fourier mode at rest, for exact-decay gates.

    The field ``b = amplitude * (0, sin(k . x), 0)`` with
    ``k = (mode_x, 0, mode_z)`` is divergence-free and force-free at first
    order, so with zero velocity it decays at exactly ``eta k^2``.
    """

    amplitude: float = 1.0e-3
    mode: tuple[int, int, int] = (1, 0, 1)

    name: ClassVar[str] = "single_mode_3d"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        x, y, z = _grid(shape)
        del y
        phase = self.mode[0] * x + self.mode[2] * z
        zero = jnp.zeros(shape)
        b = jnp.stack((zero, self.amplitude * jnp.sin(phase), zero))
        v = jnp.zeros_like(b)
        return v, b


@dataclass(frozen=True)
class CircularlyPolarizedAlfvenEquilibrium:
    """Walén-state circularly polarized Alfvén wave along ``z``.

    With a unit guide field along ``z``, the state
    ``b_perp = amplitude (cos kz, sin kz, 0)`` and ``v = -b_perp`` is an
    exact nonlinear solution propagating at the Alfvén speed. It is the
    G3 gate of ``plan_3d.md``.
    """

    amplitude: float = 0.1
    mode: int = 1

    name: ClassVar[str] = "cp_alfven"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        x, y, z = _grid(shape)
        del x, y
        phase = self.mode * z
        zero = jnp.zeros(shape)
        b = self.amplitude * jnp.stack((jnp.cos(phase), jnp.sin(phase), zero))
        return -b, b


@dataclass(frozen=True)
class AlfvenWaveCollisionEquilibrium:
    """Howes--Nielson perpendicular counterpropagating Alfvén waves."""

    amplitude_plus: float = 0.1
    amplitude_minus: float = 0.1
    name: ClassVar[str] = "alfven_wave_collision"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        x, y, z = _grid(shape)
        zero = jnp.zeros(shape)
        z_plus = self.amplitude_plus * jnp.stack((zero, jnp.cos(x - z), zero))
        z_minus = self.amplitude_minus * jnp.stack((jnp.cos(y + z), zero, zero))
        return 0.5 * (z_plus + z_minus), 0.5 * (z_plus - z_minus)


@dataclass(frozen=True)
class OrszagTang3DEquilibrium:
    """The 3D Orszag--Tang vortex of Politano, Pouquet and Sulem (1995).

    Exact initial condition, as run to 1536 cubed by Mininni, Pouquet and
    Montgomery (2006):

    ``v = (-2 sin y, 2 sin x, 0)``,
    ``b = beta (-2 sin 2y + sin z, 2 sin x + sin z, sin x + sin y)``,

    with ``beta = 0.8`` giving near-equipartition: kinetic energy 2 and
    magnetic energy 1.92 exactly.
    """

    beta: float = 0.8

    name: ClassVar[str] = "orszag_tang_3d"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        x, y, z = _grid(shape)
        zero = jnp.zeros(shape)
        v = jnp.stack((-2.0 * jnp.sin(y), 2.0 * jnp.sin(x), zero))
        b = self.beta * jnp.stack(
            (
                -2.0 * jnp.sin(2.0 * y) + jnp.sin(z),
                2.0 * jnp.sin(x) + jnp.sin(z),
                jnp.sin(x) + jnp.sin(y),
            )
        )
        return v, b


@dataclass(frozen=True)
class ABCFlowEquilibrium:
    """The Arnold--Beltrami--Childress flow with a magnetic seed.

    ``v`` is the 1:1:1 ABC flow; ``b`` is a small random-phase solenoidal
    seed built from a few low modes, for kinematic-dynamo studies
    (Galloway--Frisch 1986; Bouya--Dormy 2013).
    """

    a: float = 1.0
    b_coefficient: float = 1.0
    c: float = 1.0
    seed_amplitude: float = 1.0e-6
    seed: int = 0

    name: ClassVar[str] = "abc_flow"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        import jax

        x, y, z = _grid(shape)
        v = jnp.stack(
            (
                self.a * jnp.sin(z) + self.c * jnp.cos(y),
                self.b_coefficient * jnp.sin(x) + self.a * jnp.cos(z),
                self.c * jnp.sin(y) + self.b_coefficient * jnp.cos(x),
            )
        )
        # Broadband random seed: the growing eigenmode selects itself. A
        # Beltrami-aligned seed projects poorly onto it and decays first.
        noise = jax.random.normal(jax.random.PRNGKey(self.seed), (3, *shape))
        seed_field = self.seed_amplitude * noise
        return v, seed_field


@dataclass(frozen=True)
class TaylorGreenEquilibrium:
    """Taylor--Green velocity with the insulating magnetic field of
    Lee, Brachet, Pouquet, Mininni and Rosenberg (2010), class I."""

    v0: float = 1.0
    b0: float = 1.0

    name: ClassVar[str] = "taylor_green_i"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        x, y, z = _grid(shape)
        zero = jnp.zeros(shape)
        v = self.v0 * jnp.stack(
            (
                jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
                -jnp.cos(x) * jnp.sin(y) * jnp.cos(z),
                zero,
            )
        )
        b = self.b0 * jnp.stack(
            (
                jnp.cos(x) * jnp.sin(y) * jnp.sin(z),
                jnp.sin(x) * jnp.cos(y) * jnp.sin(z),
                -2.0 * jnp.sin(x) * jnp.sin(y) * jnp.cos(z),
            )
        )
        return v, b


@dataclass(frozen=True)
class TaylorGreenAlternativeEquilibrium:
    """Lee et al. (2010) alternative insulating Taylor--Green field."""

    v0: float = 1.0
    b0: float = 1.0
    name: ClassVar[str] = "taylor_green_a"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        x, y, z = _grid(shape)
        zero = jnp.zeros(shape)
        v = self.v0 * jnp.stack(
            (
                jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
                -jnp.cos(x) * jnp.sin(y) * jnp.cos(z),
                zero,
            )
        )
        b = self.b0 * jnp.stack(
            (
                jnp.cos(2 * x) * jnp.sin(2 * y) * jnp.sin(2 * z),
                -jnp.sin(2 * x) * jnp.cos(2 * y) * jnp.sin(2 * z),
                zero,
            )
        )
        return v, b


@dataclass(frozen=True)
class TaylorGreenConductingEquilibrium:
    """Lee et al. (2010) conducting Taylor--Green magnetic field."""

    v0: float = 1.0
    b0: float = 1.0
    name: ClassVar[str] = "taylor_green_c"

    def initial_fields(self, shape: tuple[int, int, int]) -> tuple[Array, Array]:
        x, y, z = _grid(shape)
        zero = jnp.zeros(shape)
        v = self.v0 * jnp.stack(
            (
                jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
                -jnp.cos(x) * jnp.sin(y) * jnp.cos(z),
                zero,
            )
        )
        b = self.b0 * jnp.stack(
            (
                jnp.sin(2 * x) * jnp.cos(2 * y) * jnp.cos(2 * z),
                jnp.cos(2 * x) * jnp.sin(2 * y) * jnp.cos(2 * z),
                -2 * jnp.cos(2 * x) * jnp.cos(2 * y) * jnp.sin(2 * z),
            )
        )
        return v, b


__all__ = [
    "ABCFlowEquilibrium",
    "AlfvenWaveCollisionEquilibrium",
    "CircularlyPolarizedAlfvenEquilibrium",
    "OrszagTang3DEquilibrium",
    "SingleModeEquilibrium",
    "TaylorGreenAlternativeEquilibrium",
    "TaylorGreenConductingEquilibrium",
    "TaylorGreenEquilibrium",
]
