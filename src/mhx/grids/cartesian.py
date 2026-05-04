"""Periodic Cartesian grid utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from mhx.config import MeshConfig


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform Cartesian grid with cell-centered coordinates."""

    shape: tuple[int, int]
    lower: tuple[float, float]
    upper: tuple[float, float]
    periodic: tuple[bool, bool] = (True, True)

    def __post_init__(self) -> None:
        MeshConfig(
            shape=self.shape,
            lower=self.lower,
            upper=self.upper,
            periodic=self.periodic,
        ).validated()

    @classmethod
    def from_mesh_config(cls, config: MeshConfig) -> CartesianGrid:
        """Build a grid from a mesh config."""
        return cls(
            shape=config.shape,
            lower=config.lower,
            upper=config.upper,
            periodic=config.periodic,
        )

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return 2

    @property
    def lengths(self) -> tuple[float, float]:
        """Domain lengths."""
        return tuple(hi - lo for lo, hi in zip(self.lower, self.upper, strict=True))

    @property
    def spacing(self) -> tuple[float, float]:
        """Cell spacings."""
        return tuple(
            length / points
            for length, points in zip(self.lengths, self.shape, strict=True)
        )

    @property
    def cell_volume(self) -> float:
        """Cell area for the 2D grid."""
        dx, dy = self.spacing
        return dx * dy

    def axes(self) -> tuple[Array, Array]:
        """Return cell-centered coordinate axes."""
        return tuple(
            lo + (jnp.arange(points) + 0.5) * spacing
            for lo, points, spacing in zip(self.lower, self.shape, self.spacing, strict=True)
        )

    def mesh(self) -> tuple[Array, Array]:
        """Return cell-centered mesh arrays with matrix indexing."""
        return jnp.meshgrid(*self.axes(), indexing="ij")

    def sinusoid(self, mode: tuple[int, int] = (1, 1)) -> Array:
        """Return ``sin(kx x + ky y)`` on the grid."""
        x, y = self.mesh()
        kx = 2.0 * jnp.pi * mode[0] / self.lengths[0]
        ky = 2.0 * jnp.pi * mode[1] / self.lengths[1]
        return jnp.sin(kx * x + ky * y)

    def cosinusoid(self, mode: tuple[int, int] = (1, 1)) -> Array:
        """Return ``cos(kx x + ky y)`` on the grid."""
        x, y = self.mesh()
        kx = 2.0 * jnp.pi * mode[0] / self.lengths[0]
        ky = 2.0 * jnp.pi * mode[1] / self.lengths[1]
        return jnp.cos(kx * x + ky * y)
