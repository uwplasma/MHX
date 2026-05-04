from __future__ import annotations

import jax.numpy as jnp

from mhx.config import MeshConfig
from mhx.grids import CartesianGrid


def test_cartesian_grid_from_config() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(8, 10), upper=(2.0, 5.0)))
    assert grid.ndim == 2
    assert grid.shape == (8, 10)
    assert grid.lengths == (2.0, 5.0)
    assert grid.spacing == (0.25, 0.5)
    assert grid.cell_volume == 0.125


def test_cartesian_grid_mesh_shapes() -> None:
    grid = CartesianGrid(shape=(8, 12), lower=(0.0, -1.0), upper=(1.0, 2.0))
    x, y = grid.mesh()
    assert x.shape == (8, 12)
    assert y.shape == (8, 12)
    assert jnp.isclose(x[0, 0], 1.0 / 16.0)
    assert jnp.isclose(y[0, 0], -1.0 + 3.0 / 24.0)


def test_cartesian_grid_sinusoid_shapes() -> None:
    grid = CartesianGrid(shape=(16, 16), lower=(0.0, 0.0), upper=(2.0, 2.0))
    assert grid.sinusoid(mode=(2, 1)).shape == grid.shape
    assert grid.cosinusoid(mode=(2, 1)).shape == grid.shape

