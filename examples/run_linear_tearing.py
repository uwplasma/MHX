"""Minimal Python driver for the first rebuilt MHX config."""

from __future__ import annotations

from mhx.config import load_config
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import laplacian


def main() -> None:
    cfg = load_config("examples/linear_tearing.toml")
    grid = CartesianGrid.from_mesh_config(cfg.mesh)
    field = grid.sinusoid(mode=(1, 1))
    lap = laplacian(field, lengths=grid.lengths)
    print({"shape": field.shape, "laplacian_norm": float(abs(lap).max())})


if __name__ == "__main__":
    main()

