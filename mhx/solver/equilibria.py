from __future__ import annotations

import jax.numpy as jnp

from mhx.solver.core import make_grid

Array = jnp.ndarray


def init_equilibrium(
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    *,
    B0: float = 1.0,
    a: float | None = None,
    B_g: float = 0.2,
    eps_B: float = 0.01,
    m_y: int = 1,
    m_z: int = 0,
):
    """
    Harris-sheet-like slab tearing equilibrium in a periodic box.

      B_y(x) = B0 * tanh((x - Lx/2)/a)
      B_z    = B_g
      B_x    = 0

    Perturbation via delta A_z = eps_B cos(k_y y) cos(k_z z):
      => delta B_x = -eps_B * k_y sin(k_y y) cos(k_z z)
    """
    if a is None:
        a = Lx / 16.0

    X, Y, Z = make_grid(Nx, Ny, Nz, Lx, Ly, Lz)

    sx = (X - 0.5 * Lx) / a
    By0 = B0 * jnp.tanh(sx)
    Bx0 = jnp.zeros_like(By0)
    Bz0 = B_g * jnp.ones_like(By0)

    k_y = 2.0 * jnp.pi * m_y / Ly
    k_z = 2.0 * jnp.pi * m_z / Lz

    phase_y = k_y * Y
    phase_z = k_z * Z

    delta_Bx = -eps_B * k_y * jnp.sin(phase_y) * jnp.cos(phase_z)
    delta_By = jnp.zeros_like(delta_Bx)
    delta_Bz = jnp.zeros_like(delta_Bx)

    Bx = Bx0 + delta_Bx
    By = By0 + delta_By
    Bz = Bz0 + delta_Bz

    B0_real = jnp.stack([Bx, By, Bz], axis=0)
    v0_real = jnp.zeros_like(B0_real)

    return v0_real, B0_real
