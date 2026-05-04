from __future__ import annotations

import jax.numpy as jnp

from mhx.solver.core import make_k_arrays, dissipation_rates


def test_dissipation_rates_positive():
    Nx, Ny, Nz = 8, 8, 1
    Lx = Ly = Lz = 2.0 * jnp.pi
    kx, ky, kz, k2, _, _, _ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)

    v_hat = jnp.ones((3, Nx, Ny, Nz)) * (1.0 + 1.0j)
    B_hat = jnp.ones((3, Nx, Ny, Nz)) * (0.5 + 0.5j)

    eps_visc, eps_ohm = dissipation_rates(v_hat, B_hat, k2, nu=1e-3, eta=2e-3, Lx=Lx, Ly=Ly, Lz=Lz)

    assert float(eps_visc) > 0.0
    assert float(eps_ohm) > 0.0
