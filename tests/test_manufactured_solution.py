from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from mhx.solver.core import compute_k_arrays_np, grad_from_hat


def test_grad_from_hat_matches_manufactured_solution():
    Nx = 8
    Ny = 8
    Nz = 1
    Lx = 2.0 * np.pi
    Ly = 2.0 * np.pi
    Lz = 2.0 * np.pi

    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    z = np.linspace(0.0, Lz, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    kx_mode = 2.0
    ky_mode = 3.0
    f = np.sin(kx_mode * X) * np.cos(ky_mode * Y)

    df_dx_true = kx_mode * np.cos(kx_mode * X) * np.cos(ky_mode * Y)
    df_dy_true = -ky_mode * np.sin(kx_mode * X) * np.sin(ky_mode * Y)
    df_dz_true = np.zeros_like(df_dx_true)

    f_hat = jnp.fft.fftn(jnp.asarray(f), axes=(0, 1, 2))
    kx, ky, kz, _, _, _, _ = compute_k_arrays_np(Nx, Ny, Nz, Lx, Ly, Lz)
    grad = grad_from_hat(f_hat, kx, ky, kz)

    grad_np = np.array(grad)
    err_dx = np.max(np.abs(grad_np[0] - df_dx_true))
    err_dy = np.max(np.abs(grad_np[1] - df_dy_true))
    err_dz = np.max(np.abs(grad_np[2] - df_dz_true))

    assert err_dx < 1e-6
    assert err_dy < 1e-6
    assert err_dz < 1e-6
