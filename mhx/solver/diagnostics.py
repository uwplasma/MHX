from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


# --- Flux function ---

def compute_Az_from_hat(B_hat: Array, kx: Array, ky: Array) -> Array:
    """
    Compute A_z such that (B_x, B_y) = (-∂A_z/∂y, ∂A_z/∂x).
    Only uses kx, ky (perpendicular).
    """
    Bx_hat, By_hat = B_hat[0], B_hat[1]
    k_perp2 = kx**2 + ky**2
    k_perp2 = jnp.where(k_perp2 == 0.0, 1.0, k_perp2)

    Az_hat = 1j * (kx * By_hat - ky * Bx_hat) / k_perp2
    Az_hat = jnp.where(k_perp2 == 0.0, 0.0, Az_hat)

    Az = jnp.fft.ifftn(Az_hat, axes=(0, 1, 2)).real
    return Az


# --- Tearing amplitude + growth ---

def tearing_amplitude(B_hat: Array, Lx: float, Ly: float, Lz: float, band_width_frac: float = 0.25) -> Array:
    """
    RMS of Bx in a band around the current sheet (abs(x - Lx/2) < band_width_frac*Lx/2).
    Pure JAX version returning a JAX scalar.
    """
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    Bx = B[0]

    Nx = Bx.shape[0]
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    xc = 0.5 * Lx
    band_half = band_width_frac * 0.5 * Lx

    mask = (jnp.abs(x - xc)[:, None, None] < band_half)
    Bx_band = jnp.where(mask, Bx, 0.0)

    num = jnp.sum(Bx_band**2)
    den = jnp.sum(mask.astype(Bx.dtype)) + 1e-16
    rms = jnp.sqrt(num / den)
    return rms


def tearing_mode_amplitude_from_hat(B_hat_frames: Array, ix0: int, iy1: int, iz0: int) -> Array:
    """
    Amplitude of the primary tearing mode from Fourier-space Bx:

      |B_x(kx=0, ky=1, kz=0)|(t)

    B_hat_frames: (T, 3, Nx, Ny, Nz)
    """
    Bx_mode = B_hat_frames[:, 0, ix0, iy1, iz0]
    return jnp.abs(Bx_mode)


def estimate_growth_rate(
    ts: jnp.ndarray,
    mode_amp: jnp.ndarray,
    w0: jnp.ndarray,
    *,
    lower_factor: float = 5.0,
    upper_frac_of_max: float = 0.3,
    min_points: int = 8,
) -> Tuple[Array, Array, Array]:
    """
    JAX-friendly growth-rate estimate.

    - Picks an amplitude window:
          lower_factor * w0 <= |B_x| <= upper_frac_of_max * max(|B_x|)
      to isolate the exponential phase.

    - Performs masked linear regression of ln|B_x| vs t using weighted sums
      (no boolean indexing), so it works under jit/vmap.

    Returns:
        gamma_fit, lnA_fit_full(ts), mask_lin
    """
    ts = jnp.asarray(ts)
    A = jnp.asarray(mode_amp)
    eps = 1e-30

    # Build amplitude-based mask
    wmax = jnp.max(A)
    lower = lower_factor * w0
    upper = upper_frac_of_max * wmax
    mask = (A >= lower) & (A <= upper)         # bool[T]
    w = mask.astype(ts.dtype)                  # 0/1 weights, same shape as ts

    def _fit_weighted(args):
        ts_local, A_local, w_local = args

        lnA = jnp.log(jnp.maximum(A_local, eps))

        S_w  = jnp.sum(w_local) + 1e-30
        S_t  = jnp.sum(w_local * ts_local)
        S_y  = jnp.sum(w_local * lnA)
        S_tt = jnp.sum(w_local * ts_local * ts_local)
        S_ty = jnp.sum(w_local * ts_local * lnA)

        denom = (S_tt - S_t * S_t / S_w) + 1e-30
        gamma = (S_ty - S_t * S_y / S_w) / denom
        c = (S_y - gamma * S_t) / S_w

        lnA_fit_full = gamma * ts_local + c
        return gamma, lnA_fit_full, mask

    def _fit_all(args):
        ts_local, A_local, _ = args
        lnA = jnp.log(jnp.maximum(A_local, eps))
        w_local = jnp.ones_like(ts_local, dtype=ts_local.dtype)

        S_w  = jnp.sum(w_local)
        S_t  = jnp.sum(w_local * ts_local)
        S_y  = jnp.sum(w_local * lnA)
        S_tt = jnp.sum(w_local * ts_local * ts_local)
        S_ty = jnp.sum(w_local * ts_local * lnA)

        denom = (S_tt - S_t * S_t / S_w) + 1e-30
        gamma = (S_ty - S_t * S_y / S_w) / denom
        c = (S_y - gamma * S_t) / S_w

        lnA_fit_full = gamma * ts_local + c
        mask_all = jnp.ones_like(ts_local, dtype=bool)
        return gamma, lnA_fit_full, mask_all

    # How many masked points?
    n_points = jnp.sum(w).astype(jnp.int32)

    gamma, lnA_fit_full, mask_lin = lax.cond(
        n_points >= min_points,
        _fit_weighted,
        _fit_all,
        operand=(ts, A, w),
    )

    return gamma, lnA_fit_full, mask_lin


# --- Reconnection / plasmoid diagnostics ---

def reconnection_rate_from_Az(ts: Array, Az_xpt: Array) -> Array:
    """
    Reconnection-rate proxy from the time derivative of the flux function A_z
    at the X-point:

      E_rec(t) ~ - d/dt A_z(x_X, y_X, z_X)

    Here x_X, y_X, z_X are fixed grid points near the X-point.
    """
    dAz_dt = jnp.gradient(Az_xpt, ts)
    E_rec = -dAz_dt
    return E_rec


def count_local_extrema_1d(f: Array) -> Array:
    """
    Count local extrema (minima + maxima) in a 1D array f(j).

    This is a crude plasmoid / island-count proxy.
    """
    df = jnp.diff(f)
    sign = jnp.sign(df)
    # A sign change in df indicates a local extremum
    sign_prod = sign[:-1] * sign[1:]
    extrema_mask = sign_prod < 0.0
    return jnp.sum(extrema_mask.astype(jnp.int32))


def plasmoid_complexity_metric(A_mid: Array) -> Array:
    """
    Smooth proxy for plasmoid / island complexity on a 1D cut of A_z.

    Instead of a discrete island count (which is non-differentiable), we use
    the mean-squared curvature of A(z or y):

        C = ⟨ (∂²A/∂s²)^2 ⟩_s

    where s is the coordinate along the cut (e.g., y on the midplane).

    This quantity is:
      * JAX- and autodiff-friendly,
      * larger when the profile contains more fine-scale structure / plasmoids.
    """
    A_mid = jnp.asarray(A_mid)
    s = jnp.arange(A_mid.shape[0], dtype=A_mid.dtype)

    # First and second derivatives via finite differences
    dA_ds = jnp.gradient(A_mid, s)
    d2A_ds2 = jnp.gradient(dA_ds, s)

    return jnp.mean(d2A_ds2**2)
