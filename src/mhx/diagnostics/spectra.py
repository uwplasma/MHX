"""Isotropic magnetic spectra and power-law fits for periodic 2-D fields."""

from __future__ import annotations

import numpy as np


def magnetic_power_spectrum_2d(
    psi_field: np.ndarray, domain_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1-D isotropic magnetic power spectrum of a flux function.

    The spectrum is normalized to the continuous field on a square periodic box
    of side ``L = domain_size``: summing ``power`` over the shells and
    multiplying by the shell width ``2 pi / L`` gives the box-integrated
    gradient energy ``integral |grad psi|^2 dA = L^2 <|grad psi|^2>`` of the
    resolved shells. Dividing by ``2 L^2`` gives the magnetic energy density
    ``0.5 <|grad psi|^2>``.

    Args:
        psi_field: Square ``(n, n)`` array of the magnetic flux function.
        domain_size: Side length of the periodic box.

    Returns:
        Integer shell wavenumbers ``k = 1 ... n/2 - 1`` and the power in each
        shell.
    """
    psi_hat = np.fft.fftn(psi_field)
    n = psi_field.shape[0]

    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    k_squared = kx_grid**2 + ky_grid**2
    physical_derivative_factor = (2.0 * np.pi / domain_size) ** 2
    power_2d = k_squared * physical_derivative_factor * np.abs(psi_hat) ** 2

    dx_dy = (domain_size / n) ** 2
    power_2d *= dx_dy / (n**2)

    k_radius = np.round(np.sqrt(k_squared)).astype(int)
    k_max = n // 2
    k_1d = np.arange(1, k_max)
    power_1d = np.zeros(k_max - 1)

    dk = 2.0 * np.pi / domain_size
    for shell in k_1d:
        power_1d[shell - 1] = np.sum(power_2d[k_radius == shell]) / dk

    return k_1d, power_1d


def fit_power_law(
    k: np.ndarray, power: np.ndarray, *, kmin: float, kmax: float
) -> tuple[float, float]:
    """Fit ``power ~ 10**intercept * k**slope`` over ``kmin <= k <= kmax``.

    Args:
        k: Wavenumbers.
        power: Spectrum values at ``k``.
        kmin: Lower edge of the fit band.
        kmax: Upper edge of the fit band.

    Returns:
        The log-log slope and the base-10 intercept.

    Raises:
        ValueError: If fewer than two positive samples lie in the band.
    """
    k = np.asarray(k, dtype=float)
    power = np.asarray(power, dtype=float)
    band = (k >= kmin) & (k <= kmax) & (power > 0.0)
    if np.count_nonzero(band) < 2:
        raise ValueError("fit_power_law needs at least two positive samples in [kmin, kmax]")
    slope, intercept = np.polyfit(np.log10(k[band]), np.log10(power[band]), 1)
    return float(slope), float(intercept)
