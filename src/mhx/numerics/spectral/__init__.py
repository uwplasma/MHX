"""Spectral numerical operators."""

from mhx.numerics.spectral.operators import (
    fft_derivative,
    gradient,
    inverse_laplacian,
    laplacian,
    spectral_wavenumbers,
)

__all__ = [
    "fft_derivative",
    "gradient",
    "inverse_laplacian",
    "laplacian",
    "spectral_wavenumbers",
]
