"""FFT-based spectral operators for periodic domains."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def spectral_wavenumbers(points: int, length: float) -> Array:
    """Return angular FFT wavenumbers for a periodic axis."""
    if points < 2:
        raise ValueError("points must be >= 2")
    if length <= 0.0:
        raise ValueError("length must be positive")
    spacing = length / points
    return 2.0 * jnp.pi * jnp.fft.fftfreq(points, d=spacing)


def _broadcast_wavenumbers(wavenumbers: Array, *, axis: int, ndim: int) -> Array:
    shape = [1] * ndim
    shape[axis] = wavenumbers.shape[0]
    return jnp.reshape(wavenumbers, shape)


def fft_derivative(field: Array, *, axis: int, length: float, order: int = 1) -> Array:
    """Differentiate a periodic field along one axis using complex FFTs."""
    array = jnp.asarray(field)
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return array
    axis = axis % array.ndim
    wavenumbers = spectral_wavenumbers(array.shape[axis], length)
    multiplier = (1j * _broadcast_wavenumbers(wavenumbers, axis=axis, ndim=array.ndim)) ** order
    transformed = jnp.fft.fft(array, axis=axis)
    derivative = jnp.fft.ifft(multiplier * transformed, axis=axis)
    if jnp.isrealobj(array):
        return jnp.real(derivative)
    return derivative


def gradient(field: Array, *, lengths: tuple[float, ...]) -> tuple[Array, ...]:
    """Return the spectral gradient of a periodic field."""
    array = jnp.asarray(field)
    if len(lengths) != array.ndim:
        raise ValueError(f"expected {array.ndim} lengths, got {len(lengths)}")
    return tuple(
        fft_derivative(array, axis=axis, length=length)
        for axis, length in enumerate(lengths)
    )


def laplacian(field: Array, *, lengths: tuple[float, ...]) -> Array:
    """Return the spectral Laplacian of a periodic field."""
    array = jnp.asarray(field)
    if len(lengths) != array.ndim:
        raise ValueError(f"expected {array.ndim} lengths, got {len(lengths)}")
    result = jnp.zeros_like(array)
    for axis, length in enumerate(lengths):
        result = result + fft_derivative(array, axis=axis, length=length, order=2)
    return result
