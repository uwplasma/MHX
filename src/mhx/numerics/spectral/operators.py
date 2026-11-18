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


def two_thirds_mask(shape: tuple[int, ...]) -> Array:
    """Return the tensor-product 2/3-rule mask for an FFT grid."""
    if not shape or any(points < 2 for points in shape):
        raise ValueError("shape entries must be >= 2")
    mask = jnp.ones(shape, dtype=bool)
    for axis, points in enumerate(shape):
        integer_modes = jnp.fft.fftfreq(points) * points
        axis_mask = jnp.abs(integer_modes) < points / 3.0
        mask = mask & _broadcast_wavenumbers(axis_mask, axis=axis, ndim=len(shape))
    return mask


def spectral_filter(field: Array, *, dealiasing: str = "two_thirds") -> Array:
    """Filter a periodic field with the selected pseudo-spectral rule."""
    array = jnp.asarray(field)
    if dealiasing == "none":
        return array
    if dealiasing != "two_thirds":
        raise ValueError("dealiasing must be 'none' or 'two_thirds'")
    filtered = jnp.fft.ifftn(jnp.fft.fftn(array) * two_thirds_mask(array.shape))
    if jnp.isrealobj(array):
        return jnp.real(filtered)
    return filtered


def dealiased_product(left: Array, right: Array, *, dealiasing: str = "two_thirds") -> Array:
    """Multiply periodic fields with input and output 2/3-rule filtering."""
    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"dealiased product requires equal shapes, got {left_array.shape} and "
            f"{right_array.shape}"
        )
    if dealiasing == "none":
        return left_array * right_array
    filtered_left = spectral_filter(left_array, dealiasing=dealiasing)
    filtered_right = spectral_filter(right_array, dealiasing=dealiasing)
    return spectral_filter(filtered_left * filtered_right, dealiasing=dealiasing)


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


def inverse_laplacian(field: Array, *, lengths: tuple[float, ...]) -> Array:
    """Invert the periodic Laplacian with the zero Fourier mode set to zero."""
    array = jnp.asarray(field)
    if len(lengths) != array.ndim:
        raise ValueError(f"expected {array.ndim} lengths, got {len(lengths)}")

    denominator = jnp.zeros(array.shape)
    for axis, length in enumerate(lengths):
        wavenumbers = spectral_wavenumbers(array.shape[axis], length)
        denominator = denominator + _broadcast_wavenumbers(
            wavenumbers**2,
            axis=axis,
            ndim=array.ndim,
        )

    transformed = jnp.fft.fftn(array)
    zero_mode = denominator == 0.0
    safe_denominator = jnp.where(zero_mode, 1.0, denominator)
    inverse_hat = jnp.where(zero_mode, 0.0, -transformed / safe_denominator)
    inverse = jnp.fft.ifftn(inverse_hat)
    if jnp.isrealobj(array):
        return jnp.real(inverse)
    return inverse
