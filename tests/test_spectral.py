from __future__ import annotations

import jax.numpy as jnp
import pytest

from mhx.grids import CartesianGrid
from mhx.numerics.spectral import (
    fft_derivative,
    gradient,
    inverse_laplacian,
    laplacian,
    spectral_wavenumbers,
)


def test_wavenumbers_validate_inputs() -> None:
    with pytest.raises(ValueError, match="points"):
        spectral_wavenumbers(1, 1.0)
    with pytest.raises(ValueError, match="length"):
        spectral_wavenumbers(8, 0.0)


def test_fft_derivative_matches_periodic_sinusoid() -> None:
    grid = CartesianGrid(shape=(64, 32), lower=(0.0, 0.0), upper=(2.0 * jnp.pi, 2.0 * jnp.pi))
    field = grid.sinusoid(mode=(3, 0))
    derivative = fft_derivative(field, axis=0, length=grid.lengths[0])
    expected = 3.0 * grid.cosinusoid(mode=(3, 0))
    assert float(jnp.max(jnp.abs(derivative - expected))) < 1.0e-10


def test_gradient_and_laplacian_shapes_and_values() -> None:
    grid = CartesianGrid(shape=(32, 32), lower=(0.0, 0.0), upper=(2.0 * jnp.pi, 2.0 * jnp.pi))
    field = grid.sinusoid(mode=(2, 1))
    grad_x, grad_y = gradient(field, lengths=grid.lengths)
    lap = laplacian(field, lengths=grid.lengths)
    assert grad_x.shape == grid.shape
    assert grad_y.shape == grid.shape
    assert lap.shape == grid.shape
    assert float(jnp.max(jnp.abs(lap + 5.0 * field))) < 1.0e-10


def test_inverse_laplacian_recovers_zero_mean_field() -> None:
    grid = CartesianGrid(shape=(32, 32), lower=(0.0, 0.0), upper=(2.0 * jnp.pi, 2.0 * jnp.pi))
    field = grid.sinusoid(mode=(2, 1))
    solved = inverse_laplacian(field, lengths=grid.lengths)
    residual = laplacian(solved, lengths=grid.lengths) - field
    assert float(jnp.max(jnp.abs(residual))) < 1.0e-10


def test_spectral_operator_validation() -> None:
    grid = CartesianGrid(shape=(8, 8), lower=(0.0, 0.0), upper=(1.0, 1.0))
    field = grid.sinusoid()
    assert fft_derivative(field, axis=0, length=1.0, order=0) is not None
    complex_field = field.astype(jnp.complex128) * (1.0 + 1.0j)
    complex_derivative = fft_derivative(complex_field, axis=0, length=1.0)
    assert jnp.iscomplexobj(complex_derivative)
    with pytest.raises(ValueError, match="order"):
        fft_derivative(field, axis=0, length=1.0, order=-1)
    with pytest.raises(ValueError, match="expected"):
        gradient(field, lengths=(1.0,))
    with pytest.raises(ValueError, match="expected"):
        laplacian(field, lengths=(1.0,))
    with pytest.raises(ValueError, match="expected"):
        inverse_laplacian(field, lengths=(1.0,))
