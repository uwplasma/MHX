from __future__ import annotations

import jax.numpy as jnp
import pytest

from mhx.grids import CartesianGrid
from mhx.numerics.spectral import (
    dealiased_product,
    fft_derivative,
    gradient,
    inverse_laplacian,
    laplacian,
    spectral_filter,
    spectral_wavenumbers,
    two_thirds_mask,
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


def test_two_thirds_product_removes_quadratic_alias() -> None:
    points = 24
    coordinate = 2.0 * jnp.pi * jnp.arange(points) / points
    mode = jnp.cos(7.0 * coordinate)

    unfiltered = dealiased_product(mode, mode, dealiasing="none")
    filtered = dealiased_product(mode, mode, dealiasing="two_thirds")

    assert float(jnp.max(jnp.abs(unfiltered - 0.5))) > 0.4
    assert float(jnp.max(jnp.abs(filtered - 0.5))) < 1.0e-12
    mask = two_thirds_mask((points,))
    assert bool(mask[7])
    assert not bool(mask[8])


def test_spectral_operator_validation() -> None:
    grid = CartesianGrid(shape=(8, 8), lower=(0.0, 0.0), upper=(1.0, 1.0))
    field = grid.sinusoid()
    assert fft_derivative(field, axis=0, length=1.0, order=0) is not None
    complex_field = field.astype(jnp.complex128) * (1.0 + 1.0j)
    complex_derivative = fft_derivative(complex_field, axis=0, length=1.0)
    assert jnp.iscomplexobj(complex_derivative)
    assert jnp.iscomplexobj(spectral_filter(complex_field))
    assert jnp.iscomplexobj(inverse_laplacian(complex_field, lengths=(1.0, 1.0)))
    assert jnp.allclose(spectral_filter(field, dealiasing="none"), field)
    with pytest.raises(ValueError, match="order"):
        fft_derivative(field, axis=0, length=1.0, order=-1)
    with pytest.raises(ValueError, match="expected"):
        gradient(field, lengths=(1.0,))
    with pytest.raises(ValueError, match="expected"):
        laplacian(field, lengths=(1.0,))
    with pytest.raises(ValueError, match="expected"):
        inverse_laplacian(field, lengths=(1.0,))
    with pytest.raises(ValueError, match="shape"):
        two_thirds_mask((1,))
    with pytest.raises(ValueError, match="dealiasing"):
        spectral_filter(field, dealiasing="invalid")
    with pytest.raises(ValueError, match="equal shapes"):
        dealiased_product(field, field[:, :-1])


@pytest.mark.parametrize("shape", [(12, 8), (9, 7)])
def test_real_fields_use_the_half_spectrum_with_complex_results(shape) -> None:
    import jax

    from mhx.diagnostics.reduced_mhd import mode_amplitude
    from mhx.numerics import spectral_diffusion_preconditioner
    from mhx.state import ReducedMHDParams, ReducedMHDState

    field = jax.random.normal(jax.random.PRNGKey(3), shape)
    lengths = (2.0, 3.0)
    as_complex = field.astype(jnp.complex128 if field.dtype == jnp.float64 else jnp.complex64)
    pairs = [
        (spectral_filter(field), spectral_filter(as_complex)),
        (inverse_laplacian(field, lengths=lengths), inverse_laplacian(as_complex, lengths=lengths)),
    ]
    for order in (1, 2, 3):
        for axis in (0, 1):
            pairs.append((
                fft_derivative(field, axis=axis, length=lengths[axis], order=order),
                fft_derivative(as_complex, axis=axis, length=lengths[axis], order=order),
            ))
    apply = spectral_diffusion_preconditioner(
        ReducedMHDParams(resistivity=0.1, viscosity=0.2), lengths=lengths, dt=0.5
    )
    real_state = apply(ReducedMHDState(psi=field, omega=field))
    complex_state = apply(ReducedMHDState(psi=as_complex, omega=as_complex))
    pairs += [(real_state.psi, complex_state.psi), (real_state.omega, complex_state.omega)]
    tol = 1e-12 if field.dtype == jnp.float64 else 1e-5
    for real, reference in pairs:
        assert jnp.isrealobj(real)
        assert jnp.allclose(real, reference.real, atol=tol * 100)
    reference = jnp.fft.fftn(field) / field.size
    for mode in [(2, 1), (-2, -1), (1, -3), (0, shape[1] // 2)]:
        expected = jnp.abs(reference[mode[0] % shape[0], mode[1] % shape[1]])
        state = ReducedMHDState(psi=field, omega=field)
        assert jnp.allclose(mode_amplitude(state, mode=mode), expected, rtol=tol)
