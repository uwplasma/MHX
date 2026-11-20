"""Tests for the Arakawa-bracket reduced-MHD right-hand side and its linearizations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mhx.config import MeshConfig
from mhx.equations.arakawa_reduced_mhd import (
    arakawa_poisson_bracket,
    arakawa_reduced_mhd_rhs,
    current_density,
    finite_difference_linearized_arakawa_reduced_mhd_rhs,
    linearized_arakawa_reduced_mhd_operator,
    linearized_arakawa_reduced_mhd_rhs,
    stream_function,
)
from mhx.equations.reduced_mhd import poisson_bracket, reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import laplacian
from mhx.physics import HyperResistivityTerm
from mhx.state import ReducedMHDParams, ReducedMHDState, flatten_reduced_mhd_state

PARAMS = ReducedMHDParams(resistivity=0.02, viscosity=0.03)


def _grid(n: int = 16) -> CartesianGrid:
    return CartesianGrid.from_mesh_config(MeshConfig(shape=(n, n)))


def _random_fields(grid: CartesianGrid, seed: int = 0):
    keys = jax.random.split(jax.random.key(seed), 3)
    return tuple(jax.random.normal(key, grid.shape) for key in keys)


def _smooth_state(grid: CartesianGrid) -> ReducedMHDState:
    x, y = grid.mesh()
    psi = jnp.cos(x) * jnp.sin(y) + 0.3 * jnp.cos(2 * y + x)
    omega = 0.5 * jnp.sin(x + y) - 0.2 * jnp.cos(2 * x)
    return ReducedMHDState(psi=psi, omega=omega)


def _mean(field) -> float:
    return float(jnp.mean(field))


def test_bracket_is_antisymmetric_and_vanishes_on_equal_fields() -> None:
    grid = _grid()
    a, b, _ = _random_fields(grid)
    ab = arakawa_poisson_bracket(a, b, lengths=grid.lengths)
    ba = arakawa_poisson_bracket(b, a, lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(ab + ba))) < 1.0e-12
    assert float(jnp.max(jnp.abs(arakawa_poisson_bracket(a, a, lengths=grid.lengths)))) < 1e-12


def test_bracket_conserves_mean_and_quadratic_invariants_to_roundoff() -> None:
    grid = _grid()
    a, b, _ = _random_fields(grid, seed=3)
    bracket = arakawa_poisson_bracket(a, b, lengths=grid.lengths)
    scale = float(jnp.max(jnp.abs(bracket)))
    assert scale > 1.0  # the bracket of random fields is not trivially zero
    # int {a,b} = 0, int a {a,b} = 0 and int b {a,b} = 0 (energy/enstrophy-type invariants).
    assert abs(_mean(bracket)) < 1e-12 * scale
    assert abs(_mean(a * bracket)) < 1e-12 * scale * float(jnp.max(jnp.abs(a)))
    assert abs(_mean(b * bracket)) < 1e-12 * scale * float(jnp.max(jnp.abs(b)))


def test_bracket_matches_analytic_value_for_plane_waves() -> None:
    grid = _grid(32)
    x, y = grid.mesh()
    # {sin x, sin y} = cos x cos y
    bracket = arakawa_poisson_bracket(jnp.sin(x), jnp.sin(y), lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(bracket - jnp.cos(x) * jnp.cos(y)))) < 1e-11


def test_stream_function_and_current_density_conventions() -> None:
    grid = _grid(32)
    psi = grid.sinusoid(mode=(1, 0))
    assert float(jnp.max(jnp.abs(stream_function(-psi, lengths=grid.lengths) - psi))) < 1e-10
    assert float(jnp.max(jnp.abs(current_density(psi, lengths=grid.lengths) - psi))) < 1e-10


def test_rhs_linear_diffusion_limit() -> None:
    grid = _grid(32)
    psi = grid.sinusoid(mode=(1, 0))
    state = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))
    rhs = arakawa_reduced_mhd_rhs(
        state, ReducedMHDParams(resistivity=0.1, viscosity=0.0), lengths=grid.lengths
    )
    assert float(jnp.max(jnp.abs(rhs.psi + 0.1 * psi))) < 1e-10
    assert float(jnp.max(jnp.abs(rhs.omega))) < 1e-10


def test_rhs_matches_spectral_rhs_on_low_modes() -> None:
    grid = _grid(32)
    state = _smooth_state(grid)
    arakawa = arakawa_reduced_mhd_rhs(state, PARAMS, lengths=grid.lengths)
    spectral = reduced_mhd_rhs(state, PARAMS, lengths=grid.lengths)
    # Only aliasing differs between the two discretizations; low modes are resolved.
    assert float(jnp.max(jnp.abs(arakawa.psi - spectral.psi))) < 1e-10
    assert float(jnp.max(jnp.abs(arakawa.omega - spectral.omega))) < 1e-10
    # The bracket alone also agrees with the plain spectral Poisson bracket.
    ours = arakawa_poisson_bracket(state.psi, state.omega, lengths=grid.lengths)
    plain = poisson_bracket(state.psi, state.omega, lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(ours - plain))) < 1e-10


def test_rhs_ideal_limit_conserves_energy_and_cross_helicity() -> None:
    grid = _grid(32)
    state = _smooth_state(grid)
    ideal = ReducedMHDParams(resistivity=0.0, viscosity=0.0)
    rhs = arakawa_reduced_mhd_rhs(state, ideal, lengths=grid.lengths)
    phi = stream_function(state.omega, lengths=grid.lengths)
    # dE/dt for E = <|grad phi|^2 + |grad psi|^2>/2 equals -<phi omega_t> - <lap(psi) psi_t>.
    d_energy = -_mean(phi * rhs.omega) - _mean(laplacian(state.psi, lengths=grid.lengths) * rhs.psi)
    assert abs(d_energy) < 1e-10
    # Mean flux is conserved exactly by the conservative form.
    assert abs(_mean(rhs.psi)) < 1e-12
    assert abs(_mean(rhs.omega)) < 1e-12


def test_rhs_applies_physics_terms() -> None:
    grid = _grid()
    state = _smooth_state(grid)
    term = HyperResistivityTerm(eta4=0.01, nu4=0.02)
    base = arakawa_reduced_mhd_rhs(state, PARAMS, lengths=grid.lengths)
    with_term = arakawa_reduced_mhd_rhs(state, PARAMS, lengths=grid.lengths, terms=(term,))
    expected = term.rhs_addition(state, PARAMS, lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(with_term.psi - base.psi - expected.psi))) < 1e-12
    assert float(jnp.max(jnp.abs(with_term.omega - base.omega - expected.omega))) < 1e-12
    assert float(jnp.max(jnp.abs(expected.psi))) > 0.0


def test_linearization_matches_finite_difference_and_operator() -> None:
    grid = _grid()
    state = _smooth_state(grid)
    x, y = grid.mesh()
    perturbation = ReducedMHDState(psi=0.1 * jnp.sin(y + x), omega=0.1 * jnp.cos(2 * x))
    exact = linearized_arakawa_reduced_mhd_rhs(state, perturbation, PARAMS, lengths=grid.lengths)
    approx = finite_difference_linearized_arakawa_reduced_mhd_rhs(
        state, perturbation, PARAMS, lengths=grid.lengths, epsilon=1e-4
    )
    assert float(jnp.max(jnp.abs(exact.psi - approx.psi))) < 1e-6
    assert float(jnp.max(jnp.abs(exact.omega - approx.omega))) < 1e-6
    assert float(jnp.max(jnp.abs(exact.omega))) > 1e-3
    operator = linearized_arakawa_reduced_mhd_operator(state, PARAMS, lengths=grid.lengths)
    vector = flatten_reduced_mhd_state(perturbation)
    assert operator.shape == (2 * grid.shape[0] * grid.shape[1],)
    assert float(jnp.max(jnp.abs(operator(vector) - flatten_reduced_mhd_state(exact)))) < 1e-12
    with pytest.raises(ValueError, match="expected shape"):
        operator(vector[:-1])


@pytest.mark.parametrize("epsilon", [0.0, -1e-3])
def test_finite_difference_linearization_rejects_nonpositive_epsilon(epsilon) -> None:
    grid = _grid()
    state = _smooth_state(grid)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        finite_difference_linearized_arakawa_reduced_mhd_rhs(
            state, state, PARAMS, lengths=grid.lengths, epsilon=epsilon
        )
