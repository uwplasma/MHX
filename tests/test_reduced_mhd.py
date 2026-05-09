from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mhx.benchmarks import linear_tearing_initial_state, run_linear_tearing_smoke
from mhx.config import MeshConfig, PhysicsConfig, RunConfig, TimeConfig
from mhx.diagnostics import (
    DiagnosticContext,
    DiagnosticSpec,
    compute_reduced_mhd_diagnostics,
    default_diagnostics_registry,
    fit_exponential_growth,
    island_width_from_mode,
    magnetic_divergence_linf,
    magnetic_energy,
    mode_amplitude,
    reconnected_flux_amplitude,
    rutherford_island_full_width,
    select_fit_window,
    total_energy,
    trajectory_energies,
    trajectory_mode_amplitude,
)
from mhx.equations.reduced_mhd import (
    current_density,
    poisson_bracket,
    reduced_mhd_rhs,
    stream_function,
)
from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import evolve_rk4


def test_poisson_bracket_vanishes_for_identical_fields() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    field = grid.sinusoid(mode=(2, 1))
    bracket = poisson_bracket(field, field, lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(bracket))) < 1.0e-10


def test_stream_function_and_current_density_conventions() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(32, 32)))
    psi = grid.sinusoid(mode=(1, 0))
    omega = -psi
    phi = stream_function(omega, lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(phi - psi))) < 1.0e-10
    assert float(jnp.max(jnp.abs(current_density(psi, lengths=grid.lengths) - psi))) < 1.0e-10


def test_reduced_mhd_rhs_linear_diffusion_limit() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(32, 32)))
    psi = grid.sinusoid(mode=(1, 0))
    state = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))
    params = ReducedMHDParams(resistivity=0.1, viscosity=0.0)
    rhs = reduced_mhd_rhs(state, params, lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(rhs.psi + 0.1 * psi))) < 1.0e-10
    assert float(jnp.max(jnp.abs(rhs.omega))) < 1.0e-10


def test_linear_tearing_smoke_runs_and_reports_energy() -> None:
    cfg = RunConfig(
        mesh=MeshConfig(shape=(16, 16)),
        time=TimeConfig(t1=0.02, dt=0.01, save_every=1),
        physics=PhysicsConfig(resistivity=1.0e-3, viscosity=1.0e-3),
    )
    trajectory, diagnostics = run_linear_tearing_smoke(cfg)
    lengths = CartesianGrid.from_mesh_config(cfg.mesh).lengths
    energies = trajectory_energies(trajectory, lengths=lengths)
    assert trajectory.states.psi.shape == (2, 16, 16)
    assert diagnostics["final_total_energy"] > 0.0
    assert diagnostics["final_total_energy"] <= diagnostics["initial_total_energy"]
    assert diagnostics["diagnostic_mode"] == [1, 1]
    assert diagnostics["gamma_fit"] < 0.0
    assert diagnostics["diagnostic_quantities"] == [
        "energy",
        "mode_growth",
        "divergence_error",
    ]
    assert diagnostics["final_magnetic_divergence_linf"] < 1.0e-10
    assert energies["total"].shape == (2,)


def test_rk4_diffusion_gradient_matches_finite_difference() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    psi = grid.sinusoid(mode=(1, 0))
    state0 = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))

    def objective(resistivity: float) -> jax.Array:
        params = ReducedMHDParams(resistivity=resistivity, viscosity=0.0)

        def rhs(state: ReducedMHDState) -> ReducedMHDState:
            return reduced_mhd_rhs(state, params, lengths=grid.lengths)

        trajectory = evolve_rk4(state0, rhs, dt=0.01, steps=3, save_every=3)
        final_state = ReducedMHDState(
            psi=trajectory.states.psi[-1],
            omega=trajectory.states.omega[-1],
        )
        return magnetic_energy(final_state, lengths=grid.lengths)

    eta = 0.01
    autodiff = jax.grad(objective)(eta)
    eps = 1.0e-5
    finite_difference = (objective(eta + eps) - objective(eta - eps)) / (2.0 * eps)
    assert float(jnp.abs(autodiff - finite_difference)) < 1.0e-8
    assert float(total_energy(state0, lengths=grid.lengths)) > 0.0


def test_mode_amplitude_and_growth_fit() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    x_mode = grid.cosinusoid(mode=(1, 0))
    y_mode = grid.cosinusoid(mode=(0, 1))
    state = ReducedMHDState(psi=x_mode * y_mode, omega=jnp.zeros(grid.shape))
    assert float(mode_amplitude(state, mode=(1, 1))) == pytest.approx(0.25)
    times = jnp.linspace(0.0, 1.0, 8)
    gamma = 0.4
    amplitudes = 2.0 * jnp.exp(gamma * times)
    assert float(fit_exponential_growth(times, amplitudes)) == pytest.approx(gamma)
    selected_times, selected_amplitudes = select_fit_window(
        times,
        amplitudes,
        window=(0.25, 0.75),
    )
    assert selected_times.shape[0] == 4
    assert selected_amplitudes.shape == selected_times.shape


def test_rutherford_island_width_proxy_recovers_cosine_amplitude() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(32, 32)))
    flux_amplitude = 3.0e-4
    state = ReducedMHDState(
        psi=flux_amplitude * grid.cosinusoid(mode=(1, 0)),
        omega=jnp.zeros(grid.shape),
    )
    reconnected_flux = reconnected_flux_amplitude(state, mode=(1, 0))
    expected_width = 4.0 * jnp.sqrt(flux_amplitude / 2.0)

    assert float(reconnected_flux) == pytest.approx(flux_amplitude)
    assert float(rutherford_island_full_width(reconnected_flux, magnetic_shear=2.0)) == (
        pytest.approx(float(expected_width))
    )
    assert float(island_width_from_mode(state, mode=(1, 0), magnetic_shear=2.0)) == (
        pytest.approx(float(expected_width))
    )
    with pytest.raises(ValueError, match="magnetic_shear"):
        rutherford_island_full_width(reconnected_flux, magnetic_shear=0.0)


def test_growth_fit_requires_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        fit_exponential_growth(jnp.asarray([0.0]), jnp.asarray([1.0]))
    with pytest.raises(ValueError, match="fit window"):
        select_fit_window(jnp.asarray([0.0, 1.0]), jnp.asarray([1.0, 2.0]), window=(1.0, 0.0))


def test_trajectory_mode_amplitude_shape() -> None:
    cfg = RunConfig(
        mesh=MeshConfig(shape=(16, 16)),
        time=TimeConfig(t1=0.02, dt=0.01, save_every=1),
        physics=PhysicsConfig(resistivity=1.0e-3, viscosity=1.0e-3),
    )
    trajectory, _ = run_linear_tearing_smoke(cfg)
    amplitudes = trajectory_mode_amplitude(trajectory, mode=(1, 1))
    assert amplitudes.shape == (2,)


def test_diagnostics_registry_selects_output_sets() -> None:
    cfg = RunConfig(
        mesh=MeshConfig(shape=(16, 16)),
        time=TimeConfig(t1=0.02, dt=0.01, save_every=1),
        physics=PhysicsConfig(resistivity=1.0e-3, viscosity=1.0e-3),
    )
    trajectory, _ = run_linear_tearing_smoke(cfg)
    grid = CartesianGrid.from_mesh_config(cfg.mesh)
    state0 = linear_tearing_initial_state(grid)
    energy_only = compute_reduced_mhd_diagnostics(
        trajectory,
        initial_state=state0,
        lengths=grid.lengths,
        quantities=("energy",),
        mode=(1, 1),
        fit_time_window=None,
    )
    assert energy_only["diagnostic_quantities"] == ["energy"]
    assert "final_total_energy" in energy_only
    assert "gamma_fit" not in energy_only

    registry = default_diagnostics_registry()
    assert registry.names() == ("divergence_error", "energy", "mode_growth")
    metadata = registry.metadata()
    assert {item["name"] for item in metadata} == set(registry.names())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            DiagnosticSpec(
                name="energy",
                description="duplicate",
                output_keys=("duplicate",),
                compute=lambda context: {"duplicate": 0.0},
            )
        )
    full = registry.compute(
        ("energy", "mode_growth", "divergence_error"),
        context=compute_diagnostics_context_for_test(trajectory, state0, grid),
    )
    assert full["gamma_fit"] < 0.0
    assert full["final_magnetic_divergence_linf"] < 1.0e-10
    with pytest.raises(KeyError, match="unknown diagnostic"):
        registry.get("not_registered")


def test_magnetic_divergence_error_is_spectral_zero() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    psi = grid.sinusoid(mode=(2, 1))
    state = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))
    assert float(magnetic_divergence_linf(state, lengths=grid.lengths)) < 1.0e-10


def compute_diagnostics_context_for_test(trajectory, state0, grid):
    return DiagnosticContext(
        trajectory=trajectory,
        initial_state=state0,
        lengths=grid.lengths,
        mode=(1, 1),
        fit_time_window=None,
    )
