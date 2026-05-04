from __future__ import annotations

import jax
import jax.numpy as jnp

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.config import MeshConfig, PhysicsConfig, RunConfig, TimeConfig
from mhx.diagnostics import magnetic_energy, total_energy, trajectory_energies
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
