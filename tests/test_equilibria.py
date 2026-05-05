from __future__ import annotations

import jax.numpy as jnp
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.cli.main import app
from mhx.config import MeshConfig, PhysicsConfig, RunConfig, TimeConfig
from mhx.grids import CartesianGrid
from mhx.physics import (
    CosineTearingEquilibrium,
    ZeroEquilibrium,
    build_equilibrium,
    default_equilibrium_registry,
)


def test_default_equilibrium_registry_metadata_and_errors() -> None:
    registry = default_equilibrium_registry()
    assert registry.names() == ("cosine_tearing", "zero")
    metadata = {item.name: item for item in registry.metadata()}
    assert "current sheet" in metadata["cosine_tearing"].description
    assert metadata["cosine_tearing"].parameters["perturbation_amplitude"] == pytest.approx(
        1.0e-3
    )
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", lambda _: ZeroEquilibrium())
    with pytest.raises(KeyError, match="unknown equilibrium"):
        registry.create("missing")


def test_cosine_tearing_equilibrium_parameter_controls_perturbation() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    small = CosineTearingEquilibrium(perturbation_amplitude=1.0e-3).initial_state(grid)
    large = CosineTearingEquilibrium(perturbation_amplitude=2.0e-3).initial_state(grid)
    assert small.psi.shape == (16, 16)
    perturbation_delta = large.psi - small.psi
    x, y = grid.mesh()
    expected_delta = 1.0e-3 * jnp.cos(x) * jnp.cos(y)
    assert float(jnp.max(jnp.abs(perturbation_delta - expected_delta))) < 1.0e-12
    assert float(jnp.max(jnp.abs(small.omega))) == 0.0


def test_zero_equilibrium_and_configured_run_diagnostics() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(8, 8)))
    zero = build_equilibrium("zero").initial_state(grid)
    assert float(jnp.max(jnp.abs(zero.psi))) == 0.0
    assert float(jnp.max(jnp.abs(zero.omega))) == 0.0

    cfg = RunConfig(
        mesh=MeshConfig(shape=(8, 8)),
        time=TimeConfig(t1=0.02, dt=0.01, save_every=1),
        physics=PhysicsConfig(equilibrium="zero"),
    )
    trajectory, diagnostics = run_linear_tearing_smoke(cfg)
    assert diagnostics["equilibrium"] == "zero"
    assert diagnostics["equilibrium_parameters"] == {}
    assert float(jnp.max(jnp.abs(trajectory.states.psi))) == 0.0


def test_equilibrium_cli_list() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["physics", "equilibria"])
    assert result.exit_code == 0
    assert "cosine_tearing" in result.stdout
    assert "zero" in result.stdout
