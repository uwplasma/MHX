from __future__ import annotations

import jax.numpy as jnp
import pytest
from typer.testing import CliRunner

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.cli.main import app
from mhx.config import MeshConfig, PhysicsConfig, RunConfig, TimeConfig, load_config
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.physics import (
    PHYSICS_API_VERSION,
    ElectronPressureTensorTerm,
    HyperResistivityTerm,
    ToyHallOhmTerm,
    VorticityDragTerm,
    build_physics_terms,
    default_physics_registry,
)
from mhx.state import ReducedMHDParams, ReducedMHDState


def test_default_physics_registry_metadata_and_errors() -> None:
    registry = default_physics_registry()
    assert registry.names() == (
        "electron_pressure_tensor",
        "hyper_resistivity",
        "toy_hall_ohm",
        "vorticity_drag",
    )
    metadata = {item.name: item for item in registry.metadata()}
    assert metadata["hyper_resistivity"].api_version == PHYSICS_API_VERSION
    assert "fourth-order" in metadata["hyper_resistivity"].description
    assert "electron-pressure" in metadata["electron_pressure_tensor"].description
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", lambda _: HyperResistivityTerm())
    with pytest.raises(KeyError, match="unknown physics term"):
        registry.create("missing")


def test_hyper_resistivity_damps_single_mode() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(32, 32)))
    psi = grid.sinusoid(mode=(1, 0))
    state = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))
    term = HyperResistivityTerm(eta4=0.1)
    addition = term.rhs_addition(
        state,
        ReducedMHDParams(resistivity=0.0, viscosity=0.0),
        lengths=grid.lengths,
    )
    assert float(jnp.max(jnp.abs(addition.psi + 0.1 * psi))) < 1.0e-10


def test_vorticity_drag_term() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    omega = grid.sinusoid(mode=(1, 0))
    state = ReducedMHDState(psi=jnp.zeros_like(omega), omega=omega)
    addition = VorticityDragTerm(rate=0.25).rhs_addition(
        state,
        ReducedMHDParams(resistivity=0.0, viscosity=0.0),
        lengths=grid.lengths,
    )
    assert float(jnp.max(jnp.abs(addition.omega + 0.25 * omega))) < 1.0e-10
    assert float(jnp.max(jnp.abs(addition.psi))) == 0.0


def test_electron_pressure_tensor_term_damps_current_curvature() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(32, 32)))
    psi = grid.sinusoid(mode=(1, 0))
    state = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))
    addition = ElectronPressureTensorTerm(chi_x=0.1).rhs_addition(
        state,
        ReducedMHDParams(resistivity=0.0, viscosity=0.0),
        lengths=grid.lengths,
    )
    assert float(jnp.max(jnp.abs(addition.psi + 0.1 * psi))) < 1.0e-10
    assert float(jnp.max(jnp.abs(addition.omega))) == 0.0


def test_toy_hall_ohm_term_produces_nonzero_bracket_for_mixed_modes() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(32, 32)))
    x, y = grid.mesh()
    psi = jnp.sin(x) + jnp.sin(2.0 * y)
    state = ReducedMHDState(psi=psi, omega=jnp.zeros_like(psi))
    addition = ToyHallOhmTerm(ion_skin_depth=0.1).rhs_addition(
        state,
        ReducedMHDParams(resistivity=0.0, viscosity=0.0),
        lengths=grid.lengths,
    )
    assert addition.psi.shape == psi.shape
    assert float(jnp.max(jnp.abs(addition.psi))) > 0.1
    assert float(jnp.max(jnp.abs(addition.omega))) == 0.0


def test_configured_terms_change_rhs_and_run_diagnostics() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(16, 16)))
    psi = grid.sinusoid(mode=(1, 0))
    omega = grid.sinusoid(mode=(0, 1))
    state = ReducedMHDState(psi=psi, omega=omega)
    params = ReducedMHDParams(resistivity=0.0, viscosity=0.0)
    terms = build_physics_terms(
        ("hyper_resistivity", "vorticity_drag", "electron_pressure_tensor"),
        {
            "hyper_resistivity": {"eta4": 0.01, "nu4": 0.0},
            "vorticity_drag": {"rate": 0.2},
            "electron_pressure_tensor": {"chi_x": 0.01},
        },
    )
    rhs = reduced_mhd_rhs(state, params, lengths=grid.lengths, terms=terms)
    assert float(jnp.max(jnp.abs(rhs.psi))) > 0.0
    assert float(jnp.max(jnp.abs(rhs.omega))) > 0.0

    cfg = RunConfig(
        mesh=MeshConfig(shape=(16, 16)),
        time=TimeConfig(t1=0.02, dt=0.01, save_every=1),
        physics=PhysicsConfig(
            rhs_terms=("hyper_resistivity",),
            term_parameters={"hyper_resistivity": {"eta4": 1.0e-5}},
        ),
    )
    _, diagnostics = run_linear_tearing_smoke(cfg)
    assert diagnostics["physics_terms"] == ["hyper_resistivity"]


def test_physics_config_and_example_parse() -> None:
    cfg = load_config("examples/linear_tearing_hyper.toml")
    assert cfg.physics.rhs_terms == ("hyper_resistivity", "vorticity_drag")
    assert cfg.physics.term_parameters["hyper_resistivity"]["eta4"] == pytest.approx(1.0e-5)
    twofluid_cfg = load_config("examples/linear_tearing_twofluid_toy.toml")
    assert twofluid_cfg.physics.rhs_terms == ("electron_pressure_tensor", "toy_hall_ohm")
    assert twofluid_cfg.physics.term_parameters["toy_hall_ohm"]["ion_skin_depth"] == pytest.approx(
        0.01
    )
    serialized = cfg.to_toml()
    assert "[physics.equilibrium_parameters]" in serialized
    assert "[physics.term_parameters.hyper_resistivity]" in serialized
    assert "[physics.term_parameters.vorticity_drag]" in serialized
    with pytest.raises(ValueError, match="term_parameters"):
        PhysicsConfig(term_parameters={"hyper_resistivity": {"eta4": 1.0e-5}}).validated()


def test_physics_cli_list_and_lint() -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["physics", "list"])
    assert listed.exit_code == 0
    assert PHYSICS_API_VERSION in listed.stdout
    assert "hyper_resistivity" in listed.stdout
    assert "toy_hall_ohm" in listed.stdout
    linted = runner.invoke(app, ["physics", "lint", "hyper_resistivity"])
    assert linted.exit_code == 0
    assert "ok" in linted.stdout
    failed = runner.invoke(app, ["physics", "lint", "missing"])
    assert failed.exit_code != 0
