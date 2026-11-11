"""TOML configs drive every ``Simulation`` model through ``mhx run``."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from typer.testing import CliRunner

from mhx import Simulation
from mhx.benchmarks.orszag_tang import orszag_tang_initial_state
from mhx.cli.main import _jsonable, app
from mhx.config import (
    MeshConfig,
    NumericsConfig,
    PhysicsConfig,
    RunConfig,
    load_config,
)
from mhx.grids import CartesianGrid
from mhx.physics import EquilibriumRegistry, build_equilibrium, default_equilibrium_registry
from mhx.physics.equilibria import _coerce_like

EXAMPLES = Path(__file__).parents[1] / "examples" / "toml"
TOML_FILES = sorted(EXAMPLES.glob("*.toml"))
THREE_DIMENSIONAL = ("mhd3d", "compressible")


def _small(config: RunConfig, outdir: Path) -> RunConfig:
    """Shrink an example config so a run takes about a second."""
    if config.physics.model == "compressible":
        shape = (8, 8, 4)
    elif config.physics.model == "mhd3d":
        shape = (8, 8, 8)
    else:
        shape = (16, 16)
    time = replace(config.time, t1=4.0 * config.time.dt, save_every=2)
    return replace(
        config,
        mesh=replace(config.mesh, shape=shape),
        time=time,
        output_dir=outdir,
    )


def test_example_tomls_are_present() -> None:
    names = {path.stem for path in TOML_FILES}
    assert {
        "double_harris",
        "double_harris_hyper",
        "double_harris_sharded",
        "orszag_tang_2d",
        "orszag_tang_3d",
        "taylor_green_3d",
        "alfven_collision_3d",
        "compressible_orszag_tang_3d",
    } <= names


@pytest.mark.parametrize("path", TOML_FILES, ids=lambda path: path.stem)
def test_example_config_loads_and_runs_through_the_cli(path: Path, tmp_path: Path) -> None:
    config = load_config(path)
    assert config.name == path.stem
    small = _small(config, tmp_path / "run")
    config_path = tmp_path / "small.toml"
    config_path.write_text(small.to_toml(), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path)])

    assert result.exit_code == 0, result.stdout
    run_dir = tmp_path / "run"
    for name in ("manifest.json", "trajectory.npz", "summary.png", "diagnostics.json"):
        assert (run_dir / name).stat().st_size > 0, name
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_level"] == "smoke"
    assert set(manifest["outputs"]) >= {"config", "diagnostics", "trajectory", "figure"}
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["final_magnetic_divergence_linf"] < 1.0e-6
    if "final_total_energy" in diagnostics:
        assert diagnostics["final_total_energy"] > 0.0
    else:
        assert diagnostics["final_density_relative_spread"] >= 0.0


def test_cli_outdir_override_applies_to_simulation_models(tmp_path: Path) -> None:
    config = _small(load_config(EXAMPLES / "orszag_tang_2d.toml"), tmp_path / "ignored")
    config_path = tmp_path / "run.toml"
    config_path.write_text(config.to_toml(), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path), "--outdir", str(tmp_path / "other")])

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "other" / "manifest.json").exists()
    assert not (tmp_path / "ignored").exists()


def test_from_config_matches_the_python_api(tmp_path: Path) -> None:
    config = _small(load_config(EXAMPLES / "double_harris.toml"), tmp_path)
    simulation = Simulation.from_config(config, verbose=False)

    assert simulation.equations == "reduced_mhd"
    assert simulation.shape == (16, 16)
    assert simulation.dt == config.time.dt
    assert simulation.steps == 4
    assert simulation.verbose is False
    assert simulation.equilibrium.width == pytest.approx(0.4)
    assert simulation.equilibrium.perturbation_mode == (2, 1)

    result = simulation.run()
    assert result.final_time == pytest.approx(4.0 * config.time.dt)


def test_from_config_builds_terms_and_three_dimensional_settings(tmp_path: Path) -> None:
    hyper = Simulation.from_config(
        _small(load_config(EXAMPLES / "double_harris_hyper.toml"), tmp_path), verbose=False
    )
    assert [type(term).__name__ for term in hyper.terms] == ["HyperResistivityTerm"]
    assert hyper.terms[0].eta4 == pytest.approx(1.0e-6)

    collision = Simulation.from_config(
        _small(load_config(EXAMPLES / "alfven_collision_3d.toml"), tmp_path), verbose=False
    )
    assert collision.equations == "mhd3d"
    assert collision.guide_field == (0.0, 0.0, 1.0)
    assert collision.integrator == "if_rk3"

    compressible = Simulation.from_config(
        _small(load_config(EXAMPLES / "compressible_orszag_tang_3d.toml"), tmp_path),
        verbose=False,
    )
    assert compressible.sound_speed == pytest.approx(5.0)
    assert compressible.bulk_viscosity == pytest.approx(5.0e-3)

    sharded = Simulation.from_config(
        _small(load_config(EXAMPLES / "double_harris_sharded.toml"), tmp_path), verbose=False
    )
    assert sharded.device_count == 2


def test_from_config_rejects_unsupported_settings() -> None:
    smoke = RunConfig()
    with pytest.raises(ValueError, match="smoke workflow"):
        Simulation.from_config(smoke)

    base = RunConfig(physics=PhysicsConfig(model="reduced_mhd", equilibrium="orszag_tang"))
    with pytest.raises(ValueError, match="time.t0"):
        Simulation.from_config(replace(base, time=replace(base.time, t0=0.5, t1=1.0, dt=0.1)))
    open_mesh = replace(base, mesh=MeshConfig(periodic=(True, False)))
    with pytest.raises(ValueError, match="periodic mesh"):
        Simulation.from_config(open_mesh)
    with pytest.raises(ValueError, match="initial_state"):
        Simulation.from_config(
            replace(base, physics=replace(base.physics, equilibrium="orszag_tang_3d"))
        )
    three_d = RunConfig(
        mesh=MeshConfig(shape=(8, 8, 8), lower=(0.0,) * 3, upper=(6.0,) * 3, periodic=(True,) * 3),
        physics=PhysicsConfig(model="mhd3d", equilibrium="cosine_tearing"),
    )
    with pytest.raises(ValueError, match="initial_fields"):
        Simulation.from_config(three_d)


def test_run_config_cross_section_validation() -> None:
    three_d_mesh = MeshConfig(
        shape=(8, 8, 8), lower=(0.0,) * 3, upper=(6.0,) * 3, periodic=(True,) * 3
    )
    with pytest.raises(ValueError, match="3-entry mesh.shape"):
        RunConfig(physics=PhysicsConfig(model="mhd3d", equilibrium="orszag_tang_3d")).validated()
    with pytest.raises(ValueError, match="2-entry mesh.shape"):
        RunConfig(mesh=three_d_mesh).validated()
    with pytest.raises(ValueError, match="time_integrator"):
        RunConfig(
            mesh=three_d_mesh,
            physics=PhysicsConfig(model="mhd3d", equilibrium="orszag_tang_3d"),
            numerics=NumericsConfig(time_integrator="backward_euler"),
        ).validated()
    with pytest.raises(ValueError, match="time_integrator"):
        RunConfig(numerics=NumericsConfig(time_integrator="etdrk4")).validated()
    open_three_d = replace(three_d_mesh, periodic=(True, True, False))
    with pytest.raises(ValueError, match="periodic mesh"):
        RunConfig(
            mesh=open_three_d,
            physics=PhysicsConfig(model="mhd3d", equilibrium="orszag_tang_3d"),
        ).validated()


def test_section_validation_for_the_new_fields() -> None:
    with pytest.raises(ValueError, match="physics.model"):
        PhysicsConfig(model="plasma").validated()
    with pytest.raises(ValueError, match="dissipation_order"):
        PhysicsConfig(dissipation_order=0).validated()
    with pytest.raises(ValueError, match="sound_speed"):
        PhysicsConfig(sound_speed=0.0).validated()
    with pytest.raises(ValueError, match="bulk_viscosity"):
        PhysicsConfig(bulk_viscosity=-1.0).validated()
    with pytest.raises(ValueError, match="device_count"):
        NumericsConfig(device_count=0).validated()
    with pytest.raises(ValueError, match="mesh.lower"):
        MeshConfig(shape=(8, 8, 8), lower=(0.0, 0.0), upper=(1.0,) * 3).validated()
    with pytest.raises(ValueError, match="2 or 3 entries"):
        MeshConfig(shape=(8, 8, 8, 8)).validated()


def test_new_fields_survive_a_toml_round_trip(tmp_path: Path) -> None:
    original = replace(
        load_config(EXAMPLES / "alfven_collision_3d.toml"),
        numerics=NumericsConfig(device_count=2, time_integrator="etdrk4"),
        physics=PhysicsConfig(
            model="mhd3d",
            equilibrium="single_mode_3d",
            equilibrium_parameters={"amplitude": 0.01, "mode": [1, 0, 2]},
            guide_field=(0.0, 0.5, 1.0),
            dissipation_order=2,
            sound_speed=2.0,
            bulk_viscosity=0.1,
        ),
    )
    path = tmp_path / "round_trip.toml"
    path.write_text(original.to_toml(), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.physics == original.physics
    assert loaded.numerics == original.numerics
    assert loaded.mesh == original.mesh
    assert loaded.physics.equilibrium_parameters["mode"] == [1, 0, 2]


def test_registry_lists_two_and_three_dimensional_equilibria() -> None:
    registry = default_equilibrium_registry()
    names = set(registry.names())
    assert {"orszag_tang", "orszag_tang_3d", "taylor_green_i", "taylor_green_a"} <= names
    assert {"taylor_green_c", "abc_flow", "single_mode_3d", "cp_alfven"} <= names
    assert "alfven_wave_collision" in names
    descriptions = {item.name: item.description for item in registry.metadata()}
    assert all(descriptions.values())
    assert descriptions["orszag_tang"].startswith("Incompressible")
    parameters = {item.name: item.parameters for item in registry.metadata()}
    assert parameters["orszag_tang_3d"] == {"beta": pytest.approx(0.8)}
    assert parameters["taylor_green_i"] == {"v0": 1.0, "b0": 1.0}


def test_build_equilibrium_casts_toml_values() -> None:
    equilibrium = build_equilibrium(
        "single_mode_3d", {"amplitude": 2, "mode": [1, 0, 2]}
    )
    assert equilibrium.amplitude == 2.0
    assert equilibrium.mode == (1, 0, 2)
    assert build_equilibrium("abc_flow", {"seed": 3.0}).seed == 3
    with pytest.raises(ValueError, match="unknown parameters.*bogus"):
        build_equilibrium("orszag_tang_3d", {"bogus": 1.0})
    with pytest.raises(ValueError, match="must be a list"):
        build_equilibrium("single_mode_3d", {"mode": 3})
    with pytest.raises(KeyError, match="unknown equilibrium"):
        build_equilibrium("no_such_equilibrium")


def test_coerce_like_handles_every_default_type() -> None:
    assert _coerce_like(True, 0, name="flag") is False
    assert _coerce_like(1, 2.0, name="count") == 2
    assert _coerce_like(1.0, 2, name="scale") == 2.0
    assert _coerce_like((), [1, 2], name="values") == (1.0, 2.0)
    assert _coerce_like("text", "other", name="label") == "other"


def test_orszag_tang_equilibrium_matches_the_benchmark_initial_state() -> None:
    grid = CartesianGrid(shape=(16, 16), lower=(0.0, 0.0), upper=(6.283185307179586,) * 2)
    from_registry = build_equilibrium("orszag_tang").initial_state(grid)
    from_benchmark = orszag_tang_initial_state(grid)
    assert jnp.allclose(from_registry.psi, from_benchmark.psi)
    assert jnp.allclose(from_registry.omega, from_benchmark.omega)
    x, y = grid.mesh()
    assert jnp.allclose(from_registry.psi, jnp.cos(y) + 0.5 * jnp.cos(2.0 * x))


def test_physics_equilibria_command_lists_three_dimensional_builders() -> None:
    result = CliRunner().invoke(app, ["physics", "equilibria"])
    assert result.exit_code == 0, result.stdout
    assert "orszag_tang_3d" in result.stdout
    assert "taylor_green_i" in result.stdout


def test_jsonable_converts_arrays_and_nested_containers() -> None:
    converted = _jsonable({"a": np.arange(3), "b": (1.5, [np.float64(2.0)]), "c": "text", "d": 4})
    assert converted == {"a": [0, 1, 2], "b": [1.5, [2.0]], "c": "text", "d": 4}
    assert json.dumps(converted)


def test_registry_metadata_handles_equilibria_without_dataclass_fields() -> None:
    class Plain:
        name = "plain"
        description = "A hand-written equilibrium."

    registry = EquilibriumRegistry()
    registry.register("plain", lambda _: Plain())
    (item,) = registry.metadata()
    assert item.parameters == {}
    assert item.description == "A hand-written equilibrium."


def test_mesh_entries_must_not_be_none() -> None:
    with pytest.raises(ValueError, match="mesh.lower must not be None"):
        MeshConfig.from_mapping({"shape": [8, 8], "lower": None})
