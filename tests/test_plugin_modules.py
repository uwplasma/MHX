from __future__ import annotations

import json
import sys
import types

import pytest
from typer.testing import CliRunner

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.cli.main import app
from mhx.config import (
    DiagnosticsConfig,
    MeshConfig,
    PhysicsConfig,
    RunConfig,
    TimeConfig,
    load_config,
)
from mhx.diagnostics import default_diagnostics_registry, load_diagnostics_plugin_modules
from mhx.diagnostics.reduced_mhd import load_diagnostics_entry_points
from mhx.physics import (
    build_physics_terms,
    default_physics_registry,
    load_physics_entry_points,
    load_physics_plugin_modules,
)

PLUGIN_SOURCE = '''
from dataclasses import dataclass
from typing import ClassVar
import jax.numpy as jnp
from mhx.diagnostics import DiagnosticSpec
from mhx.physics import PHYSICS_API_VERSION
from mhx.state import ReducedMHDState

@dataclass(frozen=True)
class UniformFluxDrive:
    amplitude: float = 0.0
    name: ClassVar[str] = "uniform_flux_drive"
    api_version: ClassVar[str] = PHYSICS_API_VERSION
    description: ClassVar[str] = "Uniform flux source for plugin tests."

    def rhs_addition(self, state, params, *, lengths):
        del params, lengths
        return ReducedMHDState(
            psi=self.amplitude * jnp.ones_like(state.psi),
            omega=jnp.zeros_like(state.omega),
        )

def _factory(parameters):
    return UniformFluxDrive(amplitude=float(parameters.get("amplitude", 0.0)))

def _psi_mean(context):
    return {"final_psi_mean": float(jnp.mean(context.trajectory.states.psi[-1]))}

def register_physics(registry):
    registry.register("uniform_flux_drive", _factory)

def register_diagnostics(registry):
    registry.register(
        DiagnosticSpec(
            name="final_psi_mean",
            description="Final mean flux for plugin tests.",
            output_keys=("final_psi_mean",),
            compute=_psi_mean,
        )
    )
'''


def test_config_driven_physics_and_diagnostics_plugin_modules(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "user_plugin.py"
    module_path.write_text(PLUGIN_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    cfg = RunConfig(
        mesh=MeshConfig(shape=(8, 8)),
        time=TimeConfig(t1=0.02, dt=0.01),
        physics=PhysicsConfig(
            plugin_modules=("user_plugin",),
            rhs_terms=("uniform_flux_drive",),
            term_parameters={"uniform_flux_drive": {"amplitude": 1.0e-4}},
        ),
        diagnostics=DiagnosticsConfig(
            quantities=("energy", "mode_growth", "final_psi_mean"),
            plugin_modules=("user_plugin",),
        ),
    )
    _, diagnostics = run_linear_tearing_smoke(cfg)
    assert diagnostics["physics_plugin_modules"] == ["user_plugin"]
    assert diagnostics["diagnostic_plugin_modules"] == ["user_plugin"]
    assert diagnostics["physics_terms"] == ["uniform_flux_drive"]
    assert diagnostics["diagnostic_quantities"] == ["energy", "mode_growth", "final_psi_mean"]
    assert diagnostics["final_psi_mean"] == pytest.approx(2.0e-6, abs=1.0e-12)


def test_plugin_loaders_register_metadata_and_reject_missing_hooks(tmp_path, monkeypatch) -> None:
    (tmp_path / "user_plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
    (tmp_path / "bad_plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    physics_registry = load_physics_plugin_modules(default_physics_registry(), ("user_plugin",))
    assert "uniform_flux_drive" in physics_registry.names()
    term = build_physics_terms(
        ("uniform_flux_drive",),
        {"uniform_flux_drive": {"amplitude": 0.2}},
        registry=physics_registry,
    )[0]
    assert term.amplitude == pytest.approx(0.2)

    diagnostics_registry = load_diagnostics_plugin_modules(
        default_diagnostics_registry(),
        ("user_plugin",),
    )
    assert "final_psi_mean" in diagnostics_registry.names()

    with pytest.raises(AttributeError, match="register_physics"):
        load_physics_plugin_modules(default_physics_registry(), ("bad_plugin",))
    with pytest.raises(AttributeError, match="register_diagnostics"):
        load_diagnostics_plugin_modules(default_diagnostics_registry(), ("bad_plugin",))


def test_entry_point_plugin_loaders(monkeypatch) -> None:
    module = types.ModuleType("entry_plugin")
    monkeypatch.setitem(sys.modules, "entry_plugin", module)
    exec(PLUGIN_SOURCE, module.__dict__)

    class FakeEntryPoint:
        def __init__(self, name, plugin):
            self.name = name
            self._plugin = plugin

        def load(self):
            return self._plugin

    class FakeEntryPoints(tuple):
        def select(self, *, group):
            if group == "mhx.physics":
                return (FakeEntryPoint("physics_hook", module.register_physics),)
            if group == "mhx.diagnostics":
                return (FakeEntryPoint("diagnostics_module", module),)
            return ()

    monkeypatch.setattr(
        "mhx.physics.terms.importlib_metadata.entry_points",
        lambda: FakeEntryPoints(),
    )
    monkeypatch.setattr(
        "mhx.diagnostics.reduced_mhd.importlib_metadata.entry_points",
        lambda: FakeEntryPoints(),
    )

    physics_registry = load_physics_entry_points(default_physics_registry(), ("mhx.physics",))
    assert "uniform_flux_drive" in physics_registry.names()

    diagnostics_registry = load_diagnostics_entry_points(
        default_diagnostics_registry(),
        ("mhx.diagnostics",),
    )
    assert "final_psi_mean" in diagnostics_registry.names()


def test_plugin_loaders_fallback_to_cwd_file_for_shadowed_parent(
    tmp_path,
    monkeypatch,
) -> None:
    physics_plugin_dir = tmp_path / "shadowed_physics"
    diagnostics_plugin_dir = tmp_path / "shadowed_diagnostics"
    physics_plugin_dir.mkdir()
    diagnostics_plugin_dir.mkdir()
    (physics_plugin_dir / "local_plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
    (diagnostics_plugin_dir / "local_plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "shadowed_physics", types.ModuleType("shadowed_physics"))
    monkeypatch.setitem(
        sys.modules,
        "shadowed_diagnostics",
        types.ModuleType("shadowed_diagnostics"),
    )

    physics_registry = load_physics_plugin_modules(
        default_physics_registry(),
        ("shadowed_physics.local_plugin",),
    )
    assert "uniform_flux_drive" in physics_registry.names()

    diagnostics_registry = load_diagnostics_plugin_modules(
        default_diagnostics_registry(),
        ("shadowed_diagnostics.local_plugin",),
    )
    assert "final_psi_mean" in diagnostics_registry.names()


def test_plugin_demo_example_runs_and_cli_lists_plugins(tmp_path) -> None:
    cfg = load_config("examples/linear_tearing_plugin_demo.toml")
    assert cfg.physics.plugin_modules == ("examples.local_extension_plugin",)
    assert cfg.diagnostics.plugin_modules == ("examples.local_extension_plugin",)

    outdir = tmp_path / "plugin-demo"
    runner = CliRunner()
    run_result = runner.invoke(
        app,
        ["run", "examples/linear_tearing_plugin_demo.toml", "--outdir", str(outdir)],
    )
    assert run_result.exit_code == 0, run_result.stdout
    diagnostics = json.loads((outdir / "diagnostics.json").read_text())
    assert diagnostics["final_flux_l2"] > 0.0
    assert diagnostics["physics_terms"] == ["example_flux_drive"]

    physics_result = runner.invoke(
        app,
        [
            "physics",
            "list-with-plugins",
            "--plugin-module",
            "examples.local_extension_plugin",
        ],
    )
    assert physics_result.exit_code == 0, physics_result.stdout
    assert "example_flux_drive" in physics_result.stdout

    diagnostics_result = runner.invoke(
        app,
        [
            "diagnostics",
            "list-with-plugins",
            "--plugin-module",
            "examples.local_extension_plugin",
        ],
    )
    assert diagnostics_result.exit_code == 0, diagnostics_result.stdout
    assert "final_flux_l2" in diagnostics_result.stdout

    diagnostics_lint_result = runner.invoke(
        app,
        [
            "diagnostics",
            "lint",
            "final_flux_l2",
            "--plugin-module",
            "examples.local_extension_plugin",
        ],
    )
    assert diagnostics_lint_result.exit_code == 0, diagnostics_lint_result.stdout
    assert "final_flux_l2: ok" in diagnostics_lint_result.stdout

    report_result = runner.invoke(app, ["report", str(outdir)])
    assert report_result.exit_code == 0, report_result.stdout
    report = json.loads((outdir / "report.json").read_text())
    assert report["additional_scalar_diagnostics"]["final_flux_l2"] > 0.0
    assert report["warnings"] == []
    assert any(item["name"] == "final_flux_l2" for item in report["diagnostic_metadata"])
    assert "`final_flux_l2`" in (outdir / "report.md").read_text()
    assert "Diagnostic registry metadata" in (outdir / "report.md").read_text()

    physics_lint_result = runner.invoke(
        app,
        [
            "physics",
            "lint",
            "example_flux_drive",
            "--plugin-module",
            "examples.local_extension_plugin",
        ],
    )
    assert physics_lint_result.exit_code == 0, physics_lint_result.stdout
    assert "example_flux_drive: ok" in physics_lint_result.stdout


def test_plugin_module_config_validation_errors() -> None:
    with pytest.raises(ValueError, match="plugin_modules"):
        PhysicsConfig(plugin_modules=("same", "same")).validated()
    with pytest.raises(ValueError, match="plugin_modules"):
        DiagnosticsConfig(plugin_modules=("same", "same")).validated()
