"""Tests for diagnostic figure dispatch on saved run directories (error and success paths)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import jax.numpy as jnp
import pytest

from mhx.config import DiagnosticsConfig, MeshConfig, RunConfig
from mhx.io import write_reduced_mhd_trajectory_npz
from mhx.plotting.diagnostics import _path_for_record, write_diagnostic_figures_for_run
from mhx.state import ReducedMHDState, ReducedMHDTrajectory

PLUGIN_SOURCE = textwrap.dedent(
    """
    from mhx.diagnostics import DiagnosticSpec


    def _compute(context):
        return {"extra_psi_max": float(context.initial_state.psi.max())}


    def _figure(context, diagnostics, output_dir):
        path = output_dir / "extra.png"
        path.write_bytes(b"png-bytes")
        return {"extra_figure": path}


    def register_diagnostics(registry):
        registry.register(
            DiagnosticSpec(
                name="extra_fig",
                description="test",
                output_keys=("extra_psi_max",),
                compute=_compute,
                figure=_figure,
            )
        )
    """
)


def _write_run(directory: Path, *, quantities=("extra_fig",), plugin="extra_fig_plugin") -> None:
    config = RunConfig(
        mesh=MeshConfig(shape=(8, 8)),
        diagnostics=DiagnosticsConfig(plugin_modules=(plugin,) if plugin else ()),
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config_effective.json").write_text(json.dumps(config.to_dict()), "utf-8")
    diagnostics = {"diagnostic_quantities": list(quantities)}
    (directory / "diagnostics.json").write_text(json.dumps(diagnostics), "utf-8")
    psi = jnp.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8)
    trajectory = ReducedMHDTrajectory(
        times=jnp.asarray([0.0, 1.0]), states=ReducedMHDState(psi=psi, omega=0.0 * psi)
    )
    write_reduced_mhd_trajectory_npz(
        directory / "trajectory.npz", trajectory=trajectory, config=config.to_dict(), diagnostics={}
    )


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("diagnostics.json", "diagnostics.json missing"),
        ("config_effective.json", "config_effective.json missing"),
        ("trajectory.npz", "trajectory.npz missing"),
    ],
)
def test_missing_run_artifacts_are_reported_as_warnings(tmp_path, missing, message) -> None:
    _write_run(tmp_path, plugin="")
    (tmp_path / missing).unlink()
    figures, warnings = write_diagnostic_figures_for_run(tmp_path)
    assert figures == []
    assert len(warnings) == 1
    assert message in warnings[0]


def test_empty_diagnostic_quantities_is_a_warning(tmp_path) -> None:
    _write_run(tmp_path, quantities=(), plugin="")
    figures, warnings = write_diagnostic_figures_for_run(tmp_path)
    assert figures == []
    assert warnings == ["diagnostic_quantities missing; cannot write diagnostic figures"]


def test_unreadable_inputs_and_unknown_quantity_become_warnings(tmp_path) -> None:
    _write_run(tmp_path, plugin="")
    (tmp_path / "config_effective.json").write_text("{not json", "utf-8")
    figures, warnings = write_diagnostic_figures_for_run(tmp_path)
    assert figures == []
    assert warnings[0].startswith("could not write diagnostic figures:")
    _write_run(tmp_path, quantities=("no_such_diagnostic",), plugin="")
    figures, warnings = write_diagnostic_figures_for_run(tmp_path)
    assert figures == []
    assert "no_such_diagnostic" in warnings[0]
    _write_run(tmp_path, plugin="module_that_does_not_exist_anywhere")
    figures, warnings = write_diagnostic_figures_for_run(tmp_path)
    assert figures == [] and "could not write diagnostic figures" in warnings[0]


def test_plugin_figure_inside_and_outside_run_directory(tmp_path, monkeypatch) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "extra_fig_plugin.py").write_text(PLUGIN_SOURCE, "utf-8")
    monkeypatch.syspath_prepend(str(plugin_dir))
    run = tmp_path / "run"
    _write_run(run)

    figures, warnings = write_diagnostic_figures_for_run(run)
    assert warnings == []
    assert figures == [{"key": "extra_figure", "path": "figures/diagnostics/extra.png"}]
    assert (run / "figures" / "diagnostics" / "extra.png").read_bytes() == b"png-bytes"

    elsewhere = tmp_path / "elsewhere"
    figures, warnings = write_diagnostic_figures_for_run(run, figure_dir=elsewhere)
    assert warnings == []
    assert figures == [{"key": "extra_figure", "path": str(elsewhere / "extra.png")}]
    assert (elsewhere / "extra.png").exists()


def test_path_for_record_relative_and_absolute(tmp_path) -> None:
    inside = tmp_path / "figures" / "a.png"
    assert _path_for_record(tmp_path, inside) == "figures/a.png"
    outside = tmp_path.parent / "other" / "b.png"
    assert _path_for_record(tmp_path, outside) == str(outside)
