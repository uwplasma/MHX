"""Diagnostic figure dispatch for saved MHX run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mhx.config import RunConfig
from mhx.diagnostics import (
    DiagnosticContext,
    default_diagnostics_registry,
    load_diagnostics_entry_points,
    load_diagnostics_plugin_modules,
)
from mhx.grids import CartesianGrid
from mhx.io import read_reduced_mhd_trajectory_npz
from mhx.state import ReducedMHDState


def write_diagnostic_figures_for_run(
    run_dir: str | Path,
    *,
    figure_dir: str | Path | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Write selected diagnostic figure hooks for a saved run directory.

    Paths in the returned figure records are run-relative when the figure is
    inside ``run_dir``; otherwise they are absolute strings. Warnings are
    returned instead of raised so core figure generation can still succeed when
    an optional plugin is unavailable at plotting time.
    """
    directory = Path(run_dir)
    config_path = directory / "config_effective.json"
    diagnostics_path = directory / "diagnostics.json"
    trajectory_path = directory / "trajectory.npz"
    if not diagnostics_path.exists():
        return [], ["diagnostics.json missing; cannot write diagnostic figures"]
    if not config_path.exists():
        return [], ["config_effective.json missing; cannot write diagnostic figures"]
    if not trajectory_path.exists():
        return [], ["trajectory.npz missing; cannot write diagnostic figures"]
    try:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        quantities = tuple(str(item) for item in diagnostics.get("diagnostic_quantities", ()))
        if not quantities:
            return [], ["diagnostic_quantities missing; cannot write diagnostic figures"]
        config = RunConfig.from_mapping(json.loads(config_path.read_text(encoding="utf-8")))
        trajectory, _ = read_reduced_mhd_trajectory_npz(trajectory_path)
        context = _diagnostic_context(config, trajectory)
        registry = default_diagnostics_registry()
        load_diagnostics_entry_points(
            registry,
            config.diagnostics.plugin_entry_point_groups,
        )
        load_diagnostics_plugin_modules(registry, config.diagnostics.plugin_modules)
        output_dir = (
            Path(figure_dir) if figure_dir is not None else directory / "figures" / "diagnostics"
        )
        figure_paths = registry.write_figures(quantities, context, diagnostics, output_dir)
        figures = [
            {
                "key": key,
                "path": _path_for_record(directory, path),
            }
            for key, path in sorted(figure_paths.items())
        ]
        return figures, []
    except Exception as exc:
        return [], [f"could not write diagnostic figures: {exc}"]


def _diagnostic_context(config: RunConfig, trajectory: Any) -> DiagnosticContext:
    grid = CartesianGrid.from_mesh_config(config.mesh)
    initial_state = ReducedMHDState(
        psi=trajectory.states.psi[0],
        omega=trajectory.states.omega[0],
    )
    return DiagnosticContext(
        trajectory=trajectory,
        initial_state=initial_state,
        lengths=grid.lengths,
        mode=config.diagnostics.mode,
        fit_time_window=config.diagnostics.fit_time_window,
    )


def _path_for_record(directory: Path, path: str | Path) -> str:
    resolved = Path(path)
    try:
        return resolved.relative_to(directory).as_posix()
    except ValueError:
        return str(resolved)
