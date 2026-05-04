"""MHX command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mhx._version import __version__
from mhx.benchmarks import run_linear_tearing_smoke, write_run_report
from mhx.config import RunConfig, load_config
from mhx.grids import CartesianGrid
from mhx.io import (
    read_reduced_mhd_trajectory_npz,
    write_manifest,
    write_reduced_mhd_trajectory_npz,
)
from mhx.plotting import plot_energy_history, plot_flux_contours, plot_flux_gif, plot_mode_amplitude
from mhx.state import ReducedMHDState

app = typer.Typer(no_args_is_help=True, help="MHX differentiable MHD workflows.")


@app.command()
def version() -> None:
    """Print the MHX package version."""
    typer.echo(__version__)


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Argument(help="Destination TOML config path."),
    ] = Path("examples/linear_tearing.toml"),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter linear-tearing TOML config."""
    if path.exists() and not force:
        raise typer.BadParameter(f"{path} already exists; pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RunConfig().to_toml(), encoding="utf-8")
    typer.echo(f"wrote {path}")


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Path to a TOML run config.")],
    outdir: Annotated[
        Path | None,
        typer.Option("--outdir", help="Override the output directory from the config."),
    ] = None,
) -> None:
    """Run the lightweight deterministic smoke workflow for a TOML config."""
    cfg = load_config(config)
    if outdir is not None:
        cfg = cfg.with_output_dir(outdir)

    run_dir = cfg.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory, diagnostics = run_linear_tearing_smoke(cfg)

    config_path = run_dir / "config_effective.json"
    diagnostics_path = run_dir / "diagnostics.json"
    trajectory_path = run_dir / "trajectory.npz"
    manifest_path = run_dir / "manifest.json"

    config_path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    diagnostics["grid_shape"] = list(cfg.mesh.shape)
    diagnostics["mesh_lower"] = list(cfg.mesh.lower)
    diagnostics["mesh_upper"] = list(cfg.mesh.upper)
    diagnostics["quantities"] = list(cfg.diagnostics.quantities)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    write_reduced_mhd_trajectory_npz(
        trajectory_path,
        trajectory=trajectory,
        config=cfg.to_dict(),
        diagnostics=diagnostics,
    )
    write_manifest(
        manifest_path,
        config=cfg.to_dict(),
        outputs={
            "config": str(config_path.name),
            "diagnostics": str(diagnostics_path.name),
            "trajectory": str(trajectory_path.name),
        },
    )
    typer.echo(f"wrote {manifest_path}")


@app.command()
def figures(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing trajectory.npz.")],
    outdir: Annotated[
        Path | None,
        typer.Option("--outdir", help="Figure output directory; defaults to <run>/figures."),
    ] = None,
    gif: Annotated[bool, typer.Option("--gif", help="Also write flux_movie.gif.")] = False,
) -> None:
    """Regenerate deterministic figures from a saved run directory."""
    trajectory, diagnostics = read_reduced_mhd_trajectory_npz(run_dir / "trajectory.npz")
    figure_dir = outdir or (run_dir / "figures")
    shape = tuple(int(value) for value in diagnostics["grid_shape"])
    lower = tuple(float(value) for value in diagnostics["mesh_lower"])
    upper = tuple(float(value) for value in diagnostics["mesh_upper"])
    grid = CartesianGrid(shape=shape, lower=lower, upper=upper)
    final_state = ReducedMHDState(
        psi=trajectory.states.psi[-1],
        omega=trajectory.states.omega[-1],
    )
    energy_path = plot_energy_history(
        trajectory,
        lengths=grid.lengths,
        path=figure_dir / "energy_history.png",
    )
    flux_path = plot_flux_contours(final_state, path=figure_dir / "flux_final.png")
    mode = tuple(int(value) for value in diagnostics["diagnostic_mode"])
    amplitude_path = plot_mode_amplitude(
        trajectory,
        mode=mode,
        path=figure_dir / "mode_amplitude.png",
    )
    typer.echo(f"wrote {energy_path}")
    typer.echo(f"wrote {flux_path}")
    typer.echo(f"wrote {amplitude_path}")
    if gif:
        gif_path = plot_flux_gif(trajectory, path=figure_dir / "flux_movie.gif")
        typer.echo(f"wrote {gif_path}")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing manifest outputs.")],
) -> None:
    """Write a JSON and Markdown benchmark report for a run directory."""
    json_path, markdown_path = write_run_report(run_dir)
    typer.echo(f"wrote {json_path}")
    typer.echo(f"wrote {markdown_path}")


def main() -> None:  # pragma: no cover - exercised by console entry points.
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
