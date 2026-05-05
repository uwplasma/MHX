"""MHX command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mhx._version import __version__
from mhx.benchmarks import run_linear_tearing_smoke, validate_run, write_run_report
from mhx.config import RunConfig, load_config
from mhx.grids import CartesianGrid
from mhx.io import (
    read_reduced_mhd_trajectory_npz,
    write_manifest,
    write_reduced_mhd_trajectory_npz,
)
from mhx.physics import PHYSICS_API_VERSION, default_physics_registry
from mhx.plotting import (
    plot_energy_history,
    plot_flux_contours,
    plot_flux_gif,
    plot_mode_amplitude,
)
from mhx.state import ReducedMHDState

app = typer.Typer(no_args_is_help=True, help="MHX differentiable MHD workflows.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Benchmark workflows.")
physics_app = typer.Typer(no_args_is_help=True, help="Physics plugin inspection.")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(physics_app, name="physics")


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
    manifest_path = _run_config(config, outdir=outdir)
    typer.echo(f"wrote {manifest_path}")


def _run_config(config: Path, *, outdir: Path | None = None) -> Path:
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
    return manifest_path


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
    x, y = grid.axes()
    final_state = ReducedMHDState(
        psi=trajectory.states.psi[-1],
        omega=trajectory.states.omega[-1],
    )
    energy_path = plot_energy_history(
        trajectory,
        lengths=grid.lengths,
        path=figure_dir / "energy_history.png",
    )
    flux_path = plot_flux_contours(final_state, path=figure_dir / "flux_final.png", x=x, y=y)
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
        extent = (lower[0], upper[0], lower[1], upper[1])
        gif_path = plot_flux_gif(trajectory, path=figure_dir / "flux_movie.gif", extent=extent)
        typer.echo(f"wrote {gif_path}")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing manifest outputs.")],
) -> None:
    """Write a JSON and Markdown benchmark report for a run directory."""
    json_path, markdown_path = write_run_report(run_dir)
    typer.echo(f"wrote {json_path}")
    typer.echo(f"wrote {markdown_path}")


@benchmark_app.command("run")
def benchmark_run(
    config: Annotated[
        Path,
        typer.Option("--config", help="Benchmark TOML config."),
    ] = Path("examples/linear_tearing.toml"),
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Benchmark output directory."),
    ] = Path("outputs/benchmarks/linear_tearing_fast"),
    figures_enabled: Annotated[bool, typer.Option("--figures/--no-figures")] = True,
    gif: Annotated[bool, typer.Option("--gif", help="Write flux_movie.gif when plotting.")] = False,
    report_enabled: Annotated[bool, typer.Option("--report/--no-report")] = True,
) -> None:
    """Run the FAST reduced-MHD benchmark pipeline."""
    manifest_path = _run_config(config, outdir=outdir)
    typer.echo(f"wrote {manifest_path}")
    if figures_enabled:
        figures(outdir, gif=gif)
    if report_enabled:
        report(outdir)


@benchmark_app.command("validate")
def benchmark_validate(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to validate.")],
    max_relative_energy_growth: Annotated[
        float,
        typer.Option("--max-relative-energy-growth", help="Allowed relative total-energy growth."),
    ] = 1.0e-10,
) -> None:
    """Validate a completed FAST benchmark run."""
    output_path, result = validate_run(
        run_dir,
        max_relative_energy_growth=max_relative_energy_growth,
    )
    typer.echo(f"wrote {output_path}")
    if not result["passed"]:
        raise typer.Exit(code=1)


@physics_app.command("list")
def physics_list() -> None:
    """List registered built-in physics terms."""
    typer.echo(f"Physics API: {PHYSICS_API_VERSION}")
    for item in default_physics_registry().metadata():
        typer.echo(f"- {item.name}: {item.description}")


@physics_app.command("lint")
def physics_lint(
    name: Annotated[str, typer.Argument(help="Registered physics term name.")],
) -> None:
    """Validate a registered physics term's API metadata."""
    registry = default_physics_registry()
    term = registry.create(name)
    if term.api_version != PHYSICS_API_VERSION:
        raise typer.BadParameter(
            f"{name!r} uses {term.api_version!r}, expected {PHYSICS_API_VERSION!r}"
        )
    typer.echo(f"{name}: ok ({term.api_version})")


def main() -> None:  # pragma: no cover - exercised by console entry points.
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
