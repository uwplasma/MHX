"""MHX command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mhx._version import __version__
from mhx.config import RunConfig, load_config
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics.spectral import fft_derivative

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
    grid = CartesianGrid.from_mesh_config(cfg.mesh)
    length = grid.lengths[0]
    field = grid.sinusoid(mode=(1, 0))
    dfdx = fft_derivative(field, axis=0, length=length)
    expected = (2.0 * 3.141592653589793 / length) * grid.cosinusoid(mode=(1, 0))
    max_error = float(abs(dfdx - expected).max())

    config_path = run_dir / "config_effective.json"
    diagnostics_path = run_dir / "diagnostics.json"
    manifest_path = run_dir / "manifest.json"

    config_path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    diagnostics = {
        "grid_shape": list(grid.shape),
        "cell_volume": grid.cell_volume,
        "spectral_smoke_max_error": max_error,
        "quantities": list(cfg.diagnostics.quantities),
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        manifest_path,
        config=cfg.to_dict(),
        outputs={
            "config": str(config_path.name),
            "diagnostics": str(diagnostics_path.name),
        },
    )
    typer.echo(f"wrote {manifest_path}")


def main() -> None:  # pragma: no cover - exercised by console entry points.
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
