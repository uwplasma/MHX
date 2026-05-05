"""MHX command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mhx._version import __version__
from mhx.benchmarks import (
    run_linear_tearing_smoke,
    validate_run,
    write_arnoldi_validation,
    write_benchmark_catalog,
    write_cosine_equilibrium_linearization_validation,
    write_diffusion_eigenvalue_validation,
    write_fkr_window_validation,
    write_linearized_rhs_validation,
    write_power_iteration_validation,
    write_reconnection_scaling_validation,
    write_reduced_mhd_linear_eigenmode_validation,
    write_resistive_decay_validation,
    write_run_report,
    write_timing_benchmark,
)
from mhx.config import RunConfig, load_config
from mhx.diagnostics import (
    DIAGNOSTICS_ENTRY_POINT_GROUP,
    default_diagnostics_registry,
    load_diagnostics_entry_points,
    load_diagnostics_plugin_modules,
)
from mhx.grids import CartesianGrid
from mhx.io import (
    read_reduced_mhd_trajectory_npz,
    write_artifact_manifest,
    write_manifest,
    write_reduced_mhd_trajectory_npz,
)
from mhx.physics import (
    PHYSICS_API_VERSION,
    PHYSICS_ENTRY_POINT_GROUP,
    default_equilibrium_registry,
    default_physics_registry,
    load_physics_entry_points,
    load_physics_plugin_modules,
)
from mhx.plotting import (
    plot_energy_history,
    plot_flux_contours,
    plot_flux_gif,
    plot_mode_amplitude,
    write_diagnostic_figures_for_run,
)
from mhx.state import ReducedMHDState

app = typer.Typer(no_args_is_help=True, help="MHX differentiable MHD workflows.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Benchmark workflows.")
physics_app = typer.Typer(no_args_is_help=True, help="Physics plugin inspection.")
diagnostics_app = typer.Typer(no_args_is_help=True, help="Diagnostic registry inspection.")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(physics_app, name="physics")
app.add_typer(diagnostics_app, name="diagnostics")


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
    diagnostic_figures, diagnostic_warnings = write_diagnostic_figures_for_run(
        run_dir,
        figure_dir=figure_dir / "diagnostics",
    )
    for item in diagnostic_figures:
        typer.echo(f"wrote diagnostic figure {item['path']}")
    for warning in diagnostic_warnings:
        typer.echo(f"warning: {warning}", err=True)
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


@app.command()
def artifact_manifest(
    root: Annotated[Path, typer.Argument(help="Directory whose files should be hashed.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Manifest path; defaults to <root>/artifact_manifest.json."),
    ] = None,
) -> None:
    """Write a recursive SHA-256 artifact manifest for a directory."""
    manifest = write_artifact_manifest(root, path=output)
    output_path = output or (root / "artifact_manifest.json")
    typer.echo(f"wrote {output_path} ({len(manifest['files'])} files)")


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


@benchmark_app.command("catalog")
def benchmark_catalog(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for validation catalog artifacts."),
    ] = Path("outputs/benchmarks/catalog"),
) -> None:
    """Write a reviewer-facing validation benchmark catalog."""
    json_path, markdown_path = write_benchmark_catalog(outdir)
    typer.echo(f"wrote {json_path}")
    typer.echo(f"wrote {markdown_path}")


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


@benchmark_app.command("decay")
def benchmark_decay(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for exact decay artifacts."),
    ] = Path("outputs/benchmarks/resistive_decay"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 32,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 32,
    mode_x: Annotated[int, typer.Option("--mode-x", help="Fourier mode in x.")] = 1,
    mode_y: Annotated[int, typer.Option("--mode-y", help="Fourier mode in y.")] = 0,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 5.0e-2,
    t1: Annotated[float, typer.Option("--t1", help="Final time.")] = 1.0,
    dt: Annotated[float, typer.Option("--dt", help="Fixed RK4 step.")] = 1.0e-2,
) -> None:
    """Run the exact single-mode resistive-decay validation benchmark."""
    manifest_path, validation = write_resistive_decay_validation(
        outdir,
        shape=(nx, ny),
        mode=(mode_x, mode_y),
        resistivity=eta,
        t1=t1,
        dt=dt,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("scaling")
def benchmark_scaling(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for analytic scaling artifacts."),
    ] = Path("outputs/benchmarks/reconnection_scaling"),
    ka: Annotated[float, typer.Option("--ka", help="FKR constant-psi wavenumber ka.")] = 0.5,
) -> None:
    """Run analytic reconnection scaling gates for FKR/plasmoid/ideal-tearing theory."""
    manifest_path, validation = write_reconnection_scaling_validation(outdir, ka=ka)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("fkr-window")
def benchmark_fkr_window(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for FKR regime-window artifacts."),
    ] = Path("outputs/benchmarks/fkr_window"),
    lundquist: Annotated[
        float,
        typer.Option("--lundquist", help="Local Lundquist number S_a."),
    ] = 1.0e6,
) -> None:
    """Run the analytic FKR constant-psi regime-window gate."""
    manifest_path, validation = write_fkr_window_validation(outdir, lundquist=lundquist)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("linearized-rhs")
def benchmark_linearized_rhs(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for linearized-RHS artifacts."),
    ] = Path("outputs/benchmarks/linearized_rhs"),
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", help="Centered finite-difference perturbation size."),
    ] = 1.0e-3,
) -> None:
    """Run the matrix-free reduced-MHD linearized-RHS consistency gate."""
    manifest_path, validation = write_linearized_rhs_validation(outdir, epsilon=epsilon)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("reduced-mhd-eigenmode")
def benchmark_reduced_mhd_eigenmode(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for reduced-MHD linear eigenmode artifacts.",
        ),
    ] = Path("outputs/benchmarks/reduced_mhd_eigenmode"),
) -> None:
    """Run the zero-state reduced-MHD linear diffusion eigenmode gate."""
    manifest_path, validation = write_reduced_mhd_linear_eigenmode_validation(outdir)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("cosine-equilibrium-linearization")
def benchmark_cosine_equilibrium_linearization(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for nonzero-equilibrium linearization artifacts.",
        ),
    ] = Path("outputs/benchmarks/cosine_equilibrium_linearization"),
) -> None:
    """Run the analytic nonzero-cosine-equilibrium linearized-RHS gate."""
    manifest_path, validation = write_cosine_equilibrium_linearization_validation(outdir)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("diffusion-eigenvalue")
def benchmark_diffusion_eigenvalue(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for diffusion eigenvalue artifacts."),
    ] = Path("outputs/benchmarks/diffusion_eigenvalue"),
) -> None:
    """Run the matrix-free spectral diffusion eigenvalue gate."""
    manifest_path, validation = write_diffusion_eigenvalue_validation(outdir)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("power-iteration")
def benchmark_power_iteration(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for power-iteration artifacts."),
    ] = Path("outputs/benchmarks/power_iteration"),
    iterations: Annotated[
        int,
        typer.Option("--iterations", help="Power-iteration steps."),
    ] = 30,
) -> None:
    """Run the known-operator power-iteration smoke benchmark."""
    manifest_path, validation = write_power_iteration_validation(outdir, iterations=iterations)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("arnoldi")
def benchmark_arnoldi(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for Arnoldi artifacts."),
    ] = Path("outputs/benchmarks/arnoldi"),
) -> None:
    """Run the known-operator Arnoldi Ritz-spectrum smoke benchmark."""
    manifest_path, validation = write_arnoldi_validation(outdir)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("timing")
def benchmark_timing(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for FAST timing artifacts."),
    ] = Path("outputs/benchmarks/timing"),
    repeats: Annotated[int, typer.Option("--repeats", help="Measured repeats per case.")] = 3,
    warmups: Annotated[int, typer.Option("--warmups", help="Unmeasured warmup repeats.")] = 1,
) -> None:
    """Run FAST benchmark timing and Python-memory measurements."""
    manifest_path, _ = write_timing_benchmark(outdir, repeats=repeats, warmups=warmups)
    typer.echo(f"wrote {manifest_path}")


@physics_app.command("list")
def physics_list() -> None:
    """List registered built-in physics terms."""
    typer.echo(f"Physics API: {PHYSICS_API_VERSION}")
    for item in default_physics_registry().metadata():
        typer.echo(f"- {item.name}: {item.description}")


@physics_app.command("list-with-plugins")
def physics_list_with_plugins(
    plugin_module: Annotated[
        list[str] | None,
        typer.Option("--plugin-module", help="Import module exposing register_physics(registry)."),
    ] = None,
    entry_point_group: Annotated[
        list[str] | None,
        typer.Option(
            "--entry-point-group",
            help="Load installed plugin entry points from a group; "
            f"use {PHYSICS_ENTRY_POINT_GROUP!r} for standard physics plugins.",
        ),
    ] = None,
) -> None:
    """List physics terms after loading optional user plugin modules."""
    registry = default_physics_registry()
    load_physics_entry_points(registry, tuple(entry_point_group or ()))
    load_physics_plugin_modules(registry, tuple(plugin_module or ()))
    typer.echo(f"Physics API: {PHYSICS_API_VERSION}")
    for item in registry.metadata():
        typer.echo(f"- {item.name}: {item.description}")


@physics_app.command("equilibria")
def physics_equilibria() -> None:
    """List registered reduced-MHD equilibria."""
    for item in default_equilibrium_registry().metadata():
        typer.echo(f"- {item.name}: {item.description}")


@physics_app.command("lint")
def physics_lint(
    name: Annotated[str, typer.Argument(help="Registered physics term name.")],
    plugin_module: Annotated[
        list[str] | None,
        typer.Option("--plugin-module", help="Import module exposing register_physics(registry)."),
    ] = None,
    entry_point_group: Annotated[
        list[str] | None,
        typer.Option("--entry-point-group", help="Installed plugin entry-point group."),
    ] = None,
) -> None:
    """Validate a registered physics term's API metadata."""
    registry = default_physics_registry()
    load_physics_entry_points(registry, tuple(entry_point_group or ()))
    load_physics_plugin_modules(registry, tuple(plugin_module or ()))
    term = registry.create(name)
    if term.api_version != PHYSICS_API_VERSION:
        raise typer.BadParameter(
            f"{name!r} uses {term.api_version!r}, expected {PHYSICS_API_VERSION!r}"
        )
    typer.echo(f"{name}: ok ({term.api_version})")


@diagnostics_app.command("list")
def diagnostics_list() -> None:
    """List registered reduced-MHD diagnostics and output keys."""
    for item in default_diagnostics_registry().metadata():
        keys = ", ".join(item["output_keys"])
        typer.echo(f"- {item['name']}: {item['description']} [{keys}]")


@diagnostics_app.command("list-with-plugins")
def diagnostics_list_with_plugins(
    plugin_module: Annotated[
        list[str] | None,
        typer.Option(
            "--plugin-module",
            help="Import module exposing register_diagnostics(registry).",
        ),
    ] = None,
    entry_point_group: Annotated[
        list[str] | None,
        typer.Option(
            "--entry-point-group",
            help="Load installed plugin entry points from a group; "
            f"use {DIAGNOSTICS_ENTRY_POINT_GROUP!r} for standard diagnostic plugins.",
        ),
    ] = None,
) -> None:
    """List diagnostics after loading optional user plugin modules."""
    registry = default_diagnostics_registry()
    load_diagnostics_entry_points(registry, tuple(entry_point_group or ()))
    load_diagnostics_plugin_modules(registry, tuple(plugin_module or ()))
    for item in registry.metadata():
        keys = ", ".join(item["output_keys"])
        typer.echo(f"- {item['name']}: {item['description']} [{keys}]")


@diagnostics_app.command("lint")
def diagnostics_lint(
    name: Annotated[str, typer.Argument(help="Registered diagnostic name.")],
    plugin_module: Annotated[
        list[str] | None,
        typer.Option(
            "--plugin-module",
            help="Import module exposing register_diagnostics(registry).",
        ),
    ] = None,
    entry_point_group: Annotated[
        list[str] | None,
        typer.Option("--entry-point-group", help="Installed diagnostic entry-point group."),
    ] = None,
) -> None:
    """Validate a registered diagnostic's metadata contract."""
    registry = default_diagnostics_registry()
    load_diagnostics_entry_points(registry, tuple(entry_point_group or ()))
    load_diagnostics_plugin_modules(registry, tuple(plugin_module or ()))
    spec = registry.get(name)
    if not spec.output_keys:
        raise typer.BadParameter(f"{name!r} must declare at least one output key")
    typer.echo(f"{name}: ok ({len(spec.output_keys)} output keys)")


def main() -> None:  # pragma: no cover - exercised by console entry points.
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
