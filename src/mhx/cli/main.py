"""MHX command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mhx._version import __version__
from mhx.benchmarks import (
    double_harris_seeded_long_run_presets,
    run_linear_tearing_smoke,
    validate_run,
    write_arnoldi_validation,
    write_cosine_equilibrium_linearization_validation,
    write_decaying_mhd_turbulence_validation,
    write_diffusion_eigenvalue_validation,
    write_fkr_growth_rate_validation,
    write_fkr_window_validation,
    write_forced_turbulent_reconnection_readiness_report,
    write_forced_turbulent_reconnection_validation,
    write_harris_delta_prime_validation,
    write_linear_tearing_dispersion_validation,
    write_linear_tearing_eigenvalue_validation,
    write_linear_tearing_layer_validation,
    write_linear_tearing_timedomain_validation,
    write_linearized_rhs_validation,
    write_nonlinear_energy_budget_validation,
    write_orszag_tang_vortex_validation,
    write_periodic_current_sheet_eigenvalue_validation,
    write_periodic_current_sheet_nonlinear_bridge_validation,
    write_periodic_current_sheet_timedomain_validation,
    write_periodic_double_harris_convergence_validation,
    write_periodic_double_harris_nonlinear_growth_validation,
    write_periodic_double_harris_parameter_sweep_validation,
    write_periodic_double_harris_promotion_report,
    write_periodic_double_harris_seeded_long_run_validation,
    write_power_iteration_validation,
    write_reconnection_scaling_validation,
    write_reduced_mhd_linear_eigenmode_validation,
    write_resistive_decay_validation,
    write_run_report,
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
from mhx.runtime import configure_jax
from mhx.state import ReducedMHDState
from mhx.versioning import api_version_info

app = typer.Typer(no_args_is_help=True, help="MHX differentiable MHD workflows.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Benchmark workflows.")
api_app = typer.Typer(no_args_is_help=True, help="Public API and schema metadata.")
physics_app = typer.Typer(no_args_is_help=True, help="Physics plugin inspection.")
diagnostics_app = typer.Typer(no_args_is_help=True, help="Diagnostic registry inspection.")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(api_app, name="api")
app.add_typer(physics_app, name="physics")
app.add_typer(diagnostics_app, name="diagnostics")


def _exit_if_validation_failed(validation: dict[str, object], *, context: str) -> None:
    """Emit failed validation checks before returning a nonzero CLI status."""
    if bool(validation.get("passed")):
        return
    checks = validation.get("checks", {})
    if isinstance(checks, dict):
        failed = sorted(str(key) for key, value in checks.items() if not bool(value))
        if failed:
            typer.secho(
                f"{context} failed validation checks: {', '.join(failed)}",
                fg=typer.colors.RED,
                err=True,
            )
    diagnostics = validation.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        for key in (
            "max_relative_energy_growth",
            "final_magnetic_divergence_linf",
            "steps_run",
            "dt",
            "shape",
        ):
            if key in diagnostics:
                typer.echo(f"{key}: {diagnostics[key]}", err=True)
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the MHX package version."""
    typer.echo(__version__)


@api_app.command("status")
def api_status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Print active public API and artifact-schema versions."""
    info = api_version_info().to_dict()
    if json_output:
        typer.echo(json.dumps(info, indent=2, sort_keys=True))
        return
    typer.echo(f"Package version: {info['package_version']}")
    typer.echo(f"Public API: {info['public_api_version']}")
    typer.echo(f"Supported APIs: {', '.join(info['supported_api_versions'])}")
    typer.echo(f"Physics API: {info['physics_api_version']}")
    typer.echo(f"Diagnostics API: {info['diagnostics_api_version']}")
    typer.echo(f"Trajectory schema: {info['trajectory_schema']}")
    typer.echo(f"Manifest schema: {info['manifest_schema']}")
    typer.echo(f"Artifact manifest schema: {info['artifact_manifest_schema']}")
    typer.echo(f"Claim levels: {', '.join(info['claim_levels'])}")



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
    configure_jax(enable_x64=cfg.numerics.enable_x64)
    if outdir is not None:
        cfg = cfg.with_output_dir(outdir)
    if cfg.physics.model != "reduced_mhd_linear_tearing":
        raise typer.BadParameter(
            "mhx run currently supports physics.model='reduced_mhd_linear_tearing'."
        )

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
        claim_level="smoke",
        claim_scope="Lightweight reduced-MHD run; not a production reconnection claim.",
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
    _configure_validation_precision()
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
    _configure_validation_precision()
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
    _configure_validation_precision()
    manifest_path, validation = write_fkr_window_validation(outdir, lundquist=lundquist)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("fkr-growth")
def benchmark_fkr_growth(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for FKR growth-rate artifacts."),
    ] = Path("outputs/benchmarks/fkr_growth_rate"),
    fixed_ka: Annotated[
        float,
        typer.Option("--fixed-ka", help="ka used for the Lundquist-number scan."),
    ] = 0.35,
    fixed_lundquist: Annotated[
        float,
        typer.Option("--fixed-lundquist", help="S_a used for the Delta-prime scan."),
    ] = 1.0e6,
) -> None:
    """Run the calibrated asymptotic FKR growth-rate gate."""
    _configure_validation_precision()
    manifest_path, validation = write_fkr_growth_rate_validation(
        outdir,
        fixed_ka=fixed_ka,
        fixed_lundquist=fixed_lundquist,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("harris-delta-prime")
def benchmark_harris_delta_prime(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for Harris Delta-prime artifacts."),
    ] = Path("outputs/benchmarks/harris_delta_prime"),
    steps: Annotated[int, typer.Option("--steps", help="Backward RK4 outer-ODE steps.")] = 4000,
    xmax_over_a: Annotated[
        float,
        typer.Option("--xmax-over-a", help="Positive integration boundary in sheet widths."),
    ] = 18.0,
) -> None:
    """Run the numerical Harris-sheet outer-region Delta-prime gate."""
    _configure_validation_precision()
    manifest_path, validation = write_harris_delta_prime_validation(
        outdir,
        steps=steps,
        xmax_over_a=xmax_over_a,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("linear-tearing-eigenvalue")
def benchmark_linear_tearing_eigenvalue(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for direct Harris-sheet tearing eigenvalue artifacts.",
        ),
    ] = Path("outputs/benchmarks/linear_tearing_eigenvalue"),
    grid_points: Annotated[
        str,
        typer.Option(
            "--grid-points",
            help="Comma-separated interior grid counts for the second-order convergence scan.",
        ),
    ] = "192,256,320",
    half_width: Annotated[
        float,
        typer.Option("--half-width", help="Half-domain width d/a for x in [-d,d]."),
    ] = 10.0,
    lundquist: Annotated[
        float,
        typer.Option("--lundquist", help="Magnetic Lundquist number S."),
    ] = 1000.0,
    wavenumber: Annotated[
        float,
        typer.Option("--wavenumber", help="Tearing perturbation wavenumber k a."),
    ] = 0.5,
) -> None:
    """Run the direct reduced-MHD Harris-sheet tearing eigenvalue gate."""
    _configure_validation_precision()
    manifest_path, validation = write_linear_tearing_eigenvalue_validation(
        outdir,
        grid_points=_parse_int_tuple(grid_points),
        half_width=half_width,
        lundquist=lundquist,
        wavenumber=wavenumber,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("linear-tearing-dispersion")
def benchmark_linear_tearing_dispersion(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for Harris-sheet tearing dispersion artifacts.",
        ),
    ] = Path("outputs/benchmarks/linear_tearing_dispersion"),
    grid_points: Annotated[
        int,
        typer.Option("--grid-points", help="Interior grid count for each wavenumber sample."),
    ] = 192,
    wavenumber: Annotated[
        str,
        typer.Option(
            "--wavenumber",
            help="Comma-separated ka samples including values below and above ka=1.",
        ),
    ] = "0.3,0.5,0.7,0.9,1.1,1.2",
) -> None:
    """Run a finite-domain Harris-sheet tearing dispersion gate."""
    _configure_validation_precision()
    manifest_path, validation = write_linear_tearing_dispersion_validation(
        outdir,
        grid_points=grid_points,
        wavenumber=_parse_float_tuple(wavenumber),
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("linear-tearing-layer")
def benchmark_linear_tearing_layer(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for Harris-sheet eigenfunction-layer artifacts.",
        ),
    ] = Path("outputs/benchmarks/linear_tearing_layer"),
    grid_points: Annotated[
        int,
        typer.Option("--grid-points", help="Interior grid count for each S sample."),
    ] = 192,
    lundquist: Annotated[
        str,
        typer.Option("--lundquist", help="Comma-separated Lundquist-number samples."),
    ] = "250,500,1000,2000",
) -> None:
    """Run a Harris-sheet tearing eigenfunction localization gate."""
    _configure_validation_precision()
    manifest_path, validation = write_linear_tearing_layer_validation(
        outdir,
        grid_points=grid_points,
        lundquist=_parse_float_tuple(lundquist),
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("linear-tearing-timedomain")
def benchmark_linear_tearing_timedomain(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for Harris-sheet time-domain replay artifacts.",
        ),
    ] = Path("outputs/benchmarks/linear_tearing_timedomain"),
    grid_points: Annotated[
        int,
        typer.Option("--grid-points", help="Interior grid count for the eigenmode replay."),
    ] = 192,
    dt: Annotated[
        float,
        typer.Option("--dt", help="RK4 time step for the linear replay."),
    ] = 0.25,
    t_end: Annotated[
        float,
        typer.Option("--t-end", help="Final replay time."),
    ] = 80.0,
) -> None:
    """Replay a direct Harris-sheet eigenmode in time and refit its growth rate."""
    _configure_validation_precision()
    manifest_path, validation = write_linear_tearing_timedomain_validation(
        outdir,
        grid_points=grid_points,
        dt=dt,
        t_end=t_end,
    )
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
    _configure_validation_precision()
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
    _configure_validation_precision()
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
    _configure_validation_precision()
    manifest_path, validation = write_cosine_equilibrium_linearization_validation(outdir)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("current-sheet-eigenvalue")
def benchmark_current_sheet_eigenvalue(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for periodic current-sheet eigenvalue artifacts.",
        ),
    ] = Path("outputs/benchmarks/periodic_current_sheet_eigenvalue"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 8,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 8,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 2.0e-2,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 2.0e-2,
) -> None:
    """Run the dense tiny-grid periodic current-sheet eigenvalue gate."""
    _configure_validation_precision()
    manifest_path, validation = write_periodic_current_sheet_eigenvalue_validation(
        outdir,
        shape=(nx, ny),
        resistivity=eta,
        viscosity=nu,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("current-sheet-timedomain")
def benchmark_current_sheet_timedomain(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for periodic current-sheet time-domain artifacts.",
        ),
    ] = Path("outputs/benchmarks/periodic_current_sheet_timedomain"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 8,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 8,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 2.0e-2,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 2.0e-2,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 5.0e-2,
    t_end: Annotated[float, typer.Option("--t-end", help="Final replay time.")] = 5.0,
) -> None:
    """Replay a decaying periodic current-sheet JVP eigenmode in time."""
    _configure_validation_precision()
    manifest_path, validation = write_periodic_current_sheet_timedomain_validation(
        outdir,
        shape=(nx, ny),
        resistivity=eta,
        viscosity=nu,
        dt=dt,
        t_end=t_end,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("current-sheet-nonlinear-bridge")
def benchmark_current_sheet_nonlinear_bridge(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for nonlinear current-sheet bridge artifacts.",
        ),
    ] = Path("outputs/benchmarks/periodic_current_sheet_nonlinear_bridge"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 8,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 8,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 2.0e-2,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 2.0e-2,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 2.0e-2,
    steps: Annotated[int, typer.Option("--steps", help="RK4 steps.")] = 8,
) -> None:
    """Validate nonlinear RK4 trajectory-map JVP convergence."""
    _configure_validation_precision()
    manifest_path, validation = write_periodic_current_sheet_nonlinear_bridge_validation(
        outdir,
        shape=(nx, ny),
        resistivity=eta,
        viscosity=nu,
        dt=dt,
        steps=steps,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("double-harris-growth")
def benchmark_double_harris_growth(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for unstable double-Harris nonlinear-growth artifacts.",
        ),
    ] = Path("outputs/benchmarks/periodic_double_harris_nonlinear_growth"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 8,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 8,
    width: Annotated[float, typer.Option("--width", help="Current-sheet half-width proxy.")] = 0.4,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 5.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 5.0e-3,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    t_end: Annotated[float, typer.Option("--t-end", help="Final nonlinear time.")] = 4.0,
    save_every: Annotated[int, typer.Option("--save-every", help="Saved-step stride.")] = 20,
) -> None:
    """Validate unstable nonlinear growth of a periodic double-Harris current sheet."""
    _configure_validation_precision()
    manifest_path, validation = write_periodic_double_harris_nonlinear_growth_validation(
        outdir,
        shape=(nx, ny),
        width=width,
        resistivity=eta,
        viscosity=nu,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("orszag-tang")
def benchmark_orszag_tang(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for reduced-MHD Orszag-Tang nonlinear artifacts.",
        ),
    ] = Path("outputs/benchmarks/orszag_tang_vortex"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 24,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 24,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 1.0e-2,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 1.0e-2,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 5.0e-3,
    t_end: Annotated[float, typer.Option("--t-end", help="Final nonlinear time.")] = 2.0,
    save_every: Annotated[int, typer.Option("--save-every", help="Saved-step stride.")] = 20,
    movies: Annotated[
        bool,
        typer.Option("--movies/--no-movies", help="Write flux/current/vorticity GIFs."),
    ] = False,
) -> None:
    """Run the incompressible reduced-MHD Orszag-Tang vortex validation."""
    _configure_validation_precision()
    manifest_path, validation = write_orszag_tang_vortex_validation(
        outdir,
        shape=(nx, ny),
        resistivity=eta,
        viscosity=nu,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        movies=movies,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("decaying-turbulence")
def benchmark_decaying_turbulence(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for decaying reduced-MHD turbulence artifacts.",
        ),
    ] = Path("outputs/benchmarks/decaying_mhd_turbulence"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 32,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 32,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 2.0e-2,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 2.0e-2,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    t_end: Annotated[float, typer.Option("--t-end", help="Final nonlinear time.")] = 4.0,
    save_every: Annotated[int, typer.Option("--save-every", help="Saved-step stride.")] = 20,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic spectral seed.")] = 7,
    movies: Annotated[
        bool,
        typer.Option("--movies/--no-movies", help="Write current/flux GIFs."),
    ] = False,
) -> None:
    """Run deterministic decaying 2-D reduced-MHD turbulence validation."""
    _configure_validation_precision()
    manifest_path, validation = write_decaying_mhd_turbulence_validation(
        outdir,
        shape=(nx, ny),
        resistivity=eta,
        viscosity=nu,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        seed=seed,
        movies=movies,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("forced-turbulent-reconnection")
def benchmark_forced_turbulent_reconnection(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for forced turbulent reconnection artifacts.",
        ),
    ] = Path("outputs/benchmarks/forced_turbulent_reconnection"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 32,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 32,
    width: Annotated[float, typer.Option("--width", help="Current-sheet half-width.")] = 0.32,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 1.5e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 1.5e-3,
    perturbation_amplitude: Annotated[
        float,
        typer.Option("--perturbation-amplitude", help="Coherent tearing seed amplitude."),
    ] = 1.0e-2,
    turbulent_flux_amplitude: Annotated[
        float,
        typer.Option("--turbulent-flux-amplitude", help="Broadband flux perturbation amplitude."),
    ] = 1.5e-2,
    turbulent_flow_amplitude: Annotated[
        float,
        typer.Option("--turbulent-flow-amplitude", help="Broadband flow perturbation amplitude."),
    ] = 1.5e-2,
    forcing_amplitude: Annotated[
        float,
        typer.Option("--forcing-amplitude", help="Static large-scale vorticity forcing amplitude."),
    ] = 2.0e-3,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 2.0e-2,
    t_end: Annotated[float, typer.Option("--t-end", help="Final nonlinear time.")] = 20.0,
    save_every: Annotated[int, typer.Option("--save-every", help="Saved-step stride.")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic spectral seed.")] = 11,
    max_relative_energy_growth: Annotated[
        float,
        typer.Option(
            "--max-relative-energy-growth",
            help="Allowed forced-run relative total-energy growth.",
        ),
    ] = 2.0,
    movies: Annotated[
        bool,
        typer.Option("--movies/--no-movies", help="Write current/flux GIFs."),
    ] = False,
) -> None:
    """Run a pedagogical forced-turbulence current-sheet validation replay."""
    _configure_validation_precision()
    manifest_path, validation = write_forced_turbulent_reconnection_validation(
        outdir,
        shape=(nx, ny),
        width=width,
        resistivity=eta,
        viscosity=nu,
        perturbation_amplitude=perturbation_amplitude,
        turbulent_flux_amplitude=turbulent_flux_amplitude,
        turbulent_flow_amplitude=turbulent_flow_amplitude,
        forcing_amplitude=forcing_amplitude,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        seed=seed,
        max_relative_energy_growth=max_relative_energy_growth,
        movies=movies,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("forced-turbulent-reconnection-readiness-check")
def benchmark_forced_turbulent_reconnection_readiness_check(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Forced turbulent reconnection artifact directory."),
    ],
    outdir: Annotated[
        Path | None,
        typer.Option(
            "--outdir",
            help="Readiness-report directory; defaults to <run-dir>/readiness.",
        ),
    ] = None,
    require_summary: Annotated[
        bool,
        typer.Option(
            "--require-summary/--no-require-summary",
            help="Require the summary PNG in the run bundle.",
        ),
    ] = True,
    require_movies: Annotated[
        bool,
        typer.Option(
            "--require-movies/--no-require-movies",
            help="Require current/flux GIFs in the run bundle.",
        ),
    ] = False,
    min_history_samples: Annotated[
        int,
        typer.Option("--min-history-samples", help="Minimum saved history samples."),
    ] = 20,
    min_t_end: Annotated[
        float,
        typer.Option("--min-t-end", help="Minimum terminal time for readiness."),
    ] = 20.0,
    min_reconnection_proxy_change: Annotated[
        float,
        typer.Option(
            "--min-reconnection-proxy-change",
            help="Minimum reconnecting-flux proxy range unless rate threshold passes.",
        ),
    ] = 1.0e-3,
    min_abs_reconnection_rate_proxy: Annotated[
        float,
        typer.Option(
            "--min-abs-reconnection-rate-proxy",
            help="Minimum absolute reconnection-rate proxy unless flux-change threshold passes.",
        ),
    ] = 1.0e-6,
    max_relative_energy_growth: Annotated[
        float,
        typer.Option(
            "--max-relative-energy-growth",
            help="Maximum relative total-energy growth allowed.",
        ),
    ] = 2.0,
) -> None:
    """Write a validation-only readiness report for forced turbulent reconnection."""
    manifest_path, validation = write_forced_turbulent_reconnection_readiness_report(
        run_dir,
        outdir=outdir,
        require_summary=require_summary,
        require_movies=require_movies,
        min_history_samples=min_history_samples,
        min_t_end=min_t_end,
        min_reconnection_proxy_change=min_reconnection_proxy_change,
        min_abs_reconnection_rate_proxy=min_abs_reconnection_rate_proxy,
        max_relative_energy_growth=max_relative_energy_growth,
    )
    typer.echo(f"wrote {manifest_path}")
    _exit_if_validation_failed(validation, context="forced-turbulent-reconnection-readiness-check")


@benchmark_app.command("double-harris-long-run")
def benchmark_double_harris_long_run(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for seeded double-Harris long-run artifacts.",
        ),
    ] = Path("outputs/benchmarks/periodic_double_harris_seeded_long_run"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 64,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 64,
    width: Annotated[float, typer.Option("--width", help="Current-sheet half-width proxy.")] = 0.4,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 5.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 5.0e-3,
    perturbation_amplitude: Annotated[
        float,
        typer.Option("--perturbation-amplitude", help="Seed flux perturbation amplitude."),
    ] = 1.0e-3,
    mode_x: Annotated[int, typer.Option("--mode-x", help="Seed Fourier mode in x.")] = 2,
    mode_y: Annotated[int, typer.Option("--mode-y", help="Seed Fourier mode in y.")] = 1,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    t_end: Annotated[float, typer.Option("--t-end", help="Final nonlinear time.")] = 100.0,
    save_every: Annotated[int, typer.Option("--save-every", help="Saved-step stride.")] = 200,
    fit_start: Annotated[float, typer.Option("--fit-start", help="Early fit window start.")] = 0.0,
    fit_stop: Annotated[float, typer.Option("--fit-stop", help="Early fit window stop.")] = 10.0,
    min_early_growth_rate: Annotated[
        float,
        typer.Option("--min-early-growth-rate", help="Minimum fitted early growth rate."),
    ] = 5.0e-2,
    min_max_growth_factor: Annotated[
        float,
        typer.Option("--min-max-growth-factor", help="Minimum maximum normalized growth."),
    ] = 2.0,
    min_reconnected_flux_amplification: Annotated[
        float,
        typer.Option(
            "--min-reconnected-flux-amplification",
            help="Minimum dominant reconnecting-flux proxy amplification.",
        ),
    ] = 1.05,
    min_island_width_amplification: Annotated[
        float,
        typer.Option(
            "--min-island-width-amplification",
            help="Minimum Rutherford island-width proxy amplification.",
        ),
    ] = 1.05,
    movies: Annotated[
        bool,
        typer.Option("--movies/--no-movies", help="Write fixed-scale flux/current GIFs."),
    ] = False,
    ci_fast: Annotated[
        bool,
        typer.Option(
            "--ci-fast/--validation-duration",
            help="Use the explicitly labeled bounded CI duration preset.",
        ),
    ] = False,
) -> None:
    """Run a scalable seeded periodic double-Harris nonlinear validation replay."""
    _configure_validation_precision()
    duration_label = None
    if ci_fast:
        preset = double_harris_seeded_long_run_presets()["ci_fast"]
        t_end = float(preset["t_end"])
        save_every = int(preset["save_every"])
        fit_start, fit_stop = tuple(preset["fit_window"])
        min_max_growth_factor = float(preset["min_max_growth_factor"])
        duration_label = str(preset["duration_label"])
    manifest_path, validation = write_periodic_double_harris_seeded_long_run_validation(
        outdir,
        shape=(nx, ny),
        width=width,
        resistivity=eta,
        viscosity=nu,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_mode=(mode_x, mode_y),
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        fit_window=(fit_start, fit_stop),
        duration_label=duration_label,
        min_early_growth_rate=min_early_growth_rate,
        min_max_growth_factor=min_max_growth_factor,
        min_reconnected_flux_amplification=min_reconnected_flux_amplification,
        min_island_width_amplification=min_island_width_amplification,
        movies=movies,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("double-harris-convergence")
def benchmark_double_harris_convergence(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for seeded double-Harris convergence artifacts.",
        ),
    ] = Path("outputs/benchmarks/periodic_double_harris_convergence"),
    resolutions: Annotated[
        str,
        typer.Option("--resolutions", help="Comma-separated grid sizes for resolution sweep."),
    ] = "16,24",
    dt_values: Annotated[
        str,
        typer.Option("--dt-values", help="Comma-separated RK4 time steps for time-step sweep."),
    ] = "0.02,0.01",
    reference_resolution: Annotated[
        int,
        typer.Option("--reference-resolution", help="Grid size used for the time-step sweep."),
    ] = 16,
    reference_dt: Annotated[
        float,
        typer.Option("--reference-dt", help="Time step used for the resolution sweep."),
    ] = 1.0e-2,
    width: Annotated[float, typer.Option("--width", help="Current-sheet half-width proxy.")] = 0.4,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 5.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 5.0e-3,
    perturbation_amplitude: Annotated[
        float,
        typer.Option("--perturbation-amplitude", help="Seed flux perturbation amplitude."),
    ] = 1.0e-3,
    mode_x: Annotated[int, typer.Option("--mode-x", help="Seed Fourier mode in x.")] = 2,
    mode_y: Annotated[int, typer.Option("--mode-y", help="Seed Fourier mode in y.")] = 1,
    t_end: Annotated[float, typer.Option("--t-end", help="Final nonlinear time.")] = 8.0,
    save_interval: Annotated[
        float,
        typer.Option("--save-interval", help="Approximate physical interval between saves."),
    ] = 1.0,
    fit_start: Annotated[float, typer.Option("--fit-start", help="Early fit window start.")] = 0.0,
    fit_stop: Annotated[float, typer.Option("--fit-stop", help="Early fit window stop.")] = 4.0,
    min_early_growth_rate: Annotated[
        float,
        typer.Option("--min-early-growth-rate", help="Minimum fitted early growth rate."),
    ] = 1.0e-3,
    min_max_growth_factor: Annotated[
        float,
        typer.Option("--min-max-growth-factor", help="Minimum maximum normalized growth."),
    ] = 1.05,
    max_relative_growth_rate_spread: Annotated[
        float,
        typer.Option(
            "--max-relative-growth-rate-spread",
            help="Maximum relative spread allowed in fitted early growth rates.",
        ),
    ] = 1.5,
    max_relative_max_growth_spread: Annotated[
        float,
        typer.Option(
            "--max-relative-max-growth-spread",
            help="Maximum relative spread allowed in maximum amplification.",
        ),
    ] = 3.0,
    max_relative_flux_amplification_spread: Annotated[
        float,
        typer.Option(
            "--max-relative-flux-amplification-spread",
            help="Maximum relative spread allowed in reconnecting-flux amplification.",
        ),
    ] = 3.0,
    max_relative_width_amplification_spread: Annotated[
        float,
        typer.Option(
            "--max-relative-width-amplification-spread",
            help="Maximum relative spread allowed in Rutherford-width amplification.",
        ),
    ] = 3.0,
) -> None:
    """Run seeded double-Harris resolution/time-step convergence scaffold."""
    _configure_validation_precision()
    manifest_path, validation = write_periodic_double_harris_convergence_validation(
        outdir,
        resolutions=_parse_int_tuple(resolutions),
        dt_values=_parse_float_tuple(dt_values),
        reference_resolution=reference_resolution,
        reference_dt=reference_dt,
        width=width,
        resistivity=eta,
        viscosity=nu,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_mode=(mode_x, mode_y),
        t_end=t_end,
        save_interval=save_interval,
        fit_window=(fit_start, fit_stop),
        min_early_growth_rate=min_early_growth_rate,
        min_max_growth_factor=min_max_growth_factor,
        max_relative_growth_rate_spread=max_relative_growth_rate_spread,
        max_relative_max_growth_spread=max_relative_max_growth_spread,
        max_relative_flux_amplification_spread=max_relative_flux_amplification_spread,
        max_relative_width_amplification_spread=max_relative_width_amplification_spread,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("double-harris-parameter-sweep")
def benchmark_double_harris_parameter_sweep(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for seeded double-Harris parameter-sweep artifacts.",
        ),
    ] = Path("outputs/benchmarks/periodic_double_harris_parameter_sweep"),
    sweep_axis: Annotated[
        str,
        typer.Option(
            "--sweep-axis",
            help="Parameter family to sweep: mode, width, or resistivity.",
        ),
    ] = "width",
    modes: Annotated[
        str,
        typer.Option(
            "--modes",
            help="Comma-separated Fourier modes as mx:my entries, e.g. 1:1,2:1,3:1.",
        ),
    ] = "2:1,2:2,4:1",
    widths: Annotated[
        str,
        typer.Option("--widths", help="Comma-separated sheet widths for width sweep."),
    ] = "0.35,0.4,0.45",
    etas: Annotated[
        str,
        typer.Option("--etas", help="Comma-separated resistivities for resistivity sweep."),
    ] = "0.004,0.005,0.006",
    viscosities: Annotated[
        str | None,
        typer.Option(
            "--viscosities",
            help="Optional comma-separated viscosities matching --etas; defaults to Pm=1.",
        ),
    ] = None,
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 16,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 16,
    width: Annotated[
        float,
        typer.Option("--width", help="Baseline current-sheet half-width proxy."),
    ] = 0.4,
    eta: Annotated[float, typer.Option("--eta", help="Baseline resistivity.")] = 5.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Baseline viscosity.")] = 5.0e-3,
    perturbation_amplitude: Annotated[
        float,
        typer.Option("--perturbation-amplitude", help="Seed flux perturbation amplitude."),
    ] = 1.0e-3,
    mode_x: Annotated[int, typer.Option("--mode-x", help="Baseline seed Fourier mode x.")] = 2,
    mode_y: Annotated[int, typer.Option("--mode-y", help="Baseline seed Fourier mode y.")] = 1,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    t_end: Annotated[float, typer.Option("--t-end", help="Final nonlinear time.")] = 6.0,
    save_interval: Annotated[
        float,
        typer.Option("--save-interval", help="Approximate physical interval between saves."),
    ] = 1.0,
    fit_start: Annotated[float, typer.Option("--fit-start", help="Early fit window start.")] = 0.0,
    fit_stop: Annotated[float, typer.Option("--fit-stop", help="Early fit window stop.")] = 3.0,
    min_early_growth_rate: Annotated[
        float,
        typer.Option("--min-early-growth-rate", help="Minimum fitted early growth rate."),
    ] = 1.0e-3,
    min_max_growth_factor: Annotated[
        float,
        typer.Option("--min-max-growth-factor", help="Minimum maximum normalized growth."),
    ] = 1.05,
    max_relative_growth_rate_spread: Annotated[
        float,
        typer.Option(
            "--max-relative-growth-rate-spread",
            help="Anomaly-check relative spread limit for fitted growth rates.",
        ),
    ] = 10.0,
    max_relative_max_growth_spread: Annotated[
        float,
        typer.Option(
            "--max-relative-max-growth-spread",
            help="Anomaly-check relative spread limit for maximum amplification.",
        ),
    ] = 20.0,
) -> None:
    """Run seeded double-Harris mode/width/resistivity response sweep."""
    _configure_validation_precision()
    parsed_viscosities = (
        _parse_float_tuple(viscosities) if viscosities is not None else None
    )
    manifest_path, validation = write_periodic_double_harris_parameter_sweep_validation(
        outdir,
        shape=(nx, ny),
        sweep_axis=sweep_axis,
        modes=_parse_mode_tuple(modes),
        widths=_parse_float_tuple(widths),
        resistivities=_parse_float_tuple(etas),
        viscosity_values=parsed_viscosities,
        width=width,
        resistivity=eta,
        viscosity=nu,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_mode=(mode_x, mode_y),
        dt=dt,
        t_end=t_end,
        save_interval=save_interval,
        fit_window=(fit_start, fit_stop),
        min_early_growth_rate=min_early_growth_rate,
        min_max_growth_factor=min_max_growth_factor,
        max_relative_growth_rate_spread=max_relative_growth_rate_spread,
        max_relative_max_growth_spread=max_relative_max_growth_spread,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("double-harris-promotion-check")
def benchmark_double_harris_promotion_check(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Seeded double-Harris long-run directory."),
    ],
    outdir: Annotated[
        Path | None,
        typer.Option(
            "--outdir",
            help="Promotion-report directory; defaults to <run-dir>/promotion.",
        ),
    ] = None,
    convergence_dirs: Annotated[
        list[Path] | None,
        typer.Option(
            "--convergence-dir",
            help="Double-Harris convergence evidence directory. Repeat for each sweep.",
        ),
    ] = None,
    require_movies: Annotated[
        bool,
        typer.Option(
            "--require-movies/--no-require-movies",
            help="Require flux/current GIFs in the run bundle.",
        ),
    ] = True,
    min_history_samples: Annotated[
        int,
        typer.Option("--min-history-samples", help="Minimum saved history samples."),
    ] = 30,
    min_convergence_dirs: Annotated[
        int,
        typer.Option("--min-convergence-dirs", help="Minimum convergence bundles."),
    ] = 1,
    min_t_end: Annotated[
        float,
        typer.Option("--min-t-end", help="Minimum terminal time for promotion."),
    ] = 30.0,
    min_reconnected_flux_amplification: Annotated[
        float,
        typer.Option(
            "--min-reconnected-flux-amplification",
            help="Minimum peak/initial reconnecting-flux response required.",
        ),
    ] = 1.05,
    min_island_width_amplification: Annotated[
        float,
        typer.Option(
            "--min-island-width-amplification",
            help="Minimum peak/initial Rutherford island-width response required.",
        ),
    ] = 1.05,
    max_relative_energy_increase: Annotated[
        float,
        typer.Option(
            "--max-relative-energy-increase",
            help="Maximum relative total-energy increase allowed.",
        ),
    ] = 1.0e-8,
) -> None:
    """Write a promotion-boundary report for seeded double-Harris evidence."""
    manifest_path, validation = write_periodic_double_harris_promotion_report(
        run_dir,
        outdir=outdir,
        convergence_dirs=tuple(convergence_dirs or ()),
        require_movies=require_movies,
        min_history_samples=min_history_samples,
        min_convergence_dirs=min_convergence_dirs,
        min_t_end=min_t_end,
        min_reconnected_flux_amplification=min_reconnected_flux_amplification,
        min_island_width_amplification=min_island_width_amplification,
        max_relative_energy_increase=max_relative_energy_increase,
    )
    typer.echo(f"wrote {manifest_path}")
    _exit_if_validation_failed(validation, context="double-harris-promotion-check")


@benchmark_app.command("nonlinear-energy-budget")
def benchmark_nonlinear_energy_budget(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for nonlinear energy-budget artifacts.",
        ),
    ] = Path("outputs/benchmarks/nonlinear_energy_budget"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 16,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 16,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 2.0e-2,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 2.0e-2,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    steps: Annotated[int, typer.Option("--steps", help="RK4 steps.")] = 80,
) -> None:
    """Validate the nonlinear reduced-MHD energy/dissipation budget."""
    _configure_validation_precision()
    manifest_path, validation = write_nonlinear_energy_budget_validation(
        outdir,
        shape=(nx, ny),
        resistivity=eta,
        viscosity=nu,
        dt=dt,
        steps=steps,
    )
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
    _configure_validation_precision()
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
    _configure_validation_precision()
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
    _configure_validation_precision()
    manifest_path, validation = write_arnoldi_validation(outdir)
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)












def _configure_validation_precision() -> None:
    configure_jax(enable_x64=True)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated integers") from exc


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated floats") from exc


def _parse_string_tuple(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise typer.BadParameter("expected at least one comma-separated value")
    return parsed


def _parse_mode_tuple(value: str) -> tuple[tuple[int, int], ...]:
    modes: list[tuple[int, int]] = []
    try:
        for part in value.split(","):
            item = part.strip()
            if not item:
                continue
            left, right = item.split(":", maxsplit=1)
            modes.append((int(left.strip()), int(right.strip())))
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated modes like 1:1,2:1") from exc
    if not modes:
        raise typer.BadParameter("expected at least one mode")
    return tuple(modes)


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
