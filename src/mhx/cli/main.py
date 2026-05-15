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
    write_benchmark_catalog,
    write_cosine_equilibrium_linearization_validation,
    write_diffusion_eigenvalue_validation,
    write_duration_policy,
    write_fkr_growth_rate_validation,
    write_fkr_window_validation,
    write_harris_delta_prime_validation,
    write_linear_tearing_dispersion_validation,
    write_linear_tearing_eigenvalue_validation,
    write_linear_tearing_layer_validation,
    write_linear_tearing_timedomain_validation,
    write_linearized_rhs_validation,
    write_nonlinear_duration_audit,
    write_nonlinear_energy_budget_validation,
    write_periodic_current_sheet_eigenvalue_validation,
    write_periodic_current_sheet_nonlinear_bridge_validation,
    write_periodic_current_sheet_timedomain_validation,
    write_periodic_double_harris_convergence_validation,
    write_periodic_double_harris_nonlinear_growth_validation,
    write_periodic_double_harris_seeded_long_run_validation,
    write_power_iteration_validation,
    write_readiness_report,
    write_reconnection_scaling_validation,
    write_reduced_mhd_linear_eigenmode_validation,
    write_resistive_decay_validation,
    write_run_report,
    write_rutherford_campaign_fast,
    write_rutherford_campaign_template,
    write_seed_robust_qi_sweep,
    write_seed_robust_qi_validation,
    write_timing_benchmark,
    write_validation_suite,
)
from mhx.campaigns import (
    WalltimePolicy,
    write_rutherford_production_execution,
    write_rutherford_production_plan,
    write_rutherford_resume_plan,
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
from mhx.neural_ode import (
    write_neural_ode_reproducibility_bundle,
    write_neural_ode_training_bundle,
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
campaign_app = typer.Typer(no_args_is_help=True, help="Production campaign templates.")
physics_app = typer.Typer(no_args_is_help=True, help="Physics plugin inspection.")
diagnostics_app = typer.Typer(no_args_is_help=True, help="Diagnostic registry inspection.")
validate_app = typer.Typer(no_args_is_help=True, help="Validation-suite workflows.")
neural_ode_app = typer.Typer(
    no_args_is_help=True,
    help="Neural-ODE reproducibility datasets and baseline artifacts.",
)
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(api_app, name="api")
app.add_typer(campaign_app, name="campaign")
app.add_typer(physics_app, name="physics")
app.add_typer(diagnostics_app, name="diagnostics")
app.add_typer(validate_app, name="validate")
app.add_typer(neural_ode_app, name="neural-ode")


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
    typer.echo(f"Validation-suite schema: {info['validation_suite_schema']}")
    typer.echo(f"Claim levels: {', '.join(info['claim_levels'])}")


@api_app.command("deprecations")
def api_deprecations() -> None:
    """Print active deprecation guidance for legacy entry points."""
    typer.echo("Legacy scripts live under legacy/old_mhx/ and are not imported by src/mhx.")
    typer.echo("Use mhx run, mhx benchmark, mhx validate, mhx figures, and mhx report instead.")
    typer.echo("See docs/migration.md and RELEASE.md for the deprecation window.")


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
            "mhx run currently supports physics.model='reduced_mhd_linear_tearing'. "
            "Production campaign templates are planning artifacts, not runnable "
            "through the FAST smoke runner."
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


@campaign_app.command("rutherford-template")
def campaign_rutherford_template(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for campaign-template artifacts."),
    ] = Path("outputs/campaigns/rutherford_template"),
    harris_growth_rate: Annotated[
        float,
        typer.Option("--harris-growth-rate", help="Reference Harris growth rate gamma."),
    ] = 1.31e-2,
    production_efolds: Annotated[
        float,
        typer.Option("--production-efolds", help="Required e-folds before nonlinear claims."),
    ] = 10.0,
    safety_factor: Annotated[
        float,
        typer.Option("--safety-factor", help="Extra multiplier for nonlinear tracking."),
    ] = 3.0,
    nx: Annotated[int, typer.Option("--nx", help="Template x resolution.")] = 128,
    ny: Annotated[int, typer.Option("--ny", help="Template y resolution.")] = 128,
    dt: Annotated[float, typer.Option("--dt", help="Template fixed time step.")] = 0.1,
    target_saved_frames: Annotated[
        int,
        typer.Option("--target-saved-frames", help="Target saved frames for figures/movies."),
    ] = 400,
    run_output_dir: Annotated[
        Path,
        typer.Option("--run-output-dir", help="Output directory embedded in campaign_config.toml."),
    ] = Path("outputs/production/rutherford_island"),
) -> None:
    """Write a duration-guarded nonlinear Rutherford campaign template."""
    manifest_path, validation = write_rutherford_campaign_template(
        outdir,
        harris_growth_rate=harris_growth_rate,
        production_efolds=production_efolds,
        safety_factor=safety_factor,
        shape=(nx, ny),
        dt=dt,
        target_saved_frames=target_saved_frames,
        run_output_dir=run_output_dir,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@campaign_app.command("rutherford-run-fast")
def campaign_rutherford_run_fast(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for validation campaign artifacts."),
    ] = Path("outputs/campaigns/rutherford_fast"),
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 24,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 24,
    t_end: Annotated[float, typer.Option("--t-end", help="Short validation final time.")] = 0.24,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    save_every: Annotated[int, typer.Option("--save-every", help="Saved-step stride.")] = 1,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic perturbation seed.")] = 0,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 1.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 1.0e-3,
    noise_amplitude: Annotated[
        float,
        typer.Option("--noise-amplitude", help="Smooth seeded noise amplitude."),
    ] = 1.0e-6,
    gif: Annotated[bool, typer.Option("--gif", help="Write a fixed-scale flux GIF.")] = False,
) -> None:
    """Run a short validation-grade Rutherford-style campaign."""
    _configure_validation_precision()
    manifest_path, validation = write_rutherford_campaign_fast(
        outdir,
        seeds=(seed,),
        shape=(nx, ny),
        t_end=t_end,
        dt=dt,
        save_every=save_every,
        seed=seed,
        resistivity=eta,
        viscosity=nu,
        noise_amplitude=noise_amplitude,
        write_gif=gif,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@campaign_app.command("rutherford-plan-production")
def campaign_rutherford_plan_production(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for production-plan artifacts."),
    ] = Path("outputs/campaigns/rutherford_production_plan"),
    harris_growth_rate: Annotated[
        float,
        typer.Option("--harris-growth-rate", help="Reference Harris growth rate gamma."),
    ] = 1.31e-2,
    production_efolds: Annotated[
        float,
        typer.Option("--production-efolds", help="Required e-folds before nonlinear claims."),
    ] = 10.0,
    safety_factor: Annotated[
        float,
        typer.Option("--safety-factor", help="Extra multiplier for nonlinear tracking."),
    ] = 3.0,
    nx: Annotated[int, typer.Option("--nx", help="Production x resolution.")] = 128,
    ny: Annotated[int, typer.Option("--ny", help="Production y resolution.")] = 128,
    dt: Annotated[float, typer.Option("--dt", help="Production fixed time step.")] = 0.1,
    target_saved_frames: Annotated[
        int,
        typer.Option("--target-saved-frames", help="Target saved frames for figures/movies."),
    ] = 400,
    max_walltime_hours: Annotated[
        float,
        typer.Option("--max-walltime-hours", help="Scheduler walltime budget per job."),
    ] = 12.0,
    seconds_per_step_estimate: Annotated[
        float,
        typer.Option("--seconds-per-step-estimate", help="Conservative runtime model."),
    ] = 0.5,
    checkpoint_interval_minutes: Annotated[
        float,
        typer.Option("--checkpoint-interval-minutes", help="Target checkpoint cadence."),
    ] = 30.0,
    preemption_margin_minutes: Annotated[
        float,
        typer.Option("--preemption-margin-minutes", help="Walltime reserved for safe exit."),
    ] = 20.0,
    min_production_resolution: Annotated[
        int,
        typer.Option("--min-production-resolution", help="Minimum reviewer-facing grid size."),
    ] = 96,
) -> None:
    """Write production Rutherford planning, walltime, and checkpoint artifacts."""
    manifest_path, validation = write_rutherford_production_plan(
        outdir,
        harris_growth_rate=harris_growth_rate,
        production_efolds=production_efolds,
        safety_factor=safety_factor,
        shape=(nx, ny),
        dt=dt,
        target_saved_frames=target_saved_frames,
        min_production_resolution=min_production_resolution,
        walltime_policy=WalltimePolicy(
            max_walltime_hours=max_walltime_hours,
            seconds_per_step_estimate=seconds_per_step_estimate,
            checkpoint_interval_minutes=checkpoint_interval_minutes,
            preemption_margin_minutes=preemption_margin_minutes,
        ),
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@campaign_app.command("rutherford-resume-plan")
def campaign_rutherford_resume_plan(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Production campaign run directory containing checkpoints/."),
    ],
    target_step: Annotated[
        int | None,
        typer.Option("--target-step", help="Override the campaign target step."),
    ] = None,
) -> None:
    """Write a resume plan from the latest valid production checkpoint."""
    path, validation = write_rutherford_resume_plan(run_dir, target_step=target_step)
    typer.echo(f"wrote {path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@campaign_app.command("rutherford-execute")
def campaign_rutherford_execute(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Production campaign run directory with campaign_plan.json."),
    ],
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", help="Maximum RK4 steps for this walltime chunk."),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Initial-condition seed used if no checkpoint is valid."),
    ] = 0,
    noise_amplitude: Annotated[
        float,
        typer.Option("--noise-amplitude", help="Seeded smooth perturbation amplitude."),
    ] = 1.0e-6,
    movies_enabled: Annotated[
        bool,
        typer.Option("--movies/--no-movies", help="Write fixed-scale flux/current GIFs."),
    ] = False,
    allow_production_claim: Annotated[
        bool,
        typer.Option(
            "--allow-production-claim/--validation-claim",
            help="Allow claim_level=production only when target completion gates pass.",
        ),
    ] = False,
    max_relative_energy_growth: Annotated[
        float,
        typer.Option("--max-relative-energy-growth", help="Energy monotonicity tolerance."),
    ] = 1.0e-9,
    max_divergence_linf: Annotated[
        float,
        typer.Option("--max-divergence-linf", help="Magnetic-divergence tolerance."),
    ] = 1.0e-9,
) -> None:
    """Run one restartable Rutherford production-campaign walltime chunk."""
    _configure_validation_precision()
    manifest_path, validation = write_rutherford_production_execution(
        run_dir,
        max_steps=max_steps,
        seed=seed,
        noise_amplitude=noise_amplitude,
        write_movies=movies_enabled,
        allow_production_claim=allow_production_claim,
        max_relative_energy_growth=max_relative_energy_growth,
        max_divergence_linf=max_divergence_linf,
    )
    typer.echo(f"wrote {manifest_path}")
    _exit_if_validation_failed(validation, context="rutherford-execute")


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
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


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


@benchmark_app.command("nonlinear-duration-audit")
def benchmark_nonlinear_duration_audit(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for nonlinear duration-audit artifacts.",
        ),
    ] = Path("outputs/benchmarks/nonlinear_duration_audit"),
    harris_growth_rate: Annotated[
        float,
        typer.Option(
            "--harris-growth-rate",
            help="Reference Harris-sheet growth rate gamma for the duration audit.",
        ),
    ] = 1.31e-2,
    linear_efolds: Annotated[
        float,
        typer.Option("--linear-efolds", help="Target number of linear e-folds."),
    ] = 10.0,
) -> None:
    """Audit nonlinear FAST run durations against literature-scale targets."""
    manifest_path, validation = write_nonlinear_duration_audit(
        outdir,
        harris_growth_rate=harris_growth_rate,
        requested_linear_efolds=linear_efolds,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("seed-robust-qi")
def benchmark_seed_robust_qi(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for seed-robust quality-indicator artifacts.",
        ),
    ] = Path("outputs/benchmarks/seed_robust_qi"),
    seeds: Annotated[
        str,
        typer.Option("--seeds", help="Comma-separated deterministic seed list."),
    ] = "0,1,2,3",
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 16,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 16,
    t_end: Annotated[float, typer.Option("--t-end", help="Final ensemble time.")] = 0.12,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 1.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 1.0e-3,
    noise_amplitude: Annotated[
        float,
        typer.Option("--noise-amplitude", help="Smooth seeded noise amplitude."),
    ] = 1.0e-6,
) -> None:
    """Run a seed-robust quality-indicator ensemble for FAST metrics."""
    _configure_validation_precision()
    manifest_path, validation = write_seed_robust_qi_validation(
        outdir,
        seeds=_parse_int_tuple(seeds),
        shape=(nx, ny),
        steps=max(1, round(t_end / dt)),
        dt=dt,
        resistivity=eta,
        viscosity=nu,
        psi_noise_amplitude=noise_amplitude,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("seed-robust-qi-sweep")
def benchmark_seed_robust_qi_sweep(
    outdir: Annotated[
        Path,
        typer.Option(
            "--outdir",
            help="Output directory for seed/amplitude-sweep QI artifacts.",
        ),
    ] = Path("outputs/benchmarks/seed_robust_qi_sweep"),
    seeds: Annotated[
        str,
        typer.Option("--seeds", help="Comma-separated deterministic seed list."),
    ] = "0,1,2,3",
    amplitudes: Annotated[
        str,
        typer.Option("--amplitudes", help="Comma-separated sorted noise amplitudes."),
    ] = "0,1e-9,1e-8",
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 16,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 16,
    steps: Annotated[int, typer.Option("--steps", help="RK4 steps per amplitude.")] = 12,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 1.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 1.0e-3,
) -> None:
    """Run a common-seed perturbation-amplitude sweep for FAST metric QI."""
    _configure_validation_precision()
    manifest_path, validation = write_seed_robust_qi_sweep(
        outdir,
        seeds=_parse_int_tuple(seeds),
        noise_amplitudes=_parse_float_tuple(amplitudes),
        shape=(nx, ny),
        steps=steps,
        dt=dt,
        resistivity=eta,
        viscosity=nu,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@neural_ode_app.command("dataset")
def neural_ode_dataset(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for neural-ODE artifacts."),
    ] = Path("outputs/neural_ode/seed_qi_fast"),
    seeds: Annotated[
        str,
        typer.Option("--seeds", help="Comma-separated deterministic seed list."),
    ] = "0,1,2,3,4,5",
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 16,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 16,
    steps: Annotated[int, typer.Option("--steps", help="RK4 steps per seed.")] = 24,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 1.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 1.0e-3,
    noise_amplitude: Annotated[
        float,
        typer.Option("--noise-amplitude", help="Smooth seeded psi-noise amplitude."),
    ] = 1.0e-8,
    split_seed: Annotated[
        int,
        typer.Option("--split-seed", help="Seed for train/validation/test split."),
    ] = 314159,
    observation_count: Annotated[
        int,
        typer.Option("--observation-count", help="Prefix samples visible to baselines."),
    ] = 2,
    figures_enabled: Annotated[bool, typer.Option("--figures/--no-figures")] = True,
) -> None:
    """Write deterministic neural-ODE dataset, splits, baselines, and calibration."""
    _configure_validation_precision()
    manifest_path, validation = write_neural_ode_reproducibility_bundle(
        outdir,
        seeds=_parse_int_tuple(seeds),
        shape=(nx, ny),
        steps=steps,
        dt=dt,
        resistivity=eta,
        viscosity=nu,
        psi_noise_amplitude=noise_amplitude,
        split_seed=split_seed,
        observation_count=observation_count,
        write_figures=figures_enabled,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@neural_ode_app.command("train")
def neural_ode_train(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for fitted neural-ODE artifacts."),
    ] = Path("outputs/neural_ode/latent_ode_fast"),
    seeds: Annotated[
        str,
        typer.Option("--seeds", help="Comma-separated deterministic seed list."),
    ] = "0,1,2,3,4,5",
    nx: Annotated[int, typer.Option("--nx", help="Grid points in x.")] = 16,
    ny: Annotated[int, typer.Option("--ny", help="Grid points in y.")] = 16,
    steps: Annotated[int, typer.Option("--steps", help="RK4 steps per seed.")] = 24,
    dt: Annotated[float, typer.Option("--dt", help="RK4 time step.")] = 1.0e-2,
    eta: Annotated[float, typer.Option("--eta", help="Resistivity.")] = 1.0e-3,
    nu: Annotated[float, typer.Option("--nu", help="Viscosity.")] = 1.0e-3,
    noise_amplitude: Annotated[
        float,
        typer.Option("--noise-amplitude", help="Smooth seeded psi-noise amplitude."),
    ] = 1.0e-8,
    split_seed: Annotated[
        int,
        typer.Option("--split-seed", help="Seed for train/validation/test split."),
    ] = 314159,
    observation_count: Annotated[
        int,
        typer.Option("--observation-count", help="Prefix samples visible to forecasters."),
    ] = 2,
    hidden_size: Annotated[
        int,
        typer.Option("--hidden-size", help="Random tanh features in the latent ODE."),
    ] = 8,
    ridge: Annotated[
        float,
        typer.Option("--ridge", help="Ridge regularization for latent-ODE regression."),
    ] = 1.0e-8,
    model_seed: Annotated[
        int,
        typer.Option("--model-seed", help="Deterministic random-feature seed."),
    ] = 0,
    figures_enabled: Annotated[bool, typer.Option("--figures/--no-figures")] = True,
) -> None:
    """Fit a deterministic random-feature latent ODE on FAST seed-QI trajectories."""
    _configure_validation_precision()
    manifest_path, validation = write_neural_ode_training_bundle(
        outdir,
        seeds=_parse_int_tuple(seeds),
        shape=(nx, ny),
        steps=steps,
        dt=dt,
        resistivity=eta,
        viscosity=nu,
        psi_noise_amplitude=noise_amplitude,
        split_seed=split_seed,
        observation_count=observation_count,
        hidden_size=hidden_size,
        ridge=ridge,
        model_seed=model_seed,
        write_figures=figures_enabled,
    )
    typer.echo(f"wrote {manifest_path}")
    if not validation["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("duration-policy")
def benchmark_duration_policy(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for duration-policy artifacts."),
    ] = Path("outputs/benchmarks/duration_policy"),
    harris_growth_rate: Annotated[
        float,
        typer.Option("--harris-growth-rate", help="Reference Harris growth rate gamma."),
    ] = 1.31e-2,
    production_efolds: Annotated[
        float,
        typer.Option("--production-efolds", help="Required e-folds for production claims."),
    ] = 10.0,
) -> None:
    """Write simulation-duration policy for validation and production runs."""
    manifest_path, validation = write_duration_policy(
        outdir,
        harris_growth_rate=harris_growth_rate,
        production_efolds=production_efolds,
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


@validate_app.command("all")
def validate_all(
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for validation-suite artifacts."),
    ] = Path("outputs/validation_suite"),
) -> None:
    """Run the active deterministic FAST validation suite."""
    summary_path, summary = write_validation_suite(outdir)
    typer.echo(f"wrote {summary_path}")
    if not summary["passed"]:
        raise typer.Exit(code=1)


@validate_app.command("readiness")
def validate_readiness(
    suite: Annotated[
        Path,
        typer.Option(
            "--suite",
            help="Validation-suite directory or validation_suite.json file.",
        ),
    ] = Path("outputs/validation_suite"),
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Output directory for readiness artifacts."),
    ] = Path("outputs/validation/readiness"),
) -> None:
    """Assess public-release and nonlinear-publication readiness."""
    diagnostics_path, validation = write_readiness_report(outdir, suite)
    typer.echo(f"wrote {diagnostics_path}")
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
