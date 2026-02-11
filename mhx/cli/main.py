from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhx.config import TearingSimConfig, dump_config_yaml, load_model_config
from mhx.io.npz import savez
from mhx.solver.plugins import build_terms
from mhx.io.paths import RunPaths, create_run_dir
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """MHX command line interface."""


@app.command()
def simulate(
    equilibrium: str = typer.Option("original", "--equilibrium", help="Equilibrium mode: original|forcefree"),
    eta: float = typer.Option(1e-3, "--eta"),
    nu: float = typer.Option(1e-3, "--nu"),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Run directory; if omitted a timestamped dir is created under outputs/runs."),
    model_config: Optional[Path] = typer.Option(None, "--model-config", help="YAML/JSON model config specifying equilibrium and physics terms."),
    tag: str = typer.Option("simulate", "--tag"),
    fast: bool = typer.Option(False, "--fast", help="Very small/short run for smoke tests."),
) -> None:
    eq_mode = equilibrium
    cfg = TearingSimConfig.fast(eq_mode) if fast else TearingSimConfig(equilibrium_mode=eq_mode)
    cfg = dataclasses.replace(cfg, eta=float(eta), nu=float(nu))

    model_cfg = None
    terms = None
    if model_config is not None:
        model_cfg = load_model_config(model_config)
        eq_mode = model_cfg.equilibrium_mode or eq_mode
        cfg = dataclasses.replace(cfg, equilibrium_mode=eq_mode)
        terms = build_terms(model_cfg.rhs_terms, model_cfg.term_params)

    if outdir is None:
        run = create_run_dir(tag=tag)
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "figures").mkdir(parents=True, exist_ok=True)
        run = RunPaths(run_dir=outdir)

    config_payload = {"sim": cfg.as_dict()}
    if model_cfg is not None:
        config_payload["model"] = model_cfg.as_dict()
    dump_config_yaml(run.config_yaml, config_payload)

    res = _run_tearing_simulation_and_diagnostics(
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        Nz=cfg.Nz,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        Lz=cfg.Lz,
        nu=cfg.nu,
        eta=cfg.eta,
        B0=cfg.B0,
        a=cfg.a,
        B_g=cfg.B_g,
        eps_B=cfg.eps_B,
        t0=cfg.t0,
        t1=cfg.t1,
        n_frames=cfg.n_frames,
        dt0=cfg.dt0,
        equilibrium_mode=cfg.equilibrium_mode,
        terms=terms,
        progress=cfg.progress,
        jit=cfg.jit,
        check_finite=cfg.check_finite,
    )

    # Convert JAX arrays to NumPy scalars/arrays for saving.
    payload = {}
    for k, v in res.items():
        import jax.numpy as jnp

        if isinstance(v, jnp.ndarray):
            payload[k] = np.array(v)
        else:
            payload[k] = np.array(v)

    savez(run.solution_final_npz, payload)
    log_lines = [
        "mhx simulate",
        f"run_dir: {run.run_dir}",
        f"sanity_finite_ok: {bool(payload.get('sanity_finite_ok', True))}",
        f"sanity_nonneg_ok: {bool(payload.get('sanity_nonneg_ok', True))}",
        f"sanity_energy_ratio: {float(payload.get('sanity_energy_ratio', float('nan'))):.3e}",
        f"sanity_energy_ratio_warn: {bool(payload.get('sanity_energy_ratio_warn', False))}",
    ]
    run.logs_txt.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    typer.echo(str(run.run_dir))


@app.command()
def scan(
    equilibrium: str = typer.Option("original", "--equilibrium"),
    grid: str = typer.Option("4x4", "--grid", help="Grid size as NxM in log10 space, e.g. 4x4."),
    outdir: Path = typer.Option(Path("outputs/scans"), "--outdir"),
    model_config: Optional[Path] = typer.Option(None, "--model-config", help="Optional YAML/JSON model config for physics terms."),
) -> None:
    from mhx.config import TearingSimConfig
    from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics, TearingMetrics

    n_eta, n_nu = [int(x) for x in grid.lower().split("x")]
    log10_eta_vals = np.linspace(-4.5, -2.0, n_eta)
    log10_nu_vals = np.linspace(-4.5, -2.0, n_nu)

    f_kin_grid = np.zeros((n_eta, n_nu))
    C_grid = np.zeros((n_eta, n_nu))
    gamma_grid = np.zeros((n_eta, n_nu))

    cfg = TearingSimConfig.fast(equilibrium)
    terms = None
    if model_config is not None:
        model_cfg = load_model_config(model_config)
        if model_cfg.equilibrium_mode:
            cfg = dataclasses.replace(cfg, equilibrium_mode=model_cfg.equilibrium_mode)
        terms = build_terms(model_cfg.rhs_terms, model_cfg.term_params)

    for i, log10_eta in enumerate(log10_eta_vals):
        for j, log10_nu in enumerate(log10_nu_vals):
            eta = 10.0 ** log10_eta
            nu = 10.0 ** log10_nu
            res = _run_tearing_simulation_and_diagnostics(
                Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz,
                Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
                nu=nu, eta=eta,
                B0=cfg.B0, a=cfg.a, B_g=cfg.B_g, eps_B=cfg.eps_B,
                t0=cfg.t0, t1=cfg.t1, n_frames=cfg.n_frames, dt0=cfg.dt0,
                equilibrium_mode=cfg.equilibrium_mode,
                terms=terms,
            )
            metrics = TearingMetrics.from_result(res)
            f_kin_grid[i, j] = metrics.f_kin
            C_grid[i, j] = metrics.complexity
            gamma_grid[i, j] = metrics.gamma_fit

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"reachable_region_scan_{cfg.equilibrium_mode}.npz"
    np.savez(
        outpath,
        log10_eta_vals=log10_eta_vals,
        log10_nu_vals=log10_nu_vals,
        f_kin_grid=f_kin_grid,
        C_grid=C_grid,
        gamma_grid=gamma_grid,
        eq_mode=equilibrium,
    )
    typer.echo(str(outpath))


@app.command()
def inverse_design(
    equilibrium: str = typer.Option("forcefree", "--equilibrium"),
    steps: int = typer.Option(2, "--steps"),
    fast: bool = typer.Option(True, "--fast"),
    model_config: Optional[Path] = typer.Option(None, "--model-config", help="Optional YAML/JSON model config for physics terms."),
) -> None:
    from mhx.inverse_design.train import InverseDesignConfig, run_inverse_design

    cfg = InverseDesignConfig.fast(equilibrium) if fast else InverseDesignConfig.default(equilibrium)
    if model_config is not None:
        model_cfg = load_model_config(model_config)
        cfg.model = model_cfg
        if model_cfg.equilibrium_mode:
            cfg.equilibrium_mode = model_cfg.equilibrium_mode
    cfg.n_train_steps = steps
    run_paths, _, _, _, _ = run_inverse_design(cfg)
    typer.echo(str(run_paths.run_dir))


@app.command()
def figures(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False, dir_okay=True, help="Run directory created by `mhx simulate`."),
    solution: str = typer.Option("solution_final.npz", "--solution", help="Which solution NPZ to plot (relative to --run)."),
) -> None:
    run_paths = RunPaths(run_dir=run)
    npz_path = run / solution
    data = np.load(npz_path, allow_pickle=True)

    figs = run_paths.figures_dir
    figs.mkdir(parents=True, exist_ok=True)

    # Energies
    if "ts" in data.files and "E_kin" in data.files and "E_mag" in data.files:
        ts = data["ts"]
        E_kin = data["E_kin"]
        E_mag = data["E_mag"]

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot(ts, E_kin, label="E_kin")
        ax.plot(ts, E_mag, label="E_mag")
        ax.set_xlabel("t")
        ax.set_ylabel("Energy")
        ax.set_title("Energy time series")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / "energy.png", dpi=200)
        plt.close(fig)

    # Final midplane Az profile + complexity
    if "Az_final_mid" in data.files and "Ly" in data.files:
        Az = data["Az_final_mid"]
        Ly = float(np.array(data["Ly"]))
        y = np.linspace(0.0, Ly, Az.shape[0], endpoint=False)
        complexity = float(np.array(data["complexity_final"])) if "complexity_final" in data.files else float("nan")

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot(y, Az)
        ax.set_xlabel("y")
        ax.set_ylabel(r"$A_z(x_{mid}, y, z=0, t_{final})$")
        ax.set_title(f"Final midplane $A_z$ (complexity≈{complexity:.2e})")
        fig.tight_layout()
        fig.savefig(figs / "az_midplane.png", dpi=200)
        plt.close(fig)

    typer.echo(str(figs))


if __name__ == "__main__":
    app()
