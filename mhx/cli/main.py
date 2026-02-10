from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhx.config import TearingSimConfig, dump_config_yaml
from mhx.io.npz import savez
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
    tag: str = typer.Option("simulate", "--tag"),
    fast: bool = typer.Option(False, "--fast", help="Very small/short run for smoke tests."),
) -> None:
    eq_mode = equilibrium
    cfg = TearingSimConfig.fast(eq_mode) if fast else TearingSimConfig(equilibrium_mode=eq_mode)
    cfg = dataclasses.replace(cfg, eta=float(eta), nu=float(nu))

    if outdir is None:
        run = create_run_dir(tag=tag)
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "figures").mkdir(parents=True, exist_ok=True)
        run = RunPaths(run_dir=outdir)

    config_payload = {"sim": cfg.as_dict()}
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
    typer.echo(str(run.run_dir))


@app.command()
def scan(
    equilibrium: str = typer.Option("original", "--equilibrium"),
    grid: str = typer.Option("4x4", "--grid", help="Grid size as NxM in log10 space, e.g. 4x4."),
    outdir: Path = typer.Option(Path("outputs/scans"), "--outdir"),
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
                equilibrium_mode=equilibrium,
            )
            metrics = TearingMetrics.from_result(res)
            f_kin_grid[i, j] = metrics.f_kin
            C_grid[i, j] = metrics.complexity
            gamma_grid[i, j] = metrics.gamma_fit

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"reachable_region_scan_{equilibrium}.npz"
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
) -> None:
    from mhx.config import objective_preset
    from mhx.io.paths import create_run_dir
    from mhx.io.npz import savez
    from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics, TearingMetrics
    from mhx.config import dump_config_yaml

    cfg = TearingSimConfig.fast(equilibrium) if fast else TearingSimConfig(equilibrium_mode=equilibrium)
    objective = objective_preset(equilibrium)
    run_paths = create_run_dir(tag=f"inverse_{equilibrium}")
    dump_config_yaml(run_paths.config_yaml, {"sim": cfg.as_dict(), "objective": objective.as_dict()})

    # Minimal inverse-design loop: random search (placeholder)
    # TODO: wire full training loop from mhd_tearing_inverse_design.py
    best = None
    best_loss = float("inf")
    history = {"loss": [], "f_kin": [], "complexity": [], "eta": [], "nu": [],
               "target_f_kin": [objective.target_f_kin],
               "target_complexity": [objective.target_complexity],
               "lambda_complexity": [objective.lambda_complexity]}

    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(steps):
        eta = 10 ** rng.uniform(-4.5, -2.0)
        nu = 10 ** rng.uniform(-4.5, -2.0)
        res = _run_tearing_simulation_and_diagnostics(
            Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz,
            Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
            nu=nu, eta=eta,
            B0=cfg.B0, a=cfg.a, B_g=cfg.B_g, eps_B=cfg.eps_B,
            t0=cfg.t0, t1=cfg.t1, n_frames=cfg.n_frames, dt0=cfg.dt0,
            equilibrium_mode=equilibrium,
        )
        metrics = TearingMetrics.from_result(res)
        loss = (metrics.f_kin - objective.target_f_kin) ** 2 +                objective.lambda_complexity * (metrics.complexity - objective.target_complexity) ** 2

        history["loss"].append(float(loss))
        history["f_kin"].append(float(metrics.f_kin))
        history["complexity"].append(float(metrics.complexity))
        history["eta"].append(float(eta))
        history["nu"].append(float(nu))

        if loss < best_loss:
            best_loss = loss
            best = res

    if best is not None:
        savez(run_paths.solution_final_npz, best)
    savez(run_paths.history_npz, history)
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
