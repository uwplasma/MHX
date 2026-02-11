from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mhx.config import TearingSimConfig
from mhx.inverse_design.train import InverseDesignConfig, run_inverse_design
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics, TearingMetrics


def main() -> None:
    # Run a tiny inverse design to get history
    cfg = InverseDesignConfig.fast("forcefree")
    cfg.n_train_steps = 3
    run_paths, history, _, _, _ = run_inverse_design(cfg)

    # Tiny scan for reachable region
    sim = TearingSimConfig.fast("forcefree")
    n_eta = 3
    n_nu = 3
    log10_eta_vals = np.linspace(-4.5, -2.0, n_eta)
    log10_nu_vals = np.linspace(-4.5, -2.0, n_nu)

    f_kin_grid = np.zeros((n_eta, n_nu))
    C_grid = np.zeros((n_eta, n_nu))

    for i, log10_eta in enumerate(log10_eta_vals):
        for j, log10_nu in enumerate(log10_nu_vals):
            eta = 10.0 ** log10_eta
            nu = 10.0 ** log10_nu
            res = _run_tearing_simulation_and_diagnostics(
                Nx=sim.Nx,
                Ny=sim.Ny,
                Nz=sim.Nz,
                Lx=sim.Lx,
                Ly=sim.Ly,
                Lz=sim.Lz,
                nu=nu,
                eta=eta,
                B0=sim.B0,
                a=sim.a,
                B_g=sim.B_g,
                eps_B=sim.eps_B,
                t0=sim.t0,
                t1=sim.t1,
                n_frames=sim.n_frames,
                dt0=sim.dt0,
                equilibrium_mode=sim.equilibrium_mode,
                progress=sim.progress,
                jit=sim.jit,
                check_finite=sim.check_finite,
            )
            metrics = TearingMetrics.from_result(res)
            f_kin_grid[i, j] = float(metrics.f_kin)
            C_grid[i, j] = float(metrics.complexity)

    # Heatmap of f_kin
    X, Y = np.meshgrid(log10_nu_vals, log10_eta_vals)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.pcolormesh(X, Y, f_kin_grid, shading="auto")
    fig.colorbar(im, ax=ax, label="f_kin")
    ax.set_xlabel("log10 nu")
    ax.set_ylabel("log10 eta")
    ax.set_title("Reachable f_kin (FAST)")
    fig.tight_layout()
    fig.savefig("docs/_static/fig_reachable_heatmap.png", dpi=200)
    plt.close(fig)

    # Reachable region scatter
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(f_kin_grid.ravel(), C_grid.ravel(), s=40, alpha=0.8)
    ax.set_xlabel("f_kin")
    ax.set_ylabel("C_plasmoid")
    ax.set_title("Reachable region (FAST)")
    fig.tight_layout()
    fig.savefig("docs/_static/fig_reachable_region.png", dpi=200)
    plt.close(fig)

    # Cost history: grid vs inverse design
    f_target = history["target_f_kin"][0]
    C_target = history["target_complexity"][0]
    lam = history["lambda_complexity"][0]

    costs_grid = (f_kin_grid.ravel() - f_target) ** 2 + lam * (C_grid.ravel() - C_target) ** 2
    best_so_far_grid = np.minimum.accumulate(costs_grid)

    f_inv = np.array(history["f_kin"], dtype=float)
    C_inv = np.array(history["complexity"], dtype=float)
    costs_inv = (f_inv - f_target) ** 2 + lam * (C_inv - C_target) ** 2

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.semilogy(np.arange(1, len(costs_grid) + 1), best_so_far_grid, "o-", label="grid")
    ax.semilogy(np.arange(1, len(costs_inv) + 1), costs_inv, "s-", label="inverse")
    ax.set_xlabel("# simulations")
    ax.set_ylabel("cost")
    ax.set_title("Cost history (FAST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/_static/fig_cost_history.png", dpi=200)
    plt.close(fig)

    print(f"Saved media. Run dir: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
