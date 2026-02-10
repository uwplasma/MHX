#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_tearing_inverse_design_figures.py

Publication-ready figure generator for the tearing / plasmoid inverse-design study.

This script assumes that:
  * mhd_tearing_solve.py is available and working.
  * mhd_tearing_inverse_design.py has ALREADY been run at least once,
    so that you have a run directory under outputs/runs/... with:
        - solution_initial.npz
        - solution_final.npz
        - history.npz (training history arrays, if inverse design was run)

What this script does
---------------------
1) Parameter scans in (log10_eta, log10_nu) for each equilibrium:
       eq_mode in {"original", "forcefree"}.
   For each grid point it runs a full MHD simulation (via _simulate_metrics)
   and stores:
       - f_kin: late-time kinetic-energy fraction
       - C_plasmoid: midplane plasmoid complexity
       - gamma_fit: exponential growth rate of the tearing mode

   These are saved to:
       reachable_region_scan_<eq_mode>.npz

2) Reachable-region figures:
   (a) Heatmaps vs (log10_eta, log10_nu) for each eq_mode:
         - f_kin
         - C_plasmoid
         - gamma_fit
   (b) Combined reachable region in (f_kin, C_plasmoid) plane.
   (c) Scatter of gamma_fit vs f_kin for both equilibria.

3) Inverse-design vs grid-search comparison for one equilibrium
   (by default the plasmoid-prone "forcefree" branch):

   It *does not* re-run inverse design. Instead it expects a history file:

       outputs/runs/<timestamp>_inverse_<eq_mode>/history.npz

   produced by a slightly extended version of mhd_tearing_inverse_design.py
   that saves the training history dict:

       history = {
           "loss": [...],
           "log10_eta": [...],
           "log10_nu": [...],
           "eta": [...],
           "nu": [...],
           "f_kin": [...],
           "complexity": [...],
       }

   saved as:
       np.savez(history.npz, **history)

   Using that, we build:
       - Cost vs # simulations: grid search vs inverse design.
       - Solutions in (f_kin, C_plasmoid) plane:
           grid samples, grid best, inverse-design final point, target.

All figures are saved as high-resolution PNGs with descriptive filenames
and are intended to drop straight into the paper.
"""

from __future__ import annotations

import os
import math
import glob
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhx.inverse_design.train import (
    InverseDesignConfig,
    _simulate_metrics,
)
from mhx.config import Objective, objective_preset

# -----------------------------------------------------------------------------#
# Global plotting style
# -----------------------------------------------------------------------------#

plt.rcParams.update(
    {
        "font.size": 13,
        "text.usetex": False,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 220,
        "axes.linewidth": 1.1,
        "lines.linewidth": 1.8,
    }
)

# -----------------------------------------------------------------------------#
# Configuration for scans + inverse-design postprocessing
# -----------------------------------------------------------------------------#


@dataclass
class FigureScanConfig:
    # Scan grid in log10_eta, log10_nu
    n_eta: int = 4       # for production runs, consider 6–8
    n_nu: int = 4
    log10_eta_min: float = -4.5
    log10_eta_max: float = -2.0
    log10_nu_min: float = -4.5
    log10_nu_max: float = -2.0

    # Shared physical/grid parameters (kept in sync with InverseDesignConfig)
    Nx: int = 64
    Ny: int = 64
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi
    B0: float = 1.0
    B_g: float = 0.2
    a: float = 0.25
    eps_B: float = 1e-3

    # Time integration
    t0: float = 0.0
    t1: float = 60.0
    n_frames: int = 120
    dt0: float = 5e-4

    # Default objective used when scanning and when inverse-design history does
    # not provide one. For fair comparisons, comparisons should prefer the
    # objective loaded from the saved inverse-design history.
    objective: Objective = field(default_factory=lambda: objective_preset("forcefree"))

    # Equilibria to scan
    eq_modes: Tuple[str, str] = ("original", "forcefree")

    # Which equilibrium has been used in inverse design
    inverse_eq_mode: str = "forcefree"

    # Paths for inverse-design history (no optimisation is run here).
    # By default we look under outputs/runs for a matching inverse run.
    def inverse_history_path(self) -> str:
        return f"outputs/runs/*_inverse_{self.inverse_eq_mode}/history.npz"


# -----------------------------------------------------------------------------#
#  Helper: single simulation call
# -----------------------------------------------------------------------------#


def build_inverse_cfg(fig_cfg: FigureScanConfig, eq_mode: str) -> InverseDesignConfig:
    """Create an InverseDesignConfig instance consistent with fig_cfg."""
    return InverseDesignConfig(
        Nx=fig_cfg.Nx,
        Ny=fig_cfg.Ny,
        Nz=fig_cfg.Nz,
        Lx=fig_cfg.Lx,
        Ly=fig_cfg.Ly,
        Lz=fig_cfg.Lz,
        B0=fig_cfg.B0,
        B_g=fig_cfg.B_g,
        a=fig_cfg.a,
        eps_B=fig_cfg.eps_B,
        t0=fig_cfg.t0,
        t1=fig_cfg.t1,
        n_frames=fig_cfg.n_frames,
        dt0=fig_cfg.dt0,
        equilibrium_mode=eq_mode,
        objective=fig_cfg.objective,
        log10_eta_min=fig_cfg.log10_eta_min,
        log10_eta_max=fig_cfg.log10_eta_max,
        log10_nu_min=fig_cfg.log10_nu_min,
        log10_nu_max=fig_cfg.log10_nu_max,
    )


def run_metrics_for_eq_mode(
    eta: float,
    nu: float,
    eq_mode: str,
    fig_cfg: FigureScanConfig,
) -> Tuple[float, float, float]:
    """
    Wrapper around _simulate_metrics for a given equilibrium_mode and
    dissipation parameters (eta, nu).

    Returns
    -------
    f_kin, complexity, gamma_fit
    """
    cfg = build_inverse_cfg(fig_cfg, eq_mode)
    f_kin, complexity, gamma_fit, res = _simulate_metrics(
        jnp.asarray(eta, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
        cfg,
    )

    return float(f_kin), float(complexity), float(gamma_fit)


# -----------------------------------------------------------------------------#
# 1) Parameter scans for reachable regions
# -----------------------------------------------------------------------------#


def parameter_scan(fig_cfg: FigureScanConfig, eq_mode: str) -> Dict[str, Any]:
    """
    Perform a (log10_eta, log10_nu) grid scan for a given equilibrium_mode.

    If a file reachable_region_scan_<eq_mode>.npz already exists, it is
    loaded and returned (no simulations re-run). Otherwise we run the scan
    and save the NPZ.
    """
    outdir = Path("outputs/scans")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"reachable_region_scan_{eq_mode}.npz"
    if outpath.exists():
        print(f"[SCAN] Found existing {outpath} – loading instead of re-running.")
        data = dict(np.load(outpath, allow_pickle=True))
        data["eq_mode"] = str(data.get("eq_mode", eq_mode))
        return data

    log10_eta_vals = np.linspace(
        fig_cfg.log10_eta_min, fig_cfg.log10_eta_max, fig_cfg.n_eta
    )
    log10_nu_vals = np.linspace(
        fig_cfg.log10_nu_min, fig_cfg.log10_nu_max, fig_cfg.n_nu
    )

    f_kin_grid = np.zeros((fig_cfg.n_eta, fig_cfg.n_nu))
    C_grid = np.zeros((fig_cfg.n_eta, fig_cfg.n_nu))
    gamma_grid = np.zeros((fig_cfg.n_eta, fig_cfg.n_nu))

    print(f"\n[SCAN] Starting parameter scan for equilibrium_mode='{eq_mode}'")
    print(f"       n_eta={fig_cfg.n_eta}, n_nu={fig_cfg.n_nu}")

    for i, log10_eta in enumerate(log10_eta_vals):
        for j, log10_nu in enumerate(log10_nu_vals):
            eta = 10.0 ** log10_eta
            nu = 10.0 ** log10_nu
            print(
                f"[SCAN] eq={eq_mode:9s}, i={i}/{fig_cfg.n_eta-1}, "
                f"j={j}/{fig_cfg.n_nu-1}, "
                f"log10_eta={log10_eta:.3f}, log10_nu={log10_nu:.3f}"
            )
            f_kin, comp, gamma = run_metrics_for_eq_mode(eta, nu, eq_mode, fig_cfg)
            f_kin_grid[i, j] = f_kin
            C_grid[i, j] = comp
            gamma_grid[i, j] = gamma

    scan_data = {
        "log10_eta_vals": log10_eta_vals,
        "log10_nu_vals": log10_nu_vals,
        "f_kin_grid": f_kin_grid,
        "C_grid": C_grid,
        "gamma_grid": gamma_grid,
        "eq_mode": eq_mode,
    }

    np.savez(outpath, **scan_data)
    print(f"[SAVE] Saved scan data to {outpath}")
    return scan_data


def plot_scan_heatmaps(scan_data: Dict[str, Any]):
    """Make f_kin, C_plasmoid, gamma_fit heatmaps for one equilibrium."""
    log10_eta = scan_data["log10_eta_vals"]
    log10_nu = scan_data["log10_nu_vals"]
    f_kin = scan_data["f_kin_grid"]
    C_grid = scan_data["C_grid"]
    gamma_grid = scan_data["gamma_grid"]
    eq_mode = scan_data["eq_mode"]

    X, Y = np.meshgrid(log10_nu, log10_eta)  # columns = nu, rows = eta

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, f_kin, shading="auto")
    c0 = fig.colorbar(im0, ax=axes[0])
    c0.set_label(r"$f_{\mathrm{kin}}$")
    axes[0].set_xlabel(r"$\log_{10}\nu$")
    axes[0].set_ylabel(r"$\log_{10}\eta$")
    axes[0].set_title(rf"$f_{{\rm kin}}$ for '{eq_mode}'")

    im1 = axes[1].pcolormesh(X, Y, C_grid, shading="auto")
    c1 = fig.colorbar(im1, ax=axes[1])
    c1.set_label(r"$C_{\mathrm{plasmoid}}$")
    axes[1].set_xlabel(r"$\log_{10}\nu$")
    axes[1].set_ylabel(r"$\log_{10}\eta$")
    axes[1].set_title(rf"$C_{{\rm plasmoid}}$ for '{eq_mode}'")

    im2 = axes[2].pcolormesh(X, Y, gamma_grid, shading="auto")
    c2 = fig.colorbar(im2, ax=axes[2])
    c2.set_label(r"$\gamma_{\mathrm{fit}}$")
    axes[2].set_xlabel(r"$\log_{10}\nu$")
    axes[2].set_ylabel(r"$\log_{10}\eta$")
    axes[2].set_title(rf"$\gamma_{{\rm fit}}$ for '{eq_mode}'")

    fig.suptitle(rf"Reachable dissipation space: '{eq_mode}'", fontsize=14)
    figdir = Path("outputs/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    outname = figdir / f"fig_reachable_heatmaps_{eq_mode}.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {outname}")


def plot_reachable_region_plane(
    scan_orig: Dict[str, Any],
    scan_ff: Dict[str, Any],
    objective: Objective,
):
    """Combined reachable region in (f_kin, C_plasmoid) plane."""
    f_o = scan_orig["f_kin_grid"].ravel()
    C_o = scan_orig["C_grid"].ravel()

    f_f = scan_ff["f_kin_grid"].ravel()
    C_f = scan_ff["C_grid"].ravel()

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)

    ax.scatter(
        f_o,
        C_o,
        s=40,
        marker="o",
        edgecolor="none",
        alpha=0.8,
        label="original eq.",
    )
    ax.scatter(
        f_f,
        C_f,
        s=40,
        marker="^",
        edgecolor="none",
        alpha=0.8,
        label="force-free eq.",
    )

    ax.scatter(
        [objective.target_f_kin],
        [objective.target_complexity],
        s=90,
        marker="*",
        color="k",
        label="target behaviour",
    )

    ax.set_xlabel(r"$f_{\mathrm{kin}}$ (late-time kinetic-energy fraction)")
    ax.set_ylabel(r"$C_{\mathrm{plasmoid}}$ (midplane complexity)")
    ax.set_title(r"Reachable $(f_{\rm kin}, C_{\rm plasmoid})$ region")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figdir = Path("outputs/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    outname = figdir / "fig_reachable_region_fkin_Cplasmoid.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {outname}")


def plot_gamma_vs_fkin(scan_orig: Dict[str, Any], scan_ff: Dict[str, Any]):
    """Scatter plot of gamma_fit vs f_kin for both equilibria."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)

    ax.scatter(
        scan_orig["f_kin_grid"].ravel(),
        scan_orig["gamma_grid"].ravel(),
        s=45,
        marker="o",
        alpha=0.9,
        label="original eq.",
    )
    ax.scatter(
        scan_ff["f_kin_grid"].ravel(),
        scan_ff["gamma_grid"].ravel(),
        s=55,
        marker="^",
        alpha=0.9,
        label="force-free eq.",
    )

    ax.set_xlabel(r"$f_{\mathrm{kin}}$")
    ax.set_ylabel(r"$\gamma_{\mathrm{fit}}$")
    ax.set_title("Tearing growth vs kinetic-energy fraction")
    ax.grid(True, alpha=0.3)
    ax.legend()

    figdir = Path("outputs/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    outname = figdir / "fig_gamma_vs_fkin.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {outname}")


# -----------------------------------------------------------------------------#
# 2) Inverse design vs grid search (no optimisation re-run)
# -----------------------------------------------------------------------------#


def flatten_grid(f_grid: np.ndarray, C_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Row-major flatten to define a deterministic 'grid-search order'."""
    return f_grid.ravel(), C_grid.ravel()


def grid_cost_history(
    scan_data: Dict[str, Any],
    f_target: float,
    C_target: float,
    lambda_complexity: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute:
      - cost for each grid point in a fixed order,
      - best-so-far curve,
      - info about the overall best grid point.
    """
    f_grid = scan_data["f_kin_grid"]
    C_grid = scan_data["C_grid"]
    log10_eta_vals = scan_data["log10_eta_vals"]
    log10_nu_vals = scan_data["log10_nu_vals"]

    n_eta, n_nu = f_grid.shape

    flat_f, flat_C = flatten_grid(f_grid, C_grid)
    costs = (flat_f - f_target) ** 2 + lambda_complexity * (flat_C - C_target) ** 2
    best_so_far = np.minimum.accumulate(costs)

    # Find best (global minimum) and map back to (i, j)
    idx_best = int(np.argmin(costs))
    i_best = idx_best // n_nu
    j_best = idx_best % n_nu

    best_info = {
        "f_best": float(f_grid[i_best, j_best]),
        "C_best": float(C_grid[i_best, j_best]),
        "log10_eta_best": float(log10_eta_vals[i_best]),
        "log10_nu_best": float(log10_nu_vals[j_best]),
        "cost_best": float(costs[idx_best]),
    }

    return costs, best_so_far, best_info


def load_inverse_history(
    fig_cfg: FigureScanConfig,
) -> Tuple[Dict[str, np.ndarray] | None, Objective | None]:
    """
    Load inverse-design training history NPZ if present.
    This file must be produced by mhd_tearing_inverse_design.py:

        np.savez(
            f"inverse_design_history_{cfg.equilibrium_mode}.npz",
            **history
        )

    where history has keys:
        "loss", "log10_eta", "log10_nu", "eta", "nu", "f_kin", "complexity".
    """
    pattern = fig_cfg.inverse_history_path()
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(
            f"[INV] WARNING: no history file found for pattern {pattern}; "
            "skipping inverse-design-vs-grid comparison."
        )
        return None, None

    path = matches[-1]
    print(f"[INV] Loading inverse-design training history from {path}")
    npz = np.load(path)
    history = {k: np.array(npz[k]) for k in npz.files}

    # Objective used during training (if present).
    def _scalar(name: str) -> float | None:
        if name not in history:
            return None
        arr = np.array(history[name]).reshape(-1)
        if arr.size == 0:
            return None
        return float(arr[0])

    tf = _scalar("target_f_kin")
    tc = _scalar("target_complexity")
    lam = _scalar("lambda_complexity")
    obj = None
    if tf is not None and tc is not None and lam is not None:
        obj = Objective(target_f_kin=tf, target_complexity=tc, lambda_complexity=lam)

    return history, obj


def plot_inverse_vs_grid_for_mode(
    eq_mode: str,
    scan_data: Dict[str, Any],
    inv_history: Dict[str, np.ndarray],
    objective: Objective,
):
    """
    For a single equilibrium_mode:

      - Plot cost vs number of simulations: grid search vs inverse design.
      - Plot (f_kin, C_plasmoid) reachable points and mark:
          * grid-best point
          * inverse-design final point
          * target

    Inverse-design cost is computed in the same behaviour space as the grid:

        (f - f_target)^2 + lambda * (C - C_target)^2.
    """
    f_grid = scan_data["f_kin_grid"].ravel()
    C_grid = scan_data["C_grid"].ravel()

    # --- Grid search cost history --- #
    costs_grid, best_so_far_grid, best_grid_info = grid_cost_history(
        scan_data,
        objective.target_f_kin,
        objective.target_complexity,
        objective.lambda_complexity,
    )
    n_grid = len(costs_grid)
    steps_grid = np.arange(1, n_grid + 1)

    # --- Inverse design cost history --- #
    f_inv = np.array(inv_history["f_kin"], dtype=float)
    C_inv = np.array(inv_history["complexity"], dtype=float)
    costs_inv = (f_inv - objective.target_f_kin) ** 2 + objective.lambda_complexity * (
        C_inv - objective.target_complexity
    ) ** 2
    steps_inv = np.arange(1, len(costs_inv) + 1)

    f_inv_final = float(f_inv[-1])
    C_inv_final = float(C_inv[-1])

    # --- Make the figure --- #
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)

    # Left: cost vs # simulations
    axes[0].semilogy(
        steps_grid, best_so_far_grid, "o-", label="grid search", linewidth=1.8
    )
    axes[0].semilogy(
        steps_inv, costs_inv, "s-", label="inverse design", linewidth=1.8
    )
    axes[0].set_xlabel("number of simulations")
    axes[0].set_ylabel("behaviour-space cost")
    axes[0].set_title(rf"Cost vs simulations ('{eq_mode}')")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # Right: (f_kin, C_plasmoid) plane
    axes[1].scatter(
        f_grid,
        C_grid,
        s=30,
        alpha=0.5,
        edgecolor="none",
        label="grid samples",
    )
    axes[1].scatter(
        best_grid_info["f_best"],
        best_grid_info["C_best"],
        s=90,
        marker="D",
        color="C0",
        label="grid best",
    )
    axes[1].scatter(
        f_inv_final,
        C_inv_final,
        s=90,
        marker="^",
        color="C1",
        label="inverse design",
    )
    axes[1].scatter(
        [objective.target_f_kin],
        [objective.target_complexity],
        s=100,
        marker="*",
        color="k",
        label="target",
    )

    axes[1].set_xlabel(r"$f_{\mathrm{kin}}$")
    axes[1].set_ylabel(r"$C_{\mathrm{plasmoid}}$")
    axes[1].set_title(
        rf"Solutions in $(f_{{\rm kin}}, C_{{\rm plasmoid}})$ ('{eq_mode}')"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(rf"Inverse design vs grid search ('{eq_mode}')", fontsize=14)
    figdir = Path("outputs/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    outname = figdir / f"fig_inverse_vs_grid_{eq_mode}.png"
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {outname}")


# -----------------------------------------------------------------------------#
# Main orchestration
# -----------------------------------------------------------------------------#


def main():
    fig_cfg = FigureScanConfig()

    # Optional overrides via env (used by CLI wrapper)
    eq_env = os.environ.get("MHX_SCAN_EQ_MODE")
    n_eta_env = os.environ.get("MHX_SCAN_N_ETA")
    n_nu_env = os.environ.get("MHX_SCAN_N_NU")
    if eq_env:
        fig_cfg.inverse_eq_mode = eq_env
        fig_cfg.eq_modes = ("original", "forcefree")
    if n_eta_env:
        fig_cfg.n_eta = int(n_eta_env)
    if n_nu_env:
        fig_cfg.n_nu = int(n_nu_env)

    print("========================================================")
    print(" MHD tearing/plasmoid: inverse-design figure generator ")
    print("========================================================")
    print(fig_cfg)

    # 1) Parameter scans for both equilibria (load if already done)
    scan_orig = parameter_scan(fig_cfg, eq_mode="original")
    scan_ff = parameter_scan(fig_cfg, eq_mode="forcefree")

    # 2) Heatmaps
    plot_scan_heatmaps(scan_orig)
    plot_scan_heatmaps(scan_ff)

    # 3) Combined reachable region and gamma vs f_kin
    plot_reachable_region_plane(scan_orig, scan_ff, fig_cfg.objective)
    plot_gamma_vs_fkin(scan_orig, scan_ff)

    # 4) Inverse design vs grid for the chosen equilibrium, using
    #    pre-saved training history (no optimisation here).
    inv_history, inv_obj = load_inverse_history(fig_cfg)
    if inv_history is not None:
        objective = inv_obj or fig_cfg.objective
        if inv_obj is not None:
            print(
                "[INV] Using objective loaded from inverse-design history: "
                f"target_f_kin={objective.target_f_kin}, "
                f"target_complexity={objective.target_complexity}, "
                f"lambda_complexity={objective.lambda_complexity}"
            )

        if fig_cfg.inverse_eq_mode == "original":
            scan_for_mode = scan_orig
        else:
            scan_for_mode = scan_ff

        plot_inverse_vs_grid_for_mode(
            fig_cfg.inverse_eq_mode, scan_for_mode, inv_history, objective
        )

    print("\n[DONE] All figures generated. You can now:")
    print("  - Inspect outputs/scans/reachable_region_scan_*.npz")
    print("  - Use mhd_tearing_postprocess.py on any NPZ from the scans")
    print("  - Drop outputs/figures/*.png directly into the paper.")


if __name__ == "__main__":
    main()
