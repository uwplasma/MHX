#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incompressible pseudo-spectral MHD tearing solver (Harris sheet).

Exports a JAX-based solver, equilibrium initialization, and diagnostics
used across scans and inverse design.
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

from mhx.solver.core import (
    estimate_max_dt,
    make_k_arrays,
    make_dealias_mask,
    project_div_free,
    grad_vec_from_hat,
    directional_derivative_vec,
    energy_from_hat,
    make_mhd_rhs_original,
    make_mhd_rhs_forcefree,
    run_time_integration,
    fkr_gamma,
)
from mhx.solver.equilibria import init_equilibrium
from mhx.solver.diagnostics import (
    compute_Az_from_hat,
    tearing_amplitude,
    tearing_mode_amplitude_from_hat,
    estimate_growth_rate,
    reconnection_rate_from_Az,
    count_local_extrema_1d,
    plasmoid_complexity_metric,
)  # noqa: F401
from mhx.solver.plugins import PhysicsTerm

Array = jnp.ndarray


@dataclass(frozen=True)
class TearingMetrics:
    f_kin: Array
    complexity: Array
    gamma_fit: Array

    @classmethod
    def from_result(cls, res):
        ts = res["ts"]
        E_kin = res["E_kin"]
        E_mag = res["E_mag"]
        Az_final_mid = res["Az_final_mid"]

        T = ts.shape[0]
        i0 = int(0.7 * (T - 1))
        E_kin_tail = E_kin[i0:]
        E_mag_tail = E_mag[i0:]

        E_kin_mean = jnp.mean(E_kin_tail)
        E_mag_mean = jnp.mean(E_mag_tail)
        E_tot_mean = E_kin_mean + E_mag_mean + 1e-30
        f_kin = E_kin_mean / E_tot_mean

        complexity = plasmoid_complexity_metric(Az_final_mid)
        gamma_fit = res["gamma_fit"]

        return cls(f_kin, complexity, gamma_fit)


# -----------------------------------------------------------------------------
# Sweet–Parker and basic equilibrium scalings
# -----------------------------------------------------------------------------

def sweet_parker_metrics(B0: float, a: float, eta: float) -> Dict[str, Array]:
    vA = B0
    S_a = a * vA / eta
    S_a = jnp.asarray(S_a)
    vA = jnp.asarray(vA)
    delta_SP = a / jnp.sqrt(S_a + 1e-30)
    v_in_SP = vA / jnp.sqrt(S_a + 1e-30)
    E_SP = v_in_SP * B0
    return dict(
        vA=vA,
        S_a=S_a,
        delta_SP=delta_SP,
        v_in_SP=v_in_SP,
        E_SP=E_SP,
    )


# -----------------------------------------------------------------------------
# Main driver (JAX core + I/O wrapper)
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Harris-sheet tearing mode MHD solver (pseudo-spectral)."
    )
    p.add_argument("--Nx", type=int, default=52)
    p.add_argument("--Ny", type=int, default=52)
    p.add_argument("--Nz", type=int, default=52)
    p.add_argument("--Lx", type=float, default=2.0 * math.pi)
    p.add_argument("--Ly", type=float, default=2.0 * math.pi)
    p.add_argument("--Lz", type=float, default=2.0 * math.pi)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--a", type=float, default=None,
                   help="current sheet half-width (default Lx/16)")
    p.add_argument("--Bg", type=float, default=0.2, help="guide field B_g")
    p.add_argument("--epsB", type=float, default=1e-2,
                   help="perturbation amplitude in A_z")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=50.0)
    p.add_argument("--n_frames", type=int, default=140)
    p.add_argument("--dt0", type=float, default=None,
                   help="initial dt; if None, estimated via CFL")
    p.add_argument("--outfile", type=str,
                   default="mhd_tearing_solution.npz",
                   help="output .npz file with solution and metadata")
    p.add_argument(
        "--equilibrium-mode",
        type=str,
        choices=["original", "forcefree"],
        default="original",
        help="Choose equilibrium model: 'original' or 'forcefree' (subtract equilibrium force)."
    )
    return p.parse_args()


# --- Core JAX simulation + diagnostics (no disk I/O) ------------------------

def _run_tearing_simulation_and_diagnostics(
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    nu: float,
    eta: float,
    B0: float,
    a: float,
    B_g: float,
    eps_B: float,
    t0: float,
    t1: float,
    n_frames: int,
    dt0: float,
    equilibrium_mode: str = "original",
    terms: list[PhysicsTerm] | None = None,
) -> Dict[str, Any]:
    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
    mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)

    ix0 = 0
    iy1 = 1
    iz0 = 0

    gamma_fkr, S_a, Delta_prime_a = fkr_gamma(B0, a, Ly, eta)

    v0_real, B0_real = init_equilibrium(
        Nx, Ny, Nz, Lx, Ly, Lz,
        B0=B0, a=a, B_g=B_g, eps_B=eps_B
    )

    v0_hat = jnp.fft.fftn(v0_real, axes=(1, 2, 3))
    B0_hat = jnp.fft.fftn(B0_real, axes=(1, 2, 3))

    v0_hat = v0_hat * mask_dealias
    B0_hat = B0_hat * mask_dealias
    v0_hat = project_div_free(v0_hat, kx, ky, kz, k2)
    B0_hat = project_div_free(B0_hat, kx, ky, kz, k2)

    if equilibrium_mode == "forcefree":
        grad_B0 = grad_vec_from_hat(B0_hat, kx, ky, kz)
        strB_v0 = directional_derivative_vec(B0_real, grad_B0)
        Nv0_hat = jnp.fft.fftn(strB_v0, axes=(1, 2, 3)) * mask_dealias
        Nv0_hat = project_div_free(Nv0_hat, kx, ky, kz, k2)

        rhs = make_mhd_rhs_forcefree(
            nu, eta, kx, ky, kz, k2, mask_dealias,
            B0_hat, Nv0_hat, terms=terms
        )
    else:
        rhs = make_mhd_rhs_original(
            nu, eta, kx, ky, kz, k2, mask_dealias, terms=terms
        )

    sol = run_time_integration(
        rhs,
        t0=t0,
        t1=t1,
        dt0=dt0,
        n_frames=n_frames,
        y0=(v0_hat, B0_hat),
    )

    ts = sol.ts
    v_hat_frames, B_hat_frames = sol.ys

    def energy_single(v_hat_t, B_hat_t):
        Ek, Em = energy_from_hat(v_hat_t, B_hat_t, Lx, Ly, Lz)
        return Ek, Em

    E_kin_series, E_mag_series = jax.vmap(energy_single)(v_hat_frames, B_hat_frames)

    v_hat_end = v_hat_frames[-1]
    B_hat_end = B_hat_frames[-1]
    E_kin_end, E_mag_end = energy_from_hat(v_hat_end, B_hat_end, Lx, Ly, Lz)

    sp = sweet_parker_metrics(B0=B0, a=a, eta=eta)

    tearing_amp_series = jax.vmap(
        lambda Bh: tearing_amplitude(Bh, Lx, Ly, Lz)
    )(B_hat_frames)

    mode_amp_series = tearing_mode_amplitude_from_hat(B_hat_frames, ix0, iy1, iz0)
    gamma_fit, lnA_fit, mask_lin = estimate_growth_rate(ts, mode_amp_series, w0=mode_amp_series[0])

    def Az_from_B(Bh):
        return compute_Az_from_hat(Bh, kx, ky)

    Az_frames = jax.vmap(Az_from_B)(B_hat_frames)

    ix_mid = Nx // 2
    iy0 = 0
    iz0_real = 0
    Az_xpt_series = Az_frames[:, ix_mid, iy0, iz0_real]
    E_rec_series = reconnection_rate_from_Az(ts, Az_xpt_series)

    Az_final_mid = Az_frames[-1, ix_mid, :, iz0_real]
    n_plasmoids_final = count_local_extrema_1d(Az_final_mid)
    complexity_final = plasmoid_complexity_metric(Az_final_mid)

    return dict(
        ts=ts,
        v_hat=v_hat_frames,
        B_hat=B_hat_frames,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        nu=nu, eta=eta,
        B0=B0, a=a, B_g=B_g, eps_B=eps_B,
        t0=t0, t1=t1,
        n_frames=n_frames,
        dt0=dt0,
        E_kin=E_kin_series,
        E_mag=E_mag_series,
        E_kin_end=E_kin_end,
        E_mag_end=E_mag_end,
        gamma_FKR=gamma_fkr,
        S_a=S_a,
        Delta_prime_a=Delta_prime_a,
        vA=sp["vA"],
        S_sheet=sp["S_a"],
        delta_SP=sp["delta_SP"],
        v_in_SP=sp["v_in_SP"],
        E_SP=sp["E_SP"],
        S=sp["S_a"],
        mode_amp_series=mode_amp_series,
        tearing_amp_series=tearing_amp_series,
        gamma_fit=gamma_fit,
        lnA_fit=lnA_fit,
        mask_lin=mask_lin,
        Az_xpt_series=Az_xpt_series,
        E_rec_series=E_rec_series,
        n_plasmoids_final=n_plasmoids_final,
        Az_final_mid=Az_final_mid,
        complexity_final=complexity_final,
        ix0=ix0, iy1=iy1, iz0=iz0,
        equilibrium_mode=equilibrium_mode,
    )


def solve_tearing_case(
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    nu: float,
    eta: float,
    B0: float,
    a: float | None,
    B_g: float,
    eps_B: float,
    t0: float,
    t1: float,
    n_frames: int,
    dt0: float | None,
    outfile: str,
    equilibrium_mode: str = "original",
) -> str:
    if a is None:
        a = Lx / 16.0

    print("=== Incompressible pseudo-spectral MHD Parameters ===")
    print(f"Nx,Ny,Nz = {Nx},{Ny},{Nz}")
    print(f"Lx,Ly,Lz = {Lx},{Ly},{Lz}")
    print(f"nu={nu}, eta={eta}")
    print(f"B0={B0}, a={a}, B_g={B_g}, eps_B={eps_B}")
    print(f"t0={t0}, t1={t1}, n_frames={n_frames}")
    print(f"equilibrium_mode = {equilibrium_mode}")
    print("=====================================================")

    default_out = "mhd_tearing_solution.npz"
    if outfile == default_out:
        stem, ext = os.path.splitext(default_out)
        outfile = f"{stem}_{equilibrium_mode}{ext}"
        print(f"[OUT] Using automatic outfile name: {outfile}")

    if dt0 is None:
        kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(Nx, Ny, Nz, Lx, Ly, Lz)
        mask_dealias = make_dealias_mask(Nx, Ny, Nz, NX, NY, NZ)
        v0_real, B0_real = init_equilibrium(
            Nx, Ny, Nz, Lx, Ly, Lz,
            B0=B0, a=a, B_g=B_g, eps_B=eps_B
        )
        v0_hat = jnp.fft.fftn(v0_real, axes=(1, 2, 3))
        B0_hat = jnp.fft.fftn(B0_real, axes=(1, 2, 3))
        v0_hat = v0_hat * mask_dealias
        B0_hat = B0_hat * mask_dealias
        v0_hat = project_div_free(v0_hat, kx, ky, kz, k2)
        B0_hat = project_div_free(B0_hat, kx, ky, kz, k2)

        dt_max = estimate_max_dt(v0_hat, B0_hat, Lx, Ly, Lz, nu, eta)
        print(f"[DT] Estimated dt_max from CFL/diffusion = {dt_max:.3e}")
        dt0 = min(5e-4, 0.5 * dt_max)
    print(f"[DT] Using dt0 = {dt0:.3e}")

    res = _run_tearing_simulation_and_diagnostics(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nu=nu,
        eta=eta,
        B0=B0,
        a=a,
        B_g=B_g,
        eps_B=eps_B,
        t0=t0,
        t1=t1,
        n_frames=n_frames,
        dt0=dt0,
        equilibrium_mode=equilibrium_mode,
    )

    out_np = {}
    for key, val in res.items():
        if isinstance(val, jnp.ndarray):
            out_np[key] = np.array(val)
        else:
            out_np[key] = np.array(val)

    np.savez(outfile, **out_np)
    print(f"[SAVE] Solution + diagnostics saved to {outfile}")
    return outfile


def main():
    args = parse_args()

    solve_tearing_case(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        nu=args.nu,
        eta=args.eta,
        B0=args.B0,
        a=args.a,
        B_g=args.Bg,
        eps_B=args.epsB,
        t0=args.t0,
        t1=args.t1,
        n_frames=args.n_frames,
        dt0=args.dt0,
        outfile=args.outfile,
        equilibrium_mode=args.equilibrium_mode,
    )


if __name__ == "__main__":
    main()