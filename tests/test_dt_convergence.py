from __future__ import annotations

import numpy as np

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics, TearingMetrics


def test_dt_convergence_f_kin():
    cfg = TearingSimConfig.fast("original")
    cfg = cfg.__class__(**{**cfg.as_dict(), "t1": 0.2, "n_frames": 4})

    res1 = _run_tearing_simulation_and_diagnostics(
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
        progress=False,
        jit=False,
        check_finite=True,
    )
    res2 = _run_tearing_simulation_and_diagnostics(
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
        dt0=cfg.dt0 / 2.0,
        equilibrium_mode=cfg.equilibrium_mode,
        progress=False,
        jit=False,
        check_finite=True,
    )

    f1 = float(np.array(TearingMetrics.from_result(res1).f_kin))
    f2 = float(np.array(TearingMetrics.from_result(res2).f_kin))

    assert np.isfinite(f1)
    assert np.isfinite(f2)
    assert abs(f1 - f2) < 0.5
