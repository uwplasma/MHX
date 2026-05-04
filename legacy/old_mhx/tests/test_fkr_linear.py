from __future__ import annotations

import dataclasses
import numpy as np

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics


def test_fkr_gamma_linear_benchmark():
    cfg = TearingSimConfig.fast("original")
    cfg = dataclasses.replace(cfg, t1=0.6, n_frames=10)

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
        progress=False,
        jit=False,
        check_finite=True,
    )

    gamma_fit = float(np.array(res["gamma_fit"]))
    gamma_fkr = float(np.array(res["gamma_FKR"]))

    assert np.isfinite(gamma_fit)
    assert np.isfinite(gamma_fkr)
    assert gamma_fit > 0.0
    ratio = gamma_fit / gamma_fkr
    assert 0.01 < ratio < 100.0
