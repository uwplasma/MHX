from __future__ import annotations

import numpy as np

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics, TearingMetrics


def test_tearing_metrics_from_result_shapes():
    cfg = TearingSimConfig.fast("original")
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

    assert res["ts"].shape[0] == cfg.n_frames
    metrics = TearingMetrics.from_result(res)
    assert np.isfinite(np.array(metrics.f_kin))
    assert np.isfinite(np.array(metrics.complexity))
    assert np.isfinite(np.array(metrics.gamma_fit))
