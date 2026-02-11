from __future__ import annotations

import numpy as np

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics, TearingMetrics


def test_tiny_convergence_f_kin():
    cfg8 = TearingSimConfig.fast("original")
    cfg8 = cfg8.__class__(**{**cfg8.as_dict(), "Nx": 8, "Ny": 8, "t1": 0.2, "n_frames": 4})

    cfg12 = TearingSimConfig.fast("original")
    cfg12 = cfg12.__class__(**{**cfg12.as_dict(), "Nx": 12, "Ny": 12, "t1": 0.2, "n_frames": 4})

    res8 = _run_tearing_simulation_and_diagnostics(
        Nx=cfg8.Nx,
        Ny=cfg8.Ny,
        Nz=cfg8.Nz,
        Lx=cfg8.Lx,
        Ly=cfg8.Ly,
        Lz=cfg8.Lz,
        nu=cfg8.nu,
        eta=cfg8.eta,
        B0=cfg8.B0,
        a=cfg8.a,
        B_g=cfg8.B_g,
        eps_B=cfg8.eps_B,
        t0=cfg8.t0,
        t1=cfg8.t1,
        n_frames=cfg8.n_frames,
        dt0=cfg8.dt0,
        equilibrium_mode=cfg8.equilibrium_mode,
    )
    res12 = _run_tearing_simulation_and_diagnostics(
        Nx=cfg12.Nx,
        Ny=cfg12.Ny,
        Nz=cfg12.Nz,
        Lx=cfg12.Lx,
        Ly=cfg12.Ly,
        Lz=cfg12.Lz,
        nu=cfg12.nu,
        eta=cfg12.eta,
        B0=cfg12.B0,
        a=cfg12.a,
        B_g=cfg12.B_g,
        eps_B=cfg12.eps_B,
        t0=cfg12.t0,
        t1=cfg12.t1,
        n_frames=cfg12.n_frames,
        dt0=cfg12.dt0,
        equilibrium_mode=cfg12.equilibrium_mode,
    )

    f8 = float(np.array(TearingMetrics.from_result(res8).f_kin))
    f12 = float(np.array(TearingMetrics.from_result(res12).f_kin))

    assert np.isfinite(f8)
    assert np.isfinite(f12)
    assert abs(f8 - f12) < 0.5
