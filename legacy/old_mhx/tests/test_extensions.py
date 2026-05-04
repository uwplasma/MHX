from __future__ import annotations

import dataclasses
import numpy as np

from mhx.config import TearingSimConfig
from mhx.solver.plugins import build_terms
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics


def _run_with_terms(term_names: list[str]):
    cfg = TearingSimConfig.fast("original")
    cfg = dataclasses.replace(cfg, t1=0.6, n_frames=8)
    terms = build_terms(term_names, params={"electron_pressure_tensor": {"pe_coef": 1e-2}, "two_fluid_ohm": {"d_h": 1e-2, "pe_coef": 1e-2}})
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
        progress=False,
        jit=False,
        check_finite=True,
    )
    assert np.isfinite(np.array(res["gamma_fit"]))
    assert np.isfinite(np.array(res["complexity_final"]))


def test_electron_pressure_term_runs():
    _run_with_terms(["electron_pressure_tensor"])


def test_two_fluid_term_runs():
    _run_with_terms(["two_fluid_ohm"])
