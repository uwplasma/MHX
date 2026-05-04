from __future__ import annotations

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics
from mhx.solver.plugins import LinearDragTerm


def main() -> None:
    cfg = TearingSimConfig.fast("original")
    term = LinearDragTerm(mu=0.1)
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
        terms=[term],
        progress=cfg.progress,
        jit=cfg.jit,
        check_finite=cfg.check_finite,
    )
    print("Ran with LinearDragTerm, final gamma_fit:", float(res["gamma_fit"]))


if __name__ == "__main__":
    main()
