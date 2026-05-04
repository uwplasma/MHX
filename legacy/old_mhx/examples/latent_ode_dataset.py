from __future__ import annotations

import dataclasses
from pathlib import Path
import numpy as np

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics


def generate_dataset(out_path: Path | str = "outputs/datasets/latent_ode_dataset.npz", seed: int = 0) -> Path:
    cfg = TearingSimConfig.fast("original")
    cfg = dataclasses.replace(cfg, t1=0.6, n_frames=10)

    cases = [
        (1e-3, 1e-3),
        (5e-4, 1e-3),
        (1e-3, 5e-4),
        (7e-4, 7e-4),
        (3e-4, 9e-4),
        (9e-4, 3e-4),
    ]

    ys = []
    for eta, nu in cases:
        res = _run_tearing_simulation_and_diagnostics(
            Nx=cfg.Nx,
            Ny=cfg.Ny,
            Nz=cfg.Nz,
            Lx=cfg.Lx,
            Ly=cfg.Ly,
            Lz=cfg.Lz,
            nu=nu,
            eta=eta,
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
        ts = np.array(res["ts"])
        E_kin = np.array(res["E_kin"])
        E_mag = np.array(res["E_mag"])
        f_kin = E_kin / (E_kin + E_mag + 1e-30)
        complexity = np.array(res["complexity_series"])
        y = np.stack([f_kin, complexity], axis=1)
        ys.append(y)

    ys = np.stack(ys, axis=0)
    ts = np.array(ts)
    eta_vals = np.array([c[0] for c in cases])
    nu_vals = np.array([c[1] for c in cases])

    # Train/val/test split along time
    T = ys.shape[1]
    n_train = int(0.7 * T)
    n_val = int(0.15 * T)
    train_mask = np.zeros(T, dtype=bool)
    val_mask = np.zeros(T, dtype=bool)
    test_mask = np.zeros(T, dtype=bool)
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        ts=ts,
        y=ys,
        eta=eta_vals,
        nu=nu_vals,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=seed,
    )
    return out_path


if __name__ == "__main__":
    generate_dataset()
