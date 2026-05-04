from __future__ import annotations

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics, make_k_arrays
from mhx.solver.diagnostics import compute_Az_from_hat


def main() -> None:
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
        progress=cfg.progress,
        jit=cfg.jit,
        check_finite=cfg.check_finite,
    )

    # Energy plot
    ts = np.array(res["ts"])
    E_kin = np.array(res["E_kin"])
    E_mag = np.array(res["E_mag"])

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(ts, E_kin, label="E_kin")
    ax.plot(ts, E_mag, label="E_mag")
    ax.set_xlabel("t")
    ax.set_ylabel("Energy")
    ax.set_title("Energy time series (FAST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/_static/energy.png", dpi=200)
    plt.close(fig)

    # Az midplane frames and GIF
    B_hat_frames = res["B_hat"]
    kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz)
    ix_mid = cfg.Nx // 2
    iz0 = 0
    frames = []
    for t_index in range(B_hat_frames.shape[0]):
        Az = compute_Az_from_hat(B_hat_frames[t_index], kx, ky)
        Az_mid = np.array(Az[ix_mid, :, iz0])
        y = np.linspace(0.0, cfg.Ly, Az_mid.shape[0], endpoint=False)

        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.plot(y, Az_mid)
        ax.set_xlabel("y")
        ax.set_ylabel("A_z")
        ax.set_title(f"A_z midplane (t={ts[t_index]:.2f})")
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        image = np.array(buf)[..., :3]
        frames.append(image)
        plt.close(fig)

        if t_index == len(ts) - 1:
            # Save final frame as static image
            imageio.imwrite("docs/_static/az_midplane.png", image)

    imageio.mimsave("docs/_static/az_midplane.gif", frames, fps=3)


if __name__ == "__main__":
    main()
