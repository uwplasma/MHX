from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics
from mhx.ml.latent_ode import fit_latent_ode


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
    )

    ts = np.array(res["ts"])
    E_kin = np.array(res["E_kin"])
    E_mag = np.array(res["E_mag"])
    f_kin = E_kin / (E_kin + E_mag + 1e-30)

    y = f_kin[:, None]

    out = fit_latent_ode(ts, y, latent_dim=2, steps=80, lr=1e-2, seed=0)
    params = out["params"]

    import jax
    import jax.numpy as jnp
    import diffrax as dfx

    def latent_rhs(t, z, args):
        _ = t
        return jnp.tanh(z @ args["latent"]["W1"] + args["latent"]["b1"]) @ args["latent"]["W2"] + args["latent"]["b2"]

    term = dfx.ODETerm(latent_rhs)
    solver = dfx.Dopri5()
    sol = dfx.diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=ts[1]-ts[0], y0=params["z0"], args={"latent": params["latent"]}, saveat=dfx.SaveAt(ts=ts))
    zs = np.array(sol.ys)

    def decode(z):
        return jnp.tanh(z @ params["decoder"]["W1"] + params["decoder"]["b1"]) @ params["decoder"]["W2"] + params["decoder"]["b2"]

    y_hat = np.array(jax.vmap(decode)(jnp.asarray(zs)))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(ts, y[:, 0], "o-", label="solver")
    ax.plot(ts, y_hat[:, 0], "s-", label="latent ODE")
    ax.set_xlabel("t")
    ax.set_ylabel("f_kin")
    ax.set_title("Latent ODE fit (FAST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/_static/latent_ode_fit.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
