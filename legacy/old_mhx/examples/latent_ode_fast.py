from __future__ import annotations

from pathlib import Path
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
        progress=cfg.progress,
        jit=cfg.jit,
        check_finite=cfg.check_finite,
    )

    ts = np.array(res["ts"])
    E_kin = np.array(res["E_kin"])
    E_mag = np.array(res["E_mag"])
    f_kin = E_kin / (E_kin + E_mag + 1e-30)
    complexity = np.array(res["complexity_series"])

    y = np.stack([f_kin, complexity], axis=1)

    import jax
    import jax.numpy as jnp
    import diffrax as dfx

    def predict(params, ts_np):
        def latent_rhs(t, z, args):
            _ = t
            return jnp.tanh(z @ args["latent"]["W1"] + args["latent"]["b1"]) @ args["latent"]["W2"] + args["latent"]["b2"]

        term = dfx.ODETerm(latent_rhs)
        solver = dfx.Dopri5()
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=ts_np[0],
            t1=ts_np[-1],
            dt0=ts_np[1] - ts_np[0],
            y0=params["z0"],
            args={"latent": params["latent"]},
            saveat=dfx.SaveAt(ts=ts_np),
        )
        zs = np.array(sol.ys)

        def decode(z):
            return jnp.tanh(z @ params["decoder"]["W1"] + params["decoder"]["b1"]) @ params["decoder"]["W2"] + params["decoder"]["b2"]

        y_hat_local = np.array(jax.vmap(decode)(jnp.asarray(zs)))
        return y_hat_local

    out = fit_latent_ode(ts, y, latent_dim=2, hidden_dim=32, steps=80, lr=1e-2, seed=0)
    params = out["params"]
    y_hat = predict(params, ts)

    mse = np.mean((y_hat - y) ** 2, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].plot(ts, y[:, 0], "o-", label="solver")
    axes[0].plot(ts, y_hat[:, 0], "s-", label="latent ODE")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("f_kin")
    axes[0].set_title(f"f_kin (MSE={mse[0]:.2e})")
    axes[0].legend()

    axes[1].plot(ts, y[:, 1], "o-", label="solver")
    axes[1].plot(ts, y_hat[:, 1], "s-", label="latent ODE")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("C_plasmoid")
    axes[1].set_title(f"C_plasmoid (MSE={mse[1]:.2e})")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("docs/_static/latent_ode_fit.png", dpi=200)
    plt.close(fig)

    # Ablations (fast)
    ablations = [
        {"name": "latent1", "latent_dim": 1, "hidden_dim": 32, "steps": 60, "lr": 1e-2},
        {"name": "latent2", "latent_dim": 2, "hidden_dim": 32, "steps": 60, "lr": 1e-2},
        {"name": "latent2_wide", "latent_dim": 2, "hidden_dim": 64, "steps": 60, "lr": 1e-2},
    ]
    rows = []
    for cfg_ab in ablations:
        out_ab = fit_latent_ode(
            ts,
            y,
            latent_dim=cfg_ab["latent_dim"],
            hidden_dim=cfg_ab["hidden_dim"],
            steps=cfg_ab["steps"],
            lr=cfg_ab["lr"],
            seed=0,
        )
        y_hat_ab = predict(out_ab["params"], ts)
        mse_ab = np.mean((y_hat_ab - y) ** 2, axis=0)
        rows.append(
            {
                "name": cfg_ab["name"],
                "latent_dim": cfg_ab["latent_dim"],
                "hidden_dim": cfg_ab["hidden_dim"],
                "steps": cfg_ab["steps"],
                "lr": cfg_ab["lr"],
                "mse_f_kin": float(mse_ab[0]),
                "mse_c_plasmoid": float(mse_ab[1]),
            }
        )

    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latent_ode_ablation.json").write_text(
        __import__("json").dumps(rows, indent=2), encoding="utf-8"
    )

    lines = [
        ".. list-table:: Latent ODE ablations (FAST)",
        "   :header-rows: 1",
        "",
        "   * - Name",
        "     - latent_dim",
        "     - hidden_dim",
        "     - steps",
        "     - lr",
        "     - MSE(f_kin)",
        "     - MSE(C_plasmoid)",
    ]
    for row in rows:
        lines.extend(
            [
                f"   * - {row['name']}",
                f"     - {row['latent_dim']}",
                f"     - {row['hidden_dim']}",
                f"     - {row['steps']}",
                f"     - {row['lr']}",
                f"     - {row['mse_f_kin']:.2e}",
                f"     - {row['mse_c_plasmoid']:.2e}",
            ]
        )
    Path("docs/_static/latent_ode_ablation.rst").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
