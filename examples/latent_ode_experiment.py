from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

import jax
import jax.numpy as jnp
import optax
import diffrax as dfx

from mhx.ml.latent_ode import fit_latent_ode, init_mlp, mlp_apply

sys.path.append(str(Path(__file__).resolve().parent))
from latent_ode_dataset import generate_dataset


def _predict_latent(params: dict, ts: np.ndarray) -> np.ndarray:
    def latent_rhs(t, z, args):
        _ = t
        return jnp.tanh(z @ args["latent"]["W1"] + args["latent"]["b1"]) @ args["latent"]["W2"] + args["latent"]["b2"]

    term = dfx.ODETerm(latent_rhs)
    solver = dfx.Dopri5()
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=params["z0"],
        args={"latent": params["latent"]},
        saveat=dfx.SaveAt(ts=ts),
    )
    zs = np.array(sol.ys)

    def decode(z):
        return jnp.tanh(z @ params["decoder"]["W1"] + params["decoder"]["b1"]) @ params["decoder"]["W2"] + params["decoder"]["b2"]

    y_hat = np.array(jax.vmap(decode)(jnp.asarray(zs)))
    return y_hat


def _fit_mlp_time(ts: np.ndarray, y: np.ndarray, train_mask: np.ndarray, hidden: int = 16, steps: int = 300, lr: float = 1e-2):
    t = jnp.asarray(ts)[:, None]
    yj = jnp.asarray(y)
    mask = jnp.asarray(train_mask).astype(bool)

    params = init_mlp(1, hidden, y.shape[1], jax.random.PRNGKey(0))
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(p):
        y_hat = mlp_apply(p, t)
        return jnp.mean((y_hat[mask] - yj[mask]) ** 2)

    @jax.jit
    def step(p, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = opt.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    for _ in range(steps):
        params, opt_state, _ = step(params, opt_state)

    y_hat = np.array(mlp_apply(params, t))
    return y_hat


def _ar1_baseline(y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    y_hat = np.zeros_like(y)
    for j in range(y.shape[1]):
        yj = y[:, j]
        mask = train_mask.copy()
        mask[0] = False
        num = np.sum(yj[1:][mask[1:]] * yj[:-1][mask[1:]])
        den = np.sum(yj[:-1][mask[1:]] ** 2) + 1e-12
        a = num / den
        y_hat[1:, j] = a * yj[:-1]
        y_hat[0, j] = yj[0]
    return y_hat


def main() -> None:
    dataset_path = Path("outputs/datasets/latent_ode_dataset.npz")
    if not dataset_path.exists():
        generate_dataset(dataset_path)

    data = np.load(dataset_path, allow_pickle=True)
    ts = np.array(data["ts"])
    ys = np.array(data["y"])
    eta_vals = np.array(data["eta"])
    nu_vals = np.array(data["nu"])
    train_mask = np.array(data["train_mask"]).astype(bool)
    test_mask = np.array(data["test_mask"]).astype(bool)

    rows = []
    all_pred = []
    all_true = []
    worst_case = {"idx": None, "mse": -np.inf, "y": None, "pred": None}

    for idx in range(ys.shape[0]):
        y = ys[idx]
        out = fit_latent_ode(ts, y, latent_dim=2, hidden_dim=32, steps=100, lr=1e-2, seed=0, train_mask=train_mask)
        y_latent = _predict_latent(out["params"], ts)
        y_ar1 = _ar1_baseline(y, train_mask)
        y_mlp = _fit_mlp_time(ts, y, train_mask, hidden=16, steps=300, lr=1e-2)

        def metrics(y_hat, mask):
            err = y_hat[mask] - y[mask]
            mse = np.mean(err ** 2, axis=0)
            mae = np.mean(np.abs(err), axis=0)
            return mse, mae

        mse_lat, mae_lat = metrics(y_latent, test_mask)
        mse_ar1, mae_ar1 = metrics(y_ar1, test_mask)
        mse_mlp, mae_mlp = metrics(y_mlp, test_mask)

        rows.append(
            {
                "case": idx,
                "eta": float(eta_vals[idx]),
                "nu": float(nu_vals[idx]),
                "latent_mse": mse_lat.tolist(),
                "latent_mae": mae_lat.tolist(),
                "ar1_mse": mse_ar1.tolist(),
                "ar1_mae": mae_ar1.tolist(),
                "mlp_mse": mse_mlp.tolist(),
                "mlp_mae": mae_mlp.tolist(),
            }
        )

        all_pred.append(y_latent[test_mask])
        all_true.append(y[test_mask])
        case_mse = float(np.mean((y_latent[test_mask] - y[test_mask]) ** 2))
        if case_mse > worst_case["mse"]:
            worst_case = {"idx": idx, "mse": case_mse, "y": y, "pred": y_latent}

        if idx == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
            labels = ["latent ODE", "AR(1)", "MLP-time"]
            preds = [y_latent, y_ar1, y_mlp]
            axes[0].plot(ts, y[:, 0], "k-", label="truth")
            for p, lab in zip(preds, labels):
                axes[0].plot(ts, p[:, 0], "--", label=lab)
            axes[0].set_title("f_kin (train/val/test)")
            axes[0].set_xlabel("t")
            axes[0].set_ylabel("f_kin")
            axes[0].legend()

            axes[1].plot(ts, y[:, 1], "k-", label="truth")
            for p, lab in zip(preds, labels):
                axes[1].plot(ts, p[:, 1], "--", label=lab)
            axes[1].set_title("C_plasmoid (train/val/test)")
            axes[1].set_xlabel("t")
            axes[1].set_ylabel("C_plasmoid")
            axes[1].legend()

            fig.tight_layout()
            fig.savefig("docs/_static/latent_ode_experiment.png", dpi=200)
            plt.close(fig)

    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latent_ode_experiment.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        ".. list-table:: Latent ODE experiment (FAST, test split)",
        "   :header-rows: 1",
        "",
        "   * - Case",
        "     - Latent MSE (f_kin, C)",
        "     - AR(1) MSE (f_kin, C)",
        "     - MLP MSE (f_kin, C)",
    ]
    for row in rows:
        lines.extend(
            [
                f"   * - {row['case']}",
                f"     - {row['latent_mse'][0]:.2e}, {row['latent_mse'][1]:.2e}",
                f"     - {row['ar1_mse'][0]:.2e}, {row['ar1_mse'][1]:.2e}",
                f"     - {row['mlp_mse'][0]:.2e}, {row['mlp_mse'][1]:.2e}",
            ]
        )
    static_dir = Path("docs/_static")
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "latent_ode_experiment.rst").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Calibration plot (test split)
    pred = np.vstack(all_pred)
    true = np.vstack(all_true)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for i, label in enumerate(["f_kin", "C_plasmoid"]):
        axes[i].scatter(true[:, i], pred[:, i], s=12, alpha=0.6, label="test")
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        axes[i].plot([lo, hi], [lo, hi], "k--", lw=1.0)
        axes[i].set_title(f"Calibration: {label}")
        axes[i].set_xlabel("truth")
        axes[i].set_ylabel("prediction")
    fig.tight_layout()
    fig.savefig("docs/_static/latent_ode_calibration.png", dpi=200)
    plt.close(fig)

    # Failure case (worst test MSE)
    if worst_case["idx"] is not None:
        y = worst_case["y"]
        y_hat = worst_case["pred"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
        axes[0].plot(ts, y[:, 0], "k-", label="truth")
        axes[0].plot(ts, y_hat[:, 0], "--", label="latent ODE")
        axes[0].set_title("Failure case: f_kin")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("f_kin")
        axes[0].legend()

        axes[1].plot(ts, y[:, 1], "k-", label="truth")
        axes[1].plot(ts, y_hat[:, 1], "--", label="latent ODE")
        axes[1].set_title("Failure case: C_plasmoid")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("C_plasmoid")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig("docs/_static/latent_ode_failure.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
